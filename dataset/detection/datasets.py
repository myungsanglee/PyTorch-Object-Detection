import sys
import os
import random
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.yolo_utils import collater, get_tagged_img, get_target_boxes, get_tagged_lpr_img
from dataset.detection.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)


class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size, augment=False):
        super().__init__()

        with open(path, 'r') as f:
            self.imgs = f.read().splitlines()
        self.img_size = img_size
        self.augment = augment
        self.albumentations = Albumentations() if augment else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)
        
        # Letterbox
        shape = self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self._get_labels(index)
        
        if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        if self.augment:
            img, labels = random_perspective(img, labels,
                                             degrees=0.0,
                                             translate=0.1,
                                             scale=0.5,
                                             shear=0.0,
                                             perspective=0.0)
        
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)

            # Flip up-down
            if random.random() < 0.0:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < 0.5:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
        
        # Convert labels shape [cid, cx, cy, w, h] to [cx, cy, w, h, cid]
        labels_out = torch.zeros((nl, 5))
        if nl:
            labels_out[:, :4] = torch.from_numpy(labels[:, 1:])
            labels_out[:, 4:] = torch.from_numpy(labels[:, :1])
        
        # Convert
        img = img.astype(np.float32) / 255 # uint8 to float32, 0-255 to 0.0-1.0
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), labels_out

    def _get_labels(self, i):
        path = self.imgs[i]
        label_path = path.replace('.jpg', '.txt')
        boxes = np.zeros((0, 5))
        with open(label_path, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                class_id, cx, cy, w, h = map(float, annot.split(' '))
                # annotation = np.array([[cx, cy, w, h, class_id]])
                annotation = np.array([[class_id, cx, cy, w, h]])
                boxes = np.append(boxes, annotation, axis=0)

        return boxes

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        batch_size = len(img)
        
        max_num_annots = max(annots.shape[0] for annots in label)
        
        if max_num_annots > 0:
            padded_annots = torch.ones((batch_size, max_num_annots, 5)) * -1
            for idx, annot in enumerate(label):
                if annot.shape[0] > 0:
                    padded_annots[idx, :annot.shape[0], :] = annot
                    
        else:
            padded_annots = torch.ones((batch_size, 1, 5)) * -1
        
        return {'img': torch.stack(img, 0), 'annot': padded_annots}


class DataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, input_size, batch_size):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.workers = workers
        self.input_size = input_size
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = LoadImagesAndLabels(
            self.train_list,
            self.input_size,
            True
        )
        
        self.valid_dataset = LoadImagesAndLabels(
            self.val_list,
            self.input_size,
            False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=LoadImagesAndLabels.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=LoadImagesAndLabels.collate_fn
        )


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    path = self.imgs[i]
    im = cv2.imread(path)  # BGR
    assert im is not None, f'Image Not Found {path}'
    h0, w0 = im.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


def load_mosaic(self, index):
    #  4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


if __name__ == '__main__':
    input_size = 224
    # data_list = '/home/fssv2/myungsang/datasets/voc/yolo_format/train.txt'
    # data_list = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    data_list = '/home/fssv2/myungsang/datasets/lpr/val.txt'
    
    # names_list = '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names'
    names_list = '/home/fssv2/myungsang/datasets/lpr/lpr_kr.names'

    train_dataset = LoadImagesAndLabels(
        data_list,
        input_size,
        True
    )
    
    valid_dataset = LoadImagesAndLabels(
        data_list,
        input_size,
        False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=LoadImagesAndLabels.collate_fn
    )

    origin_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=LoadImagesAndLabels.collate_fn
    )

    for train_batch, origin_batch in zip(train_dataloader, origin_dataloader):
        train_x = train_batch['img']
        train_y = train_batch['annot']

        origin_x = origin_batch['img']
        origin_y = origin_batch['annot']

        train_img = train_x[0].numpy()   
        train_img = (np.transpose(train_img, (1, 2, 0))*255.).astype(np.uint8).copy()
        train_img = cv2.cvtColor(train_img, cv2.COLOR_RGB2BGR)

        origin_img = origin_x[0].numpy()   
        origin_img = (np.transpose(origin_img, (1, 2, 0))*255.).astype(np.uint8).copy()
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)


        train_true_boxes = get_target_boxes(train_y, input_size)
        origin_true_boxes = get_target_boxes(origin_y, input_size)
        
        # train_img = get_tagged_img(train_img, train_true_boxes, names_list, (0, 0, 255))
        # origin_img = get_tagged_img(origin_img, origin_true_boxes, names_list, (0, 0, 255))
        
        train_img = get_tagged_lpr_img(train_img, train_true_boxes, names_list, (0, 0, 255))
        origin_img = get_tagged_lpr_img(origin_img, origin_true_boxes, names_list, (0, 0, 255))

        cv2.imshow('Train', train_img)
        cv2.imshow('Original', origin_img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()
    
    
