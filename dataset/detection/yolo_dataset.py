import sys
import os
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.yolo_utils import collater, get_tagged_img, get_target_boxes


class YoloDataset(Dataset):
    def __init__(self, transforms, files_list):
        super().__init__()

        with open(files_list, 'r') as f:
            self.imgs = f.read().splitlines()
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self._get_boxes(img_file.replace('.jpg', '.txt'))
        
        transformed = self.transforms(image=img, bboxes=boxes)
        
        return transformed

    def _get_boxes(self, label_path):
        boxes = np.zeros((0, 5))
        with open(label_path, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                class_id, cx, cy, w, h = map(float, annot.split(' '))
                annotation = np.array([[cx, cy, w, h, class_id]])
                boxes = np.append(boxes, annotation, axis=0)

        return boxes

class YoloDataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, input_size, batch_size):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.workers = workers
        self.input_size = input_size
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        train_transforms = A.Compose([
            A.HorizontalFlip(),
            A.CLAHE(),
            A.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1
            ),
            A.RandomResizedCrop(self.input_size, self.input_size, (0.5, 1), (0.4, 1.6)),
            A.Normalize(0, 1),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))

        valid_transform = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(0, 1),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))
        
        self.train_dataset = YoloDataset(
            train_transforms, 
            self.train_list
        )
        
        self.valid_dataset = YoloDataset(
            valid_transform, 
            self.val_list
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater
        )


if __name__ == '__main__':
    input_size = 416
    train_list = '/home/fssv2/myungsang/datasets/voc/yolo_format/train.txt'
    val_list = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'

    train_transforms = A.Compose([
        # A.HorizontalFlip(),
        # A.GaussNoise(var_limit=(100, 150), p=1),
        # A.RGBShift(p=1),
        # A.CLAHE(p=1),
        # A.ColorJitter(
        #     brightness=0.5,
        #     contrast=0.2,
        #     saturation=0.5,
        #     hue=0.1,
        #     p=1
        # ),
        A.Affine(scale=(0.5, 1.5)),
        # A.RandomResizedCrop(input_size, input_size, (0.7, 1)),
        # A.Resize(input_size, input_size, always_apply=True),
        A.Normalize(0, 1),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))

    valid_transform = A.Compose([
        A.Resize(input_size, input_size, always_apply=True),
        A.Normalize(0, 1),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))
    
    train_dataset = YoloDataset(
        train_transforms, 
        train_list
    )
    
    valid_dataset = YoloDataset(
        valid_transform, 
        train_list
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collater
    )

    origin_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collater
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
        
        train_img = get_tagged_img(train_img, train_true_boxes, '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names', (0, 0, 255))
        origin_img = get_tagged_img(origin_img, origin_true_boxes, '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names', (0, 0, 255))

        cv2.imshow('Train', train_img)
        cv2.imshow('Original', origin_img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()