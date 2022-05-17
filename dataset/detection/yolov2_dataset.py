import sys
import os
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import albumentations
import albumentations.pytorch

from dataset.detection.yolov2_utils import collater


class YoloV2Dataset(Dataset):
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

class YoloV2DataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, input_size, batch_size):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.workers = workers
        self.input_size = input_size
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        train_transforms = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1    
            ),
            albumentations.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))

        valid_transform = albumentations.Compose([
            albumentations.Resize(self.input_size, self.input_size, always_apply=True),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))
        
        self.train_dataset = YoloV2Dataset(
            train_transforms, 
            self.train_list
        )
        
        self.valid_dataset = YoloV2Dataset(
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
    data_module = YoloV2DataModule(
        train_list='/home/fssv2/myungsang/datasets/voc/yolo_format/train.txt', 
        val_list='/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt',
        workers=0, 
        input_size=416,
        batch_size=1
    )
    data_module.prepare_data()
    data_module.setup()

    for batch, sample in enumerate(data_module.train_dataloader()):
        print(sample['img'].shape)
        print(sample['annot'].shape)
        
        img = sample['img'][0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        labels = sample['annot'][0].numpy() # [max_num_annots, 5(cx, cy, w, h, cid)]
        for label in labels:
            if label.sum() <= 0:
                continue
            
            x_min = int((label[0] - (label[2]/2.)) * w)
            y_min = int((label[1] - (label[3]/2.)) * h)
            x_max = int((label[0] + (label[2]/2.)) * w)
            y_max = int((label[1] + (label[3]/2.)) * h)

            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # boxes = non_max_suppression_numpy(decode_predictions_numpy(label, num_classes, num_boxes)[0])
        
        # img = get_tagged_img(img, boxes, os.path.join(os.getcwd(), 'data/test.names'), (0, 255, 0))
        
        cv2.imshow('test', img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
