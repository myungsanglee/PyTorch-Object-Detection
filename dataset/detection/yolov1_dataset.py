import sys
import os
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.detection.yolov1_utils import decode_predictions_numpy, non_max_suppression_numpy, get_tagged_img, get_target_boxes


class YoloV1Dataset(Dataset):
    def __init__(self, transforms, files_list, num_classes, num_boxes):
        super().__init__()

        with open(files_list, 'r') as f:
            self.imgs = f.read().splitlines()
        self.transforms = transforms
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid = 7
        self.output_shape = (self.grid, self.grid, self.num_classes + (self.num_boxes*5))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = self._get_boxes(img_file.replace('.jpg', '.txt'))
        
        transformed = self.transforms(image=img, bboxes=boxes)
        
        label = self._get_labels(transformed['bboxes'])
        
        return {'image': transformed['image'], 'label': label}
    
    def _get_labels(self, boxes):
        # labels matrix = (C + B*5)*S*S
        labels_matrix = np.zeros(self.output_shape)

        for box in boxes:
            # Get Class index, bbox info
            cls = int(box[-1])
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]

            # Start from grid position and calculate x, y
            grid_x = int(self.grid * cx)
            grid_y = int(self.grid * cy)
            x = self.grid * cx - grid_x
            y = self.grid * cy - grid_y

            if labels_matrix[grid_y, grid_x, self.num_classes] == 0: # confidence
                labels_matrix[grid_y, grid_x, cls] = 1 # class
                labels_matrix[grid_y, grid_x, self.num_classes+1:self.num_classes+5] = [x, y, w, h]
                labels_matrix[grid_y, grid_x, self.num_classes] = 1 # confidence

        return labels_matrix

    def _get_boxes(self, label_path):
        boxes = np.zeros((0, 5))
        with open(label_path, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                class_id, cx, cy, w, h = map(float, annot.split(' '))
                annotation = np.array([[cx, cy, w, h, class_id]])
                boxes = np.append(boxes, annotation, axis=0)

        return boxes

class YoloV1DataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, input_size, batch_size, num_classes, num_boxes):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.workers = workers
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        
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
        
        self.train_dataset = YoloV1Dataset(
            train_transforms, 
            self.train_list, 
            self.num_classes, 
            self.num_boxes
        )
        
        self.valid_dataset = YoloV1Dataset(
            valid_transform, 
            self.val_list, 
            self.num_classes, 
            self.num_boxes
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )


if __name__ == '__main__':
    data_module = YoloV1DataModule(
        train_list='/home/fssv2/myungsang/datasets/voc/yolo_format/train.txt', 
        val_list='/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt',
        workers=0, 
        input_size=448,
        batch_size=1,
        num_classes=20,
        num_boxes=2
    )
    data_module.prepare_data()
    data_module.setup()

    for batch, sample in enumerate(data_module.train_dataloader()):
        print(sample['image'].shape)
        print(sample['label'].shape)
        
        img = sample['image'][0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # label = sample['label'].numpy()
        # boxes = non_max_suppression_numpy(decode_predictions_numpy(label, 20, 2)[0])
        
        boxes = get_target_boxes(sample['label'], 20, 2, 448)
        # print(boxes)
        
        img = get_tagged_img(img, boxes, '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names', (0, 255, 0))
        
        cv2.imshow('test', img)
        key = cv2.waitKey(0)
        if key == 27:
            flag = True
            break
        
    cv2.destroyAllWindows()