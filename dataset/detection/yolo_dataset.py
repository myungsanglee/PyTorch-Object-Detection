import sys
sys.path.append('C:/my_github/PyTorch-Object-Detection')

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pytorch_lightning as pl

from dataset.detection.utils import collater, decode_predictions_numpy, non_max_suppression_numpy, get_tagged_img


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
    def __init__(self, train_list, val_list, workers, train_transforms, val_transforms, batch_size, num_classes, num_boxes):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.workers = workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes

    def train_dataloader(self):
        return DataLoader(YoloV1Dataset(self.train_transforms, self.train_list, self.num_classes, self.num_boxes),
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0)

    def val_dataloader(self):
        return DataLoader(YoloV1Dataset(self.val_transforms, self.val_list, self.num_classes, self.num_boxes),
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0)


if __name__ == '__main__':
    """
    Data loader 테스트 코드
    python -m dataset.detection.yolo_format
    """
    import albumentations
    import albumentations.pytorch
    from dataset.detection.utils import visualize

    train_transforms = albumentations.Compose([
        # albumentations.HorizontalFlip(p=0.5),
        # albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(448, 448, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))
    num_classes = 3
    num_boxes = 2
    
    loader = DataLoader(YoloV1Dataset(train_transforms, 'C:/my_github/PyTorch-Object-Detection/data/train.txt', num_classes, num_boxes),
                        batch_size=1, 
                        shuffle=False)

    for batch, sample in enumerate(loader):
        print(sample['image'].shape)
        print(sample['label'].shape)
        
        img = sample['image'][0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        
        label = sample['label'].numpy()
        boxes = non_max_suppression_numpy(decode_predictions_numpy(label, num_classes, num_boxes)[0])
        
        img = get_tagged_img(img, boxes, './data/test.names')
        
        cv2.imshow('test', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()
