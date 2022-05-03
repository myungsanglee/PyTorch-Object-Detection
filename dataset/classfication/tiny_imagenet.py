import os
import sys  
import glob
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations
import albumentations.pytorch
import cv2
import numpy as np

from dataset.augmentation import AugMix


class TinyImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train):
        super().__init__()
        self.transforms = transforms
        self.is_train = is_train
        with open(path + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()

        if is_train:
            self.data = glob.glob(path + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)
        else:
            self.data = glob.glob(path + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(path + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]

        transformed = self.transforms(image=img)['image']
        return transformed, label


class TinyImageNet(pl.LightningDataModule):
    def __init__(self, path, workers, batch_size):
        super().__init__()
        self.path = path
        self.workers = workers
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        augs = [
            albumentations.HorizontalFlip(always_apply=True),
            albumentations.Blur(always_apply=True),
            albumentations.OneOf(
                [albumentations.ShiftScaleRotate(always_apply=True),
                albumentations.GaussNoise(always_apply=True)]
            ),
            albumentations.Cutout(always_apply=True),
            albumentations.PiecewiseAffine(always_apply=True)
        ]

        train_transforms = albumentations.Compose([
            AugMix(width=3, depth=1, alpha=.2, p=1., augmentations=augs),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        valid_transform = albumentations.Compose([
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        self.train_dataset = TinyImageNetDataset(
            self.path,
            train_transforms,
            True
        )

        self.valid_dataset = TinyImageNetDataset(
            self.path,
            valid_transform,
            False
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

    augs = [albumentations.HorizontalFlip(always_apply=True),
        albumentations.Blur(always_apply=True),
        albumentations.OneOf(
        [albumentations.ShiftScaleRotate(always_apply=True),
        albumentations.GaussNoise(always_apply=True)]
        ),
        albumentations.Cutout(always_apply=True),
        albumentations.PiecewiseAffine(always_apply=True)]

    test_transform = albumentations.Compose([
        AugMix(width=3, depth=1, alpha=.2, p=1., augmentations=augs),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    loader = DataLoader(TinyImageNetDataset(
                            '/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200/',
                            test_transform,
                            True
                        ),
                        batch_size=1,
                        shuffle=True
    )

    label_name_path = '/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200/tiny-imagenet.names'
    with open(label_name_path, 'r') as f:
        label_name_list = f.read().splitlines()

    for sample in loader:
        batch_x, batch_y = sample

        img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

        print(batch_x.size())
        print(f'label: {label_name_list[batch_y[0]]}')

        cv2.imshow('sample', img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
