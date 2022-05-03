import os
import glob
import sys
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from dataset.classfication import augmentations


def aug(image, preprocess, augmentations):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
    Returns:
    mixed: Augmented and mixed image.
    """
    aug_list = augmentations

    ws = np.float32(np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(3):
        image_aug = image.copy()
        # depth = -1 if -1 > 0 else np.random.randint(1, 4)
        depth = 2
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, 1)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, path, preprocess, is_train, augmentations=None):
        super().__init__()
        self.preprocess = preprocess
        self.augmentations = augmentations
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

    def __getitem__(self, index):
        img_file = self.data[index]
        img = Image.open(img_file).convert('RGB')
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]

        if self.augmentations is None:
            return self.preprocess(img), label
        else:
            return aug(img, self.preprocess, self.augmentations), label

    def __len__(self):
        return len(self.data)


class AugMix(pl.LightningDataModule):
    def __init__(self, path, workers, batch_size):
        super().__init__()
        self.path = path
        self.workers = workers
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        augs = augmentations.augmentations_all

        preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0, 1)
            ]
        )

        self.train_dataset = AugMixDataset(
            self.path,
            preprocess,
            True,
            augs
        )

        self.valid_dataset = AugMixDataset(
            self.path,
            preprocess,
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

    augs = augmentations.augmentations_all

    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ]
    )

    traindir = '/home/fssv2/myungsang/datasets/tiny_imagenet/tiny-imagenet-200/'
    train_dataset = AugMixDataset(
        traindir,
        preprocess,
        True,
        augs
    )

    loader = DataLoader(
        train_dataset,
        1,
        False
    )

    for sample in loader:
        batch_x, batch_y = sample

        img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(batch_x.size())
        print(batch_y)

        cv2.imshow('sample', img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
