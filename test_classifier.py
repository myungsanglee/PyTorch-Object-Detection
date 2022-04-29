import argparse
import time
import os

import albumentations
import albumentations.pytorch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining
from pytorch_lightning.plugins import DDPPlugin
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

from dataset.classfication.tiny_imagenet import TinyImageNetDataset
from module.classifier import Classifier
from utils.utility import make_model_name
from utils.module_select import get_model
from utils.yaml_helper import get_configs


def test(cfg):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    input_size = cfg['input_size']

    test_transform = albumentations.Compose([
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    data_loader = DataLoader(
        TinyImageNetDataset(
            path=cfg['data_path'],
            transforms=test_transform,
            is_train=False
        ),
        batch_size=1,
        shuffle=False
    )

    model = get_model(cfg['model'])(in_channels=cfg['in_channels'], num_classes=cfg['num_classes'])

    # if torch.cuda.is_available:
        # model = model.to('cuda')

    model_module =Classifier.load_from_checkpoint(
        checkpoint_path='./saved/darknet19_tiny-imagenet/version_0/checkpoints/last.ckpt',
        model=model,
        cfg=cfg
    )
    model_module.eval()

    # Inference
    for sample in data_loader:
        batch_x, batch_y = sample

        # if torch.cuda.is_available:
            # batch_x = batch_x.cuda()    
        
        before = time.time()
        with torch.no_grad():
            predictions = model_module(batch_x)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        print(f'Label: {batch_y[0]}, Prediction: {torch.argmax(predictions)}')

        # batch_x to img
        if torch.cuda.is_available:
            img = batch_x.cpu()[0].numpy()   
        else:
            img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

        cv2.imshow('Result', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg)
