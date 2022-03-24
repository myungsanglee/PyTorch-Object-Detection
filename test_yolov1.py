import argparse
import time
# from utils.utility import make_model_name

import albumentations
import albumentations.pytorch
from sklearn.utils import shuffle
# import albumentations.pytorch
# import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import cv2

from utils.module_select import get_model
from utils.yaml_helper import get_train_configs
from module.detector import YoloV1Detector
from models.detector.yolov1 import YoloV1
from dataset.detection.utils import decode_predictions, non_max_suppression, get_tagged_img, DecodeYoloV1
from dataset.detection.yolo_dataset import YoloV1Dataset


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
def test(cfg):
    # Prepare data
    input_size = cfg['input_size']
    
    test_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='yolo', min_visibility=0.1))

    data_loader = DataLoader(
        YoloV1Dataset(
            test_transform,
            cfg['train_list'],
            cfg['num_classes'],
            cfg['num_boxes']
        ),
        batch_size=1,
        shuffle=False
    )

    # Load trained model
    # backbone = get_model(cfg['backbone'])
    backbone = models.vgg16(pretrained=True)
    backbone = nn.Sequential(*list(backbone.features.children()))
    set_parameter_requires_grad(backbone, True)
    
    model = YoloV1(
        backbone=backbone,
        backbone_out_features=512,
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes']
    )
    
    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = YoloV1Detector.load_from_checkpoint(
        checkpoint_path='./saved/yolov1_test/version_0/checkpoints/epoch=475-step=475.ckpt',
        model=model
    )
    model_module.eval()

    yolov1_decoder = DecodeYoloV1(cfg['num_classes'], cfg['num_boxes'])

    # Inference
    for sample in data_loader:
        batch_x = sample['image']
        batch_y = sample['label']

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()
        
        # before = time.time()
        with torch.no_grad():
            predictions = model_module(batch_x)
        # print(f'Inference: {(time.time()-before)*1000}ms')
        
        # batch_x to img
        if torch.cuda.is_available:
            img = batch_x.cpu()[0].numpy()   
        else:
            img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        
        boxes = yolov1_decoder(predictions)

        tagged_img = get_tagged_img(img, boxes, './data/test.names', (0, 255, 0))
        
        true_boxes = yolov1_decoder(batch_y)
        
        tagged_img = get_tagged_img(tagged_img, true_boxes, './data/test.names', (0, 0, 255))

        cv2.imshow('test', tagged_img)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)

    test(cfg)
