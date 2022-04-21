import argparse
import time

import albumentations
import albumentations.pytorch
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import cv2

from utils.yaml_helper import get_train_configs
from module.detector import YoloV1Detector
from models.detector.yolov1 import YoloV1
from dataset.detection.utils import get_tagged_img, DecodeYoloV1
from dataset.detection.yolo_dataset import YoloV1Dataset
from utils.module_select import get_model


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
    # vgg16 = models.vgg16(pretrained=True)
    backbone = get_model(cfg['backbone'])
    
    model = YoloV1(
        backbone=backbone,
        backbone_out_channels=backbone(torch.randn((1, 3, cfg['input_size'], cfg['input_size']), dtype=torch.float32)).size()[1],
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes']
    )
    
    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = YoloV1Detector.load_from_checkpoint(
        checkpoint_path='./saved/yolov1_voc/version_0/checkpoints/epoch=804-step=177099.ckpt',
        model=model,
        cfg=cfg
    )
    model_module.eval()

    yolov1_decoder = DecodeYoloV1(cfg['num_classes'], cfg['num_boxes'])

    # Inference
    for sample in data_loader:
        
        for _ in range(10):
            batch_x = sample['image']
            batch_y = sample['label']

            if torch.cuda.is_available:
                batch_x = batch_x.cuda()
            
            before = time.time()
            with torch.no_grad():
                predictions = model_module(batch_x)
            boxes = yolov1_decoder(predictions)
            print(f'Inference: {(time.time()-before)*1000:.2f}ms')
            
            # batch_x to img
            if torch.cuda.is_available:
                img = batch_x.cpu()[0].numpy()   
            else:
                img = batch_x[0].numpy()   
            img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        
            true_boxes = yolov1_decoder(batch_y)
        
        tagged_img = get_tagged_img(img, boxes, cfg['names'], (0, 255, 0))
        tagged_img = get_tagged_img(tagged_img, true_boxes, cfg['names'], (0, 0, 255))

        cv2.imshow('test', tagged_img)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Train config file')
    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)

    test(cfg)
