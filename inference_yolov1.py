import argparse
import time
import os

import albumentations
import albumentations.pytorch
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

from utils.yaml_helper import get_configs
from module.yolov1_detector import YoloV1Detector
from models.detector.yolov1 import YoloV1
from dataset.detection.yolov1_utils import get_tagged_img, DecodeYoloV1
from dataset.detection.yolov1_dataset import YoloV1DataModule
from utils.module_select import get_model


def test(cfg, ckpt):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = YoloV1DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=1,
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes']
    )
    data_module.prepare_data()
    data_module.setup()

    backbone = get_model(cfg['backbone'])
    
    model = YoloV1(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes'],
        in_channels=cfg['in_channels'],
        input_size=cfg['input_size']
    )
    
    if torch.cuda.is_available:
        model = model.cuda()

    model_module = YoloV1Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model,
        cfg=cfg
    )
    model_module.eval()

    yolov1_decoder = DecodeYoloV1(cfg['num_classes'], cfg['num_boxes'], conf_threshold=0.6)

    # Inference
    for sample in data_module.val_dataloader():
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
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)
