import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torchsummary

from utils.yaml_helper import get_configs
from module.yolov1_detector import YoloV1Detector
from models.detector.yolov1 import YoloV1
from dataset.detection.yolov1_dataset import YoloV1DataModule
from utils.module_select import get_model


def test(cfg, ckpt):
    data_module = YoloV1DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size'],
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes']
    )

    backbone = get_model(cfg['backbone'])
    
    model = YoloV1(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes'],
        in_channels=cfg['in_channels'],
        input_size=cfg['input_size']
    )
    
    torchsummary.summary(model, (cfg['in_channels'], cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = YoloV1Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )

    trainer = pl.Trainer(
        logger=False,
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
    )
    
    trainer.validate(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)
