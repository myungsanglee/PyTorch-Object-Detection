import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from torchinfo import summary

from utils.yaml_helper import get_configs
from module.yolov2_detector import YoloV2Detector
from models.detector.yolov2 import YoloV2
from dataset.detection.yolo_dataset import YoloDataModule
from utils.module_select import get_model


def test(cfg, ckpt):
    data_module = YoloDataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )
    
    backbone_features_module = get_model(cfg['backbone'])(
        pretrained=cfg['backbone_pretrained'], 
        devices=cfg['devices'],
        features_only=True,
        out_indices=[4, 5]
    )
    
    model = YoloV2(
        backbone_features_module=backbone_features_module,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )
    
    summary(model, input_size=(1, cfg['in_channels'], cfg['input_size'], cfg['input_size']), device='cpu')

    model_module = YoloV2Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )

    trainer = pl.Trainer(
        logger=False,
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None
    )
    
    trainer.validate(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoints file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)
    