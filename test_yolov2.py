import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torchsummary
from torchinfo import summary
import timm

from utils.yaml_helper import get_configs
from module.yolov2_detector import YoloV2Detector
from models.detector.yolov2 import YoloV2, Resnet34YoloV2
from dataset.detection.yolov2_dataset import YoloV2DataModule
from utils.module_select import get_model


def test(cfg, ckpt):
    data_module = YoloV2DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )

    if cfg['backbone'] == 'darknet19':

        backbone = get_model(cfg['backbone'])(pretrained=cfg['backbone_pretrained'], devices=cfg['devices'])
        
        model = YoloV2(
            backbone_module_list=backbone.get_features_module_list(),
            num_classes=cfg['num_classes'],
            num_anchors=len(cfg['scaled_anchors'])
        )
        
    else:
        backbone_feature_module = timm.create_model('resnet34', pretrained=True, features_only=True, out_indices=[3, 4])
    
        model = Resnet34YoloV2(
            backbone_feature_module=backbone_feature_module,
            num_classes=cfg['num_classes'],
            num_anchors=len(cfg['scaled_anchors'])
        )
    
    # torchsummary.summary(model, (cfg['in_channels'], cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')
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
