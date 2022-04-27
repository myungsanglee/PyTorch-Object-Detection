import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining
from pytorch_lightning.plugins import DDPPlugin
import torch
import torchvision.models as models
import torchsummary

from dataset.detection.yolo_dataset import YoloV1DataModule
from utils.yaml_helper import get_train_configs
from module.detector import YoloV1Detector
from models.detector.yolov1 import YoloV1
from utils.utility import make_model_name
from utils.module_select import get_model


def add_experimental_callbacks(cfg, train_callbacks):
    options = {
        'SWA': StochasticWeightAveraging(),
        'QAT': QuantizationAwareTraining()
    }
    callbacks = cfg['experimental_options']['callbacks']
    if callbacks:
        for option in callbacks:
            train_callbacks.append(options[option])

    return train_callbacks


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train(cfg):
    data_module = YoloV1DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size'],
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes']
    )

    # vgg16 = models.vgg16(pretrained=True)
    # backbone = vgg16.features
    # # backbone = nn.Sequential(*list(backbone.features.children()))
    # set_parameter_requires_grad(backbone, True)
    
    backbone = get_model(cfg['backbone'])
    
    model = YoloV1(
        backbone=backbone,
        backbone_out_channels=backbone(torch.randn((1, 3, cfg['input_size'], cfg['input_size']), dtype=torch.float32)).size()[1],
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes']
    )
    
    torchsummary.summary(model, (3, cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = YoloV1Detector(
        model=model, 
        cfg=cfg
    )

    # model_module = YoloV1Detector.load_from_checkpoint(
    #     checkpoint_path='./saved/yolov1_test/version_0/checkpoints/epoch=319-step=319.ckpt',
    #     model=model,
    #     cfg=cfg,
    #     epoch_length=data_module.train_dataloader().__len__()
    # )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True,
            every_n_epochs=cfg['save_freq']
        )
    ]

    # callbacks = add_expersimental_callbacks(cfg, callbacks)

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], make_model_name(cfg), default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        # strategy='ddp' if platform.system() != 'Windows' else None,
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )
    
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='Train config file')
    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)

    train(cfg)
