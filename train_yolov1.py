import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining
from pytorch_lightning.plugins import DDPPlugin
import torchsummary

from dataset.detection.yolov1_dataset import YoloV1DataModule
from module.yolov1_detector import YoloV1Detector
from models.detector.yolov1 import YoloV1
from utils.utility import make_model_name
from utils.module_select import get_model
from utils.yaml_helper import get_configs


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

    backbone = get_model(cfg['backbone'])
    
    model = YoloV1(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_boxes=cfg['num_boxes'],
        in_channels=cfg['in_channels'],
        input_size=cfg['input_size']
    )
    
    torchsummary.summary(model, (cfg['in_channels'], cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = YoloV1Detector(
        model=model, 
        cfg=cfg
    )

    # model_module = YoloV1Detector.load_from_checkpoint(
    #     checkpoint_path='./saved/yolov1_test/version_0/checkpoints/last.ckpt',
    #     model=model,
    #     cfg=cfg
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
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )
    
    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    train(cfg)
