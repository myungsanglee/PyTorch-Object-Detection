from tabnanny import verbose
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from utils.module_select import get_optimizer
from models.loss.yolov1_loss import YoloV1Loss
# from module.lr_scheduler import CosineAnnealingWarmUpRestarts
from utils.metric import MeanAveragePrecision


class YoloV1Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = YoloV1Loss(cfg['num_classes'], cfg['num_boxes'])
        self.map_metric = MeanAveragePrecision(cfg['num_classes'], cfg['num_boxes'])

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss = self.loss_fn(batch['label'], pred)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_start(self):
        self.map_metric.reset_states()

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss = self.loss_fn(batch['label'], pred)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        self.map_metric.update_state(batch['label'], pred)        
        
    def on_validation_epoch_end(self) -> None:
        map = self.map_metric.result()
        self.log('val_mAP', map, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        epoch_length = self.hparams.epoch_length
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )

        # scheduler = CosineAnnealingWarmUpRestarts(
        #     optim,
        #     epoch_length*4,
        #     T_mult=2,
        #     eta_max=cfg['optimizer_options']['lr'],
        #     T_up=epoch_length,
        #     gamma=0.96
        # )
        
        epochs = cfg['epochs']
        scheduler = MultiStepLR(optim, milestones=[int(epochs*0.8), int(epochs*0.9)], gamma=0.1)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }
        
        # return optim
