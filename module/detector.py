import pytorch_lightning as pl

from utils.module_select import get_optimizer
from models.loss.yolov1_loss import YoloV1Loss
# from module.lr_scheduler import CosineAnnealingWarmUpRestarts


class YoloV1Detector(pl.LightningModule):
    def __init__(self, model, cfg, epoch_length=None):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = YoloV1Loss(cfg['num_classes'], cfg['num_boxes'])

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss = self.loss_fn(batch['label'], pred)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    # def on_validation_epoch_start(self):
    #     self.mAP.reset_accumulators()

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss = self.loss_fn(batch['label'], pred)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        
    # def on_validation_epoch_end(self) -> None:
    #     ap_per_class, mAP = self.mAP.compute_map()
    #     self.log('val_mAP', mAP, on_epoch=True, prog_bar=True, sync_dist=True)
    #     for k, v in ap_per_class.items():
    #         self.log(f'val_AP_{k}', v, on_epoch=True, sync_dist=True)

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

        # return {
        #     "optimizer": optim,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         'interval': 'step'
        #     }
        # }
        
        return optim
