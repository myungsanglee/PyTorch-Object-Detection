import pytorch_lightning as pl

from utils.module_select import get_optimizer
from module.lr_scheduler import CosineAnnealingWarmUpRestarts
from models.loss.yolov1_loss import YoloV1Loss


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

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss = self.loss_fn(batch['label'], pred)

        self.log('val_loss', loss, logger=True, on_epoch=True)

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
        #     gamma=0.96)

        # return torch.optim.SGD(self.model.parameters(), lr=0.001,
        #                        momentum=0.9, weight_decay=5e-4, nesterov=True)

        # return {"optimizer": optim,
        # "lr_scheduler": {
        # "scheduler": scheduler,
        # 'interval': 'step'}
        # }
        
        return optim
