import pytorch_lightning as pl

from utils.module_select import get_optimizer, get_scheduler
from models.loss.yolov3_loss import YoloV3Loss, YoloV3LossV2
from dataset.detection.yolov3_utils import MeanAveragePrecision


class YoloV3Detector(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        # self.loss_fn = YoloV3Loss(cfg['num_classes'], cfg['anchors'], cfg['input_size'])
        self.loss_fn = YoloV3LossV2(cfg['num_classes'], cfg['anchors'], cfg['input_size'])
        self.map_metric = MeanAveragePrecision(cfg['num_classes'], cfg['anchors'], cfg['input_size'], cfg['conf_threshold'])

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        p3, p4, p5 = self.model(batch['img'])
        
        loss = self.loss_fn([p3, p4, p5], batch['annot'])
        
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_start(self):
        self.map_metric.reset_states()

    def validation_step(self, batch, batch_idx):
        p3, p4, p5 = self.model(batch['img'])
        
        loss = self.loss_fn((p3, p4, p5), batch['annot'])

        self.log('val_loss', loss, prog_bar=True, logger=True)

        self.map_metric.update_state(batch['annot'], [p3, p4, p5])        

    def on_validation_epoch_end(self):
        map = self.map_metric.result()
        self.log('val_mAP', map, prog_bar=True, logger=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        try:
            scheduler = get_scheduler(
                cfg['scheduler'],
                optim,
                **cfg['scheduler_options']
            )

            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            } 
        
        except KeyError:
            return optim
