import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
import numpy as np

from utils.yolo_utils import bbox_iou


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class YoloV1Loss(nn.Module):
    """YoloV1 Loss Function

    Arguments:
      num_classes (int): Number of classes in the dataset
      num_boxes (int): Number of boxes to predict
    """
    def __init__(self, num_classes, num_boxes):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_obj = 5
        self.lambda_noobj = 1
        self.lambda_coord = 1
        self.lambda_class = 1
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, 7 * 7 * (num_boxes*5 + num_classes)]
            target (tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            
        Returns:
            loss (float): total loss values
        """
        y_pred = torch.sigmoid(input.view(-1, 7, 7, self.num_boxes*5 + self.num_classes))
        batch_size, layer_h, layer_w, _ = y_pred.size()
        
        y_true = self._encode_target(target, self.num_classes, self.num_classes, layer_w, layer_h)
        if y_pred.is_cuda:
            y_true = y_true.cuda()
        
        # Calculate IoU for the prediction boxes with true boxes
        ious = []
        for idx in torch.arange(self.num_boxes):
            iou = bbox_iou( # (batch, 7, 7, 1)
                y_true[..., self.num_classes+1:self.num_classes+5],
                y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4]
            )
            ious.append(iou)
        ious = torch.stack(ious) # (num_boxes, batch, 7, 7, 1)
        # Get best iou index & one_hot
        best_iou_idx = torch.argmax(ious, dim=0) #(batch, 7, 7, 1)
        best_iou_one_hot = torch.nn.functional.one_hot(best_iou_idx, self.num_boxes).view(-1, 7, 7, self.num_boxes) # (batch, 7, 7, num_boxes)
        
        # Get prediction info
        pbox = []
        pconf = []
        piou = []
        for idx in torch.arange(self.num_boxes):
            pbox.append(best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4])
            pconf.append(best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(5*idx):self.num_classes+(5*idx)+1])
            piou.append(best_iou_one_hot[..., idx:idx+1] * ious[idx])
        pbox = torch.sum(torch.stack(pbox), dim=0)
        pconf = torch.sum(torch.stack(pconf), dim=0)
        piou = torch.sum(torch.stack(piou), dim=0)
        pcls = y_pred[..., :self.num_classes]
        
        # Get true info
        mask = y_true[..., self.num_classes:self.num_classes + 1]  # (batch, S, S, 1)
        noobj_mask = 1 - mask
        tbox = y_true[..., self.num_classes+1:self.num_classes+5] # (batch, S, S, 4)
        tcls = y_true[..., :self.num_classes]
        
        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #
        box_loss = self.lambda_coord * self.mse_loss(pbox*mask, tbox)
        # print(f'box_loss: {box_loss}')
        
        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        object_loss = self.lambda_obj * self.mse_loss(pconf * mask, piou)
        # print(f"object_loss: {object_loss}")

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.lambda_noobj * self.mse_loss(pconf * noobj_mask, noobj_mask * 0)
        # print(f"no_object_loss: {no_object_loss}")

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.lambda_class * self.mse_loss(pcls[mask.squeeze(dim=-1)==1], tcls[mask.squeeze(dim=-1)==1])
        # print(f"class_loss: {class_loss}")

        loss = (box_loss + object_loss + no_object_loss + class_loss) / batch_size
        
        return loss

    def _encode_target(self, target, num_classes, num_boxes, layer_w, layer_h):
        """
        Arguments:
            target (Tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            num_classes (int): Number of classes in the dataset
            num_boxes (int): Number of boxes to predict
            layer_w (int): size of predictions width
            layer_h (int): size of predictions height
        
        Retruns:
            y_true (Tensor): Ground Truth Tensor, [batch_size, layer_h, layer_w, num_boxes*5 + num_classes]
        """
        batch_size = target.size(0)
        
        y_true = torch.zeros(batch_size, layer_h, layer_w, num_boxes*5 + num_classes)
        
        for b in torch.arange(batch_size):
            for t in torch.arange(target.size(1)):
                if target[b, t].sum() <= 0:
                    continue
                gx = target[b, t, 0] * layer_w
                gy = target[b, t, 1] * layer_h
                # gw = target[b, t, 2] * layer_w
                # gh = target[b, t, 3] * layer_h
                gi = int(gx)
                gj = int(gy)
                
                if y_true[b, gj, gi, self.num_classes] == 0:
                    y_true[b, gj, gi, int(target[b, t, 4])] = 1 # class
                    y_true[b, gj, gi, self.num_classes+1:self.num_classes+5] = torch.tensor([gx - gi, gy - gj, target[b, t, 2], target[b, t, 3]])
                    y_true[b, gj, gi, self.num_classes] = 1 # confidence
                    
        return y_true


if __name__ == "__main__":
    y_true = np.zeros((1, 7, 7, 13))
    y_true[:, 0, 0, 2] = 1 # class
    y_true[:, 0, 0, 3] = 1 # confidence
    y_true[:, 0, 0, 4:8] = (0.5, 0.5, 0.1, 0.1)
    print("y_true:\n{}".format(y_true))

    y_pred = np.zeros((1, 7, 7, 13))
    y_pred[:, 0, 0, 2] = 0.6  # class
    y_pred[:, 0, 0, 3] = 0.7  # confidence
    y_pred[:, 0, 0, 4:8] = (0.49, 0.49, 0.09, 0.09)
    y_pred[:, 0, 0, 9] = 0.4  # confidence
    y_pred[:, 0, 0, 9:13] = (0.45, 0.45, 0.09, 0.09)
    print("y_pred:\n{}".format(y_pred))

    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)

    loss = YoloV1Loss(num_classes=3, num_boxes=2)
    print(loss(y_true, y_pred))
