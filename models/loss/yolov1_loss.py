import sys
sys.path.append('C:/my_github/PyTorch-Object-Detection')

import torch
from torch import nn
import numpy as np

from dataset.detection.utils import intersection_over_union


class YoloV1Loss(nn.Module):
    """YoloV1 Loss Function

    Arguments:
      num_classes: Number of classes in the dataset
      num_boxes: Number of boxes to predict
    """
    
    def __init__(self, num_classes=20, num_boxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
        self.batch_size = 0

    def forward(self, y_true, y_pred):
        """
        Arguments:
            y_true (tensor): [batch, grid, grid, num_classes+(5*num_boxes)]
            y_pred (tensor): [batch, grid, grid, num_classes+(5*num_boxes)]
        """
        
        self.batch_size = y_true.shape[0]
        
        # Calculate IoU for the prediction boxes with true boxes
        ious = []
        for idx in torch.arange(self.num_boxes):
            iou = intersection_over_union( # (batch, 7, 7, 1)
                y_true[..., self.num_classes+1:self.num_classes+5],
                y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4]
            )
            ious.append(iou)
        ious = torch.stack(ious) # (num_boxes, batch, 7, 7, 1)
        
        # Get best iou index & one_hot
        best_iou_idx = torch.argmax(ious, dim=0) #(batch, 7, 7, 1)
        best_iou_one_hot = torch.nn.functional.one_hot(best_iou_idx, self.num_boxes).view(-1, 7, 7, self.num_boxes) # (batch, 7, 7, num_boxes)
        
        # Get prediction box & iou
        pred_box = []
        pred_conf = []
        pred_iou = []
        for idx in torch.arange(self.num_boxes):
            pred_box.append(best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(1+(5*idx)):self.num_classes+(1+(5*idx))+4])
            pred_conf.append(best_iou_one_hot[..., idx:idx+1] * y_pred[..., self.num_classes+(5*idx):self.num_classes+(5*idx)+1])
            pred_iou.append(best_iou_one_hot[..., idx:idx+1] * ious[idx])
        pred_box = torch.sum(torch.stack(pred_box), dim=0)
        pred_conf = torch.sum(torch.stack(pred_conf), dim=0)
        pred_iou = torch.sum(torch.stack(pred_iou), dim=0)
        
        # Get true box
        true_box = y_true[..., self.num_classes+1:self.num_classes+5] # (batch, S, S, 4)

        # Get true box confidence(object exist in cell)
        # in paper this is 1obj_ij
        obj = y_true[..., self.num_classes:self.num_classes + 1]  # (batch, S, S, 1)
        noobj = 1 - obj  # (batch, S, S, 1)
        
        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        xy_loss = obj * torch.square(true_box[..., 0:2] - pred_box[..., 0:2])  # (batch, S, S, 2)
        xy_loss = torch.sum(xy_loss)  # scalar value
        # print(f"xy_loss: {xy_loss}")
        
        # Take sqrt of width, height of boxes to ensure that
        wh_loss = obj * torch.square(
            torch.sqrt(true_box[..., 2:4]) - (torch.sign(pred_box[..., 2:4]) * torch.sqrt(torch.abs(pred_box[..., 2:4]) + 1e-6))
        )  # (batch, S, S, 2)
        wh_loss = torch.sum(wh_loss)  # scalar value
        # print(f"wh_loss: {wh_loss}")

        box_loss = xy_loss + wh_loss  # scalar value
        
        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # conf_pred is the confidence score for the bbox with highest IoU
        object_loss = obj * torch.square(pred_iou - pred_conf)  # (batch, S, S, 1)
        object_loss = torch.sum(object_loss)  # scalar value
        # print(f"object_loss: {object_loss}")

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = noobj * torch.square(0 - pred_conf)  # (batch, S, S, 1)
        no_object_loss = torch.sum(no_object_loss)  # scalar value
        # print(f"no_object_loss: {no_object_loss}")


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = obj * torch.square(y_true[..., :self.num_classes] - y_pred[..., :self.num_classes])  # (batch, S, S, C)
        class_loss = torch.sum(class_loss)  # scalar value
        # print(f"class_loss: {class_loss}")

        loss = (self.lambda_coord * box_loss) + \
               object_loss + \
               (self.lambda_noobj * no_object_loss) + \
               class_loss

        return loss


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
