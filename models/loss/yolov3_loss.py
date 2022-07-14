import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn

from dataset.detection.yolov3_utils import bbox_iou


class YoloV3Loss(nn.Module):
    """YoloV3 Loss Function

    Arguments:
      num_classes (int): Number of classes in the dataset
      anchors (List): Anchors of a specific dataset, [num_anchors, 2(anchor_w, anchor_h)]
      input_size (int): input size of Model (image input size)
    """
    
    def __init__(self, num_classes, anchors, input_size):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_size = input_size

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_obj = 1
        self.lambda_noobj = 1
        self.lambda_coord = 5
        self.lambda_class = 1
        
        self.ignore_threshold = 0.7

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, num_anchors*(5+num_classes), layer_h, layer_w]
            target (tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            
        Returns:
            loss (float): total loss values
        """
        batch_size, _, layer_h, layer_w = input.size()
        scaled_anchors = [[anchor_w * (layer_w/self.input_size), anchor_h * (layer_h/self.input_size)] for anchor_w, anchor_h in self.anchors]
        # [batch_size, num_anchors, 5+num_classes, layer_h, layer_w] to [batch_size, num_anchors, layer_h, layer_w, 5+num_classes]
        prediction = input.view(batch_size, len(self.anchors), -1, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.exp(prediction[..., 2])
        h = torch.exp(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.encode_target(target, self.num_classes, scaled_anchors, layer_w, layer_h, self.ignore_threshold)
        if prediction.is_cuda:
            mask = mask.cuda()
            noobj_mask = noobj_mask.cuda()
            tx = tx.cuda()
            ty = ty.cuda()
            tw = tw.cuda()
            th = th.cuda()
            tconf = tconf.cuda()
            tcls = tcls.cuda()

        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #
        loss_x = self.mse_loss(x * mask, tx * mask)
        loss_y = self.mse_loss(y * mask, ty * mask)
        loss_w = self.mse_loss(w * mask, tw * mask)
        loss_h = self.mse_loss(h * mask, th * mask)
        box_loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        object_loss = self.lambda_obj * self.bce_loss(conf * mask, tconf)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.lambda_noobj * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.lambda_class * self.bce_loss(pred_cls[mask==1], tcls[mask==1])

        loss = (box_loss + object_loss + no_object_loss + class_loss) * batch_size

        return loss


    def encode_target(self, target, num_classes, scaled_anchors, layer_w, layer_h, ignore_threshold):
        """YoloV3 Loss Function

        Arguments:
            target (Tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            num_classes (int): Number of classes in the dataset
            scaled_anchors (List): Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
            layer_w (int): size of predictions width
            layer_h (int): size of predictions height
            ignore_threshold (float): float value of ignore iou
        
        Retruns:
            mask (Tensor): Objectness Mask Tensor, [batch_size, num_anchors, layer_h, layer_w]
            noobj_mask (Tensor): No Objectness Mask Tensor, [batch_size, num_anchors, layer_h, layer_w]
            tx (Tensor): Ground Truth X, [batch_size, num_anchors, layer_h, layer_w]
            ty (Tensor): Ground Truth Y, [batch_size, num_anchors, layer_h, layer_w]
            tw (Tensor): Ground Truth W, [batch_size, num_anchors, layer_h, layer_w]
            th (Tensor): Ground Truth H, [batch_size, num_anchors, layer_h, layer_w]
            tconf (Tensor): Ground Truth Confidence Score, [batch_size, num_anchors, layer_h, layer_w]
            tcls (Tensor): Ground Truth Class index, [batch_size, num_anchors, layer_h, layer_w, num_classes]
        """
        batch_size = target.size(0)
        num_anchors = len(scaled_anchors)
        
        mask = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        noobj_mask = torch.ones(batch_size, num_anchors, layer_h, layer_w)
        tx = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        ty = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tw = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        th = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tconf = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tcls = torch.zeros(batch_size, num_anchors, layer_h, layer_w, num_classes)

        for b in torch.arange(batch_size):
            for t in torch.arange(target.size(1)):
                if target[b, t].sum() <= 0:
                    continue
                gx = target[b, t, 0] * layer_w
                gy = target[b, t, 1] * layer_h
                gw = target[b, t, 2] * layer_w
                gh = target[b, t, 3] * layer_h
                gi = int(gx)
                gj = int(gy)

                gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0) # [1, 4]
                anchors_box = torch.cat([torch.zeros((num_anchors, 2), dtype=torch.float32), torch.FloatTensor(scaled_anchors)], 1) # [num_anchors, 4]

                calc_iou = bbox_iou(gt_box, anchors_box, x1y1x2y2=True) # [num_anchors, 1]
                calc_iou = calc_iou.squeeze(dim=-1) # [num_anchors]
                
                noobj_mask[b, calc_iou > ignore_threshold, gj, gi] = 0
                best_n = torch.argmax(calc_iou)
                mask[b, best_n, gj, gi] = 1
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = gw/scaled_anchors[best_n][0]
                th[b, best_n, gj, gi] = gh/scaled_anchors[best_n][1]
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1
                
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls    
