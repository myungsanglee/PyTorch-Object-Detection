import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn

from dataset.detection.yolov2_utils import bbox_iou


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class YoloV2Loss(nn.Module):
    """YoloV2 Loss Function

    Arguments:
      num_classes (int): Number of classes in the dataset
      scaled_anchors (List): Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
    """
    
    def __init__(self, num_classes, scaled_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.scaled_anchors = scaled_anchors

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_obj = 10
        self.lambda_noobj = 1
        self.lambda_coord = 10
        self.lambda_class = 1
        
        self.ignore_threshold = 0.5
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        
        # self.positive_class_target, self.negative_class_target = smooth_BCE()

    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, num_anchors*(5+num_classes), 13, 13]
            target (tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            
        Returns:
            loss (float): total loss values
        """
        batch_size, _, layer_h, layer_w = input.size()
        
        # [batch_size, num_anchors, 5+num_classes, layer_h, layer_w] to [batch_size, num_anchors, layer_h, layer_w, 5+num_classes]
        prediction = input.view(batch_size, len(self.scaled_anchors), -1, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()

        px = torch.sigmoid(prediction[..., 0])
        py = torch.sigmoid(prediction[..., 1])
        pw = torch.exp(prediction[..., 2])
        ph = torch.exp(prediction[..., 3])
        pconf = torch.sigmoid(prediction[..., 4])
        pcls = torch.sigmoid(prediction[..., 5:])
        
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.encode_target(target, self.num_classes, self.scaled_anchors, layer_w, layer_h, self.ignore_threshold)
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
        loss_x = self.mse_loss(px * mask, tx * mask)
        loss_y = self.mse_loss(py * mask, ty * mask)
        loss_w = self.mse_loss(pw * mask, tw * mask)
        loss_h = self.mse_loss(ph * mask, th * mask)
        box_loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
        # print(f'box_loss: {box_loss}')

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        object_loss = self.lambda_obj * self.bce_loss(pconf * mask, tconf)
        # print(f'object_loss: {object_loss}')
        
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.lambda_noobj * self.bce_loss(pconf * noobj_mask, noobj_mask * 0)
        # print(f'no_object_loss: {no_object_loss}')
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.lambda_class * self.bce_loss(pcls[mask==1], tcls[mask==1])
        # class_loss = self.lambda_class * self.bce_loss(pred_cls * mask.unsqueeze(dim=-1), tcls * mask.unsqueeze(dim=-1))
        # print(f'class_loss: {class_loss}\n')

        loss = (box_loss + object_loss + no_object_loss + class_loss) * batch_size

        return loss


    def encode_target(self, target, num_classes, scaled_anchors, layer_w, layer_h, ignore_threshold):
        """YoloV2 Loss Function

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
                
                # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf
                # tcls[b, best_n, gj, gi, :] = self.negative_class_target
                # tcls[b, best_n, gj, gi, int(target[b, t, 4])] = self.positive_class_target

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls    


class YoloV2LossV2(nn.Module):
    """YoloV2 Loss Function

    Arguments:
      num_classes (int): Number of classes in the dataset
      scaled_anchors (List): Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
    """
    
    def __init__(self, num_classes, scaled_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.scaled_anchors = scaled_anchors

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_obj = 10
        self.lambda_noobj = 1
        self.lambda_coord = 0.05
        self.lambda_class = 1
        
        self.ignore_threshold = 0.5
        
        # self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        
    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, num_anchors*(5+num_classes), 13, 13]
            target (tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            
        Returns:
            loss (float): total loss values
        """
        batch_size, _, layer_h, layer_w = input.size()
        
        # [batch_size, num_anchors, 5+num_classes, layer_h, layer_w] to [batch_size, num_anchors, layer_h, layer_w, 5+num_classes]
        prediction = input.view(batch_size, len(self.scaled_anchors), -1, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()
        
        pxy = torch.sigmoid(prediction[..., 0:2])
        pwh = torch.exp(prediction[..., 2:4])
        pbox = torch.cat([pxy, pwh], dim=-1)
        pconf = torch.sigmoid(prediction[..., 4:5])
        pcls = torch.sigmoid(prediction[..., 5:])
        
        mask, noobj_mask, tbox, tconf, tcls = self.encode_target(target, self.num_classes, self.scaled_anchors, layer_w, layer_h, self.ignore_threshold)
        if prediction.is_cuda:
            mask = mask.cuda()
            noobj_mask = noobj_mask.cuda()
            tbox = tbox.cuda()
            tconf = tconf.cuda()
            tcls = tcls.cuda()

        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #
        box_iou = bbox_iou(pbox * mask, tbox * mask, CIoU=True)
        box_iou = (1 - box_iou[box_iou>0]).mean()
        box_loss = self.lambda_coord * box_iou

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        object_loss = self.lambda_obj * self.bce_loss(pconf * mask, tconf)
        # print(f'object_loss: {object_loss}')
        
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.lambda_noobj * self.bce_loss(pconf * noobj_mask, noobj_mask * 0)
        # print(f'no_object_loss: {no_object_loss}')
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.lambda_class * self.bce_loss(pcls[mask.squeeze(dim=-1)==1], tcls[mask.squeeze(dim=-1)==1])
        # class_loss = self.lambda_class * self.bce_loss(pred_cls * mask.unsqueeze(dim=-1), tcls * mask.unsqueeze(dim=-1))
        # print(f'class_loss: {class_loss}\n')

        loss = (box_loss + object_loss + no_object_loss + class_loss) * batch_size

        return loss


    def encode_target(self, target, num_classes, scaled_anchors, layer_w, layer_h, ignore_threshold):
        """YoloV2 Loss Function

        Arguments:
            target (Tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            num_classes (int): Number of classes in the dataset
            scaled_anchors (List): Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
            layer_w (int): size of predictions width
            layer_h (int): size of predictions height
            ignore_threshold (float): float value of ignore iou
        
        Retruns:
            mask (Tensor): Objectness Mask Tensor, [batch_size, num_anchors, layer_h, layer_w, 1]
            noobj_mask (Tensor): No Objectness Mask Tensor, [batch_size, num_anchors, layer_h, layer_w, 1]
            tbox (Tensor): Ground Truth Box, [batch_size, num_anchors, layer_h, layer_w, 4]
            tconf (Tensor): Ground Truth Confidence Score, [batch_size, num_anchors, layer_h, layer_w, 1]
            tcls (Tensor): Ground Truth Class index, [batch_size, num_anchors, layer_h, layer_w, num_classes]
        """
        batch_size = target.size(0)
        num_anchors = len(scaled_anchors)
        
        mask = torch.zeros(batch_size, num_anchors, layer_h, layer_w, 1)
        noobj_mask = torch.ones(batch_size, num_anchors, layer_h, layer_w, 1)
        tbox = torch.zeros(batch_size, num_anchors, layer_h, layer_w, 4)
        tconf = torch.zeros(batch_size, num_anchors, layer_h, layer_w, 1)
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
                tbox[b, best_n, gj, gi] = torch.tensor([gx - gi, gy - gj, gw/scaled_anchors[best_n][0], gh/scaled_anchors[best_n][1]])
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1

        return mask, noobj_mask, tbox, tconf, tcls  