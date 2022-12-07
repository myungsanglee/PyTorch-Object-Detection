import sys
import os
sys.path.append(os.getcwd())
import time

import torch
from torch import nn

from utils.yolo_utils import bbox_iou, get_target_boxes_for_map, nms_v1, nms_v2, nms_v3, mean_average_precision


def decode_predictions(input, num_classes, anchors, input_size):
    """decodes predictions of the YOLO v3 model
    
    Convert predictions to boundig boxes info

    Arguments:
        input (Tensor): predictions of the YOLO v3 model with shape  '(batch, num_anchors*(5 + num_classes), 13, 13)'
        num_classes: Number of classes in the dataset
        anchors: Anchors of a specific dataset, [num_anchors, 2(anchor_w, anchor_h)]
        input_size: input size of Image

    Returns:
        Tensor: boxes after decoding predictions '(batch, num_anchors*13*13, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
                cx, cy, w, h values are input_size scale
    """
    batch_size, _, layer_h, layer_w = input.size()
    num_anchors = len(anchors)
    stride_h = input_size / layer_h
    stride_w = input_size / layer_w
    scaled_anchors = [[anchor_w / stride_w, anchor_h / stride_h] for anchor_w, anchor_h in anchors]
    
    # [batch_size, num_anchors, 5+num_classes, layer_h, layer_w] to [batch_size, num_anchors, layer_h, layer_w, 5+num_classes]
    prediction = input.view(batch_size, num_anchors, -1, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()
    
    FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
    
    stride = FloatTensor([stride_w, stride_h] * 2)
    
    scaled_anchors = FloatTensor(scaled_anchors).unsqueeze(dim=0)
    scaled_anchors = scaled_anchors.repeat(batch_size, 1, 1).repeat(1, 1, layer_h*layer_w).view(batch_size, num_anchors, layer_h, layer_w, 2)
    
    grid_x = torch.arange(0, layer_w).repeat(layer_h, 1).repeat(batch_size*num_anchors, 1, 1).view(batch_size, num_anchors, layer_h, layer_w, 1).type(FloatTensor)
    grid_y = torch.arange(0, layer_w).repeat(layer_h, 1).t().repeat(batch_size*num_anchors, 1, 1).view(batch_size, num_anchors, layer_h, layer_w, 1).type(FloatTensor)
    grid_xy = torch.cat([grid_x, grid_y], dim=-1) # [batch_size, num_anchors, layer_h, layer_w, 2]
    
    pxy = torch.sigmoid(prediction[..., 0:2]) + grid_xy
    pwh = torch.exp(prediction[..., 2:4]) * scaled_anchors
    pbox = torch.cat([pxy, pwh], dim=-1)
    pconf = torch.sigmoid(prediction[..., 4:5])
    pcls = torch.sigmoid(prediction[..., 5:])

    pbox = pbox.view(batch_size, -1, 4) # [batch_size, num_anchors*layer_h*layer_w, 4]
    pbox *= stride # convert ouput scale to input scale
    pconf = pconf.view(batch_size, -1, 1) # [batch_size, num_anchors*layer_h*layer_w, 1]
    pcls = pcls.view(batch_size, -1, num_classes) # [batch_size, num_anchors*layer_h*layer_w, num_classes]
    pcls = torch.argmax(pcls, dim=-1, keepdim=True) # [batch_size, num_anchors*layer_h*layer_w, 1]

    return torch.cat((pbox, pconf, pcls), -1)


class DecodeYoloV3(nn.Module):
    '''Decode Yolo V3 Predictions to bunding boxes
    '''
    
    def __init__(self, num_classes, anchors, input_size, conf_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
    def forward(self, x):
        assert x[0].size(0) == 1
        decode_preds = 0
        for idx, pred in enumerate(x):
            tmp_idx = 3 * idx
            anchors = self.anchors[tmp_idx:tmp_idx+3]
            decode_pred = decode_predictions(pred, self.num_classes, anchors, self.input_size)

            if idx == 0:
                decode_preds = decode_pred
            else:
                decode_preds = torch.cat([decode_preds, decode_pred], dim=1)
        
        # boxes = nms_v1(decode_preds[0], conf_threshold=self.conf_threshold)
        # boxes = nms_v2(decode_preds[0], conf_threshold=self.conf_threshold)
        boxes = nms_v3(decode_preds[0], conf_threshold=self.conf_threshold)
        
        return boxes


class MeanAveragePrecision:
    def __init__(self, num_classes, anchors, input_size, conf_threshold):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_size = input_size
        self.conf_threshold = conf_threshold

    def reset_states(self):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0

    def update_state(self, y_true, y_preds):
        true_boxes = get_target_boxes_for_map(y_true, self.input_size)

        pred_boxes = 0
        for idx, y_pred in enumerate(y_preds):
            tmp_idx = 3 * idx
            anchors = self.anchors[tmp_idx:tmp_idx+3]
            tmp_pred_boxes = decode_predictions(y_pred, self.num_classes, anchors, self.input_size)

            if idx == 0:
                pred_boxes = tmp_pred_boxes
            else:
                pred_boxes = torch.cat([pred_boxes, tmp_pred_boxes], dim=1)

        for idx in torch.arange(y_true.size(0)):
            # pred_nms = nms_v1(pred_boxes[idx], conf_threshold=self.conf_threshold)
            # pred_nms = nms_v2(pred_boxes[idx], conf_threshold=self.conf_threshold)
            pred_nms = nms_v3(pred_boxes[idx], conf_threshold=self.conf_threshold)
            pred_img_idx = torch.zeros([pred_nms.size(0), 1], dtype=torch.float32) + self.img_idx
            if pred_nms.is_cuda:
                pred_img_idx = pred_img_idx.cuda()
            pred_concat = torch.cat([pred_img_idx, pred_nms], dim=1)
            
            true_nms = true_boxes[int(idx)]
            if pred_nms.is_cuda:
                true_nms = true_nms.cuda()
            true_img_idx = torch.zeros([true_nms.size(0), 1], dtype=torch.float32) + self.img_idx
            if true_nms.is_cuda:
                true_img_idx = true_img_idx.cuda()
            true_concat = torch.cat([true_img_idx, true_nms], dim=1)
            
            if self.img_idx == 0.:
                self.all_true_boxes_variable = true_concat
                self.all_pred_boxes_variable = pred_concat
            else:
                self.all_true_boxes_variable = torch.cat([self.all_true_boxes_variable, true_concat], dim=0)
                self.all_pred_boxes_variable = torch.cat([self.all_pred_boxes_variable, pred_concat], dim=0)

            self.img_idx += 1

    def result(self):
        return mean_average_precision(self.all_true_boxes_variable, self.all_pred_boxes_variable, self.num_classes)


if __name__ == '__main__':
    tbox = torch.zeros((1, 4))
    tbox[0, :] = torch.tensor([0.5, 0.5, 4, 4])
    pbox = torch.zeros((1, 4))
    pbox[0, :] = torch.tensor([0.55, 0.55, 5, 5])
    iou = bbox_iou(tbox, pbox, x1y1x2y2=True, CIoU=True)
    print(iou)

    # Test nms
    import time
    # tmp_boxes = torch.zeros((5, 6))
    # tmp_boxes[0, :] = torch.tensor([100, 100, 50, 50, 0.8, 1.])
    # tmp_boxes[1, :] = torch.tensor([100, 100, 50, 50, 0.7, 1.])
    # tmp_boxes[2, :] = torch.tensor([60, 60, 50, 50, 0.6, 1.])
    # tmp_boxes[3, :] = torch.tensor([100, 100, 50, 50, 0.8, 2.])
    # tmp_boxes[4, :] = torch.tensor([100, 100, 50, 50, 0.7, 2.])
    
    tmp_boxes = torch.randn((500, 6))
    
    start = time.time()
    box_1 = nms_v1(tmp_boxes)
    print(f'Time: {1000*(time.time() - start)}ms')
    # print(box_1)
    
    start = time.time()
    box_2 = nms_v2(tmp_boxes)
    print(f'Time: {1000*(time.time() - start)}ms')
    # print(box_2)
    # print(box_2.shape)
    
    start = time.time()
    box_3 = nms_v3(tmp_boxes)
    print(f'Time: {1000*(time.time() - start)}ms')
    # print(box_3)
    