import sys
import os
sys.path.append(os.getcwd())
import time

import torch
from torch import nn

from utils.yolo_utils import bbox_iou, get_target_boxes_for_map, nms_v1, nms_v2, nms_v3, mean_average_precision


def decode_predictions(input, num_classes, scaled_anchors, input_size):
    """decodes predictions of the YOLO v2 model
    
    Convert predictions to boundig boxes info

    Arguments:
        input (Tensor): predictions of the YOLO v2 model with shape  '(batch, num_anchors*(5 + num_classes), 13, 13)'
        num_classes: Number of classes in the dataset
        scaled_anchors: Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
        input_size: input size of Image

    Returns:
        Tensor: boxes after decoding predictions '(batch, num_anchors*13*13, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
                cx, cy, w, h values are input_size scale
    """
    batch_size, _, layer_h, layer_w = input.size()
    num_anchors = len(scaled_anchors)
    stride_h = input_size / layer_h
    stride_w = input_size / layer_w
    
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


class DecodeYoloV2(nn.Module):
    '''Decode Yolo V2 Predictions to bunding boxes
    '''
    
    def __init__(self, num_classes, scaled_anchors, input_size, conf_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.scaled_anchors = scaled_anchors
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
    def forward(self, x):
        assert x.size(0) == 1
        decode_pred = decode_predictions(x, self.num_classes, self.scaled_anchors, self.input_size)
        # boxes = nms_v1(decode_pred[0], conf_threshold=self.conf_threshold)
        # boxes = nms_v2(decode_pred[0], conf_threshold=self.conf_threshold)
        boxes = nms_v3(decode_pred[0], conf_threshold=self.conf_threshold)
        return boxes


class MeanAveragePrecision:
    def __init__(self, num_classes, scaled_anchors, input_size, conf_threshold):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0
        self.num_classes = num_classes
        self.scaled_anchors = scaled_anchors
        self.input_size = input_size
        self.conf_threshold = conf_threshold

    def reset_states(self):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0

    def update_state(self, y_true, y_pred):
        true_boxes = get_target_boxes_for_map(y_true, self.input_size)

        pred_boxes = decode_predictions(y_pred, self.num_classes, self.scaled_anchors, self.input_size)

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
                self.all_true_boxes_variable = torch.cat([self.all_true_boxes_variable, true_concat], axis=0)
                self.all_pred_boxes_variable = torch.cat([self.all_pred_boxes_variable, pred_concat], axis=0)

            self.img_idx += 1

    def result(self):
        return mean_average_precision(self.all_true_boxes_variable, self.all_pred_boxes_variable, self.num_classes)


if __name__ == '__main__':
    num_classes = 20
    scaled_anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    num_anchors = len(scaled_anchors)
    input_size = 416
    
    tbox = torch.zeros((1, 4))
    tbox[0, :] = torch.tensor([0.5, 0.5, 4, 4])
    pbox = torch.zeros((1, 4))
    pbox[0, :] = torch.tensor([0.55, 0.55, 5, 5])
    iou = bbox_iou(tbox, pbox, x1y1x2y2=True, CIoU=True)
    print(iou)
    