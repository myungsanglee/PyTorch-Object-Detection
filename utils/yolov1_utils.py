import sys
import os
sys.path.append(os.getcwd())
import time

import torch
from torch import nn

from utils.yolo_utils import bbox_iou, get_target_boxes_for_map, nms_v1, nms_v2, nms_v3, mean_average_precision


def decode_predictions(input, num_classes, num_boxes, input_size):
    """decodes predictions of the YOLO v1 model
    
    Convert predictions to boundig boxes info

    Arguments:
        input (Tensor): predictions of the YOLO v1 model with shape  '(batch, 7 * 7 * (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict
        input_size: input size of Image

    Returns:
        Tensor: boxes after decoding predictions '(batch, 7*7, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
                cx, cy, w, h values are input_size scale
    """
    prediction = torch.sigmoid(input.view(-1, 7, 7, num_boxes*5 + num_classes))
    
    batch_size, layer_h, layer_w, _ = prediction.size()
    stride_h = input_size / layer_h
    stride_w = input_size / layer_w
    
    FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor
    
    stride = FloatTensor([stride_w, stride_h] * 2)
    
    # Get best confidence one-hot
    confidences = []
    for idx in torch.arange(num_boxes):
        confidence = prediction[..., num_classes+(5*idx):num_classes+(5*idx)+1]
        confidences.append(confidence)
    confidences = torch.stack(confidences) # (num_boxes, batch, S, S, 1)
    best_conf_idx = torch.argmax(confidences, dim=0) # (batch, S, S, 1)
    best_conf_one_hot = torch.nn.functional.one_hot(best_conf_idx, num_boxes).view(-1, 7, 7, num_boxes) # (batch, S, S, num_boxes)

    # Get prediction box & confidence
    pbox = []
    pconf = []
    for idx in torch.arange(num_boxes):
        pbox.append(best_conf_one_hot[..., idx:idx+1] * prediction[..., num_classes+(1+(5*idx)):num_classes+(1+(5*idx))+4])
        pconf.append(best_conf_one_hot[..., idx:idx+1] * prediction[..., num_classes+(5*idx):num_classes+(5*idx)+1])
    pbox = torch.sum(torch.stack(pbox), dim=0) # (batch, S, S, 4)
    pconf = torch.sum(torch.stack(pconf), dim=0) # (batch, S, S, 1)
    
    grid_x = torch.arange(0, layer_w).repeat(layer_h, 1).repeat(batch_size, 1, 1).view(batch_size, layer_h, layer_w, 1).type(FloatTensor)
    grid_y = torch.arange(0, layer_w).repeat(layer_h, 1).t().repeat(batch_size, 1, 1).view(batch_size, layer_h, layer_w, 1).type(FloatTensor)
    grid_xy = torch.cat([grid_x, grid_y], dim=-1) # [batch_size, layer_h, layer_w, 2]
    
    pbox[..., 0:2] += grid_xy
    pbox[..., 2:3] *= layer_w
    pbox[..., 3:4] *= layer_h
    pbox *= stride
    
    pcls = torch.argmax(prediction[..., :num_classes], dim=-1, keepdim=True) # (batch, S, S, 1)
    
    return torch.cat([pbox, pconf, pcls], dim=-1).view(-1, layer_h*layer_w, 6)


class DecodeYoloV1(nn.Module):
    '''Decode Yolo V1 Predictions to bunding boxes
    '''
    
    def __init__(self, num_classes, num_boxes, input_size, conf_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
    def forward(self, x):
        assert x.size(0) == 1
        decode_pred = decode_predictions(x, self.num_classes, self.num_boxes, self.input_size)
        # boxes = nms_v1(decode_pred[0], conf_threshold=self.conf_threshold)
        # boxes = nms_v2(decode_pred[0], conf_threshold=self.conf_threshold)
        boxes = nms_v3(decode_pred[0], conf_threshold=self.conf_threshold)
        return boxes


class MeanAveragePrecision:
    def __init__(self, num_classes, num_boxes, input_size, conf_threshold):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.input_size = input_size
        self.conf_threshold = conf_threshold

    def reset_states(self):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0

    def update_state(self, y_true, y_pred):
        true_boxes = get_target_boxes_for_map(y_true, self.input_size)

        pred_boxes = decode_predictions(y_pred, self.num_classes, self.num_boxes, self.input_size)

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
    