from collections import Counter
import math

import torch
from torch import nn
from torchvision.ops import batched_nms
import numpy as np
import cv2


def intersection_over_union_numpy(boxes1, boxes2):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]
        boxes2 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]

    Returns:
        Numpy Array: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
    """

    box1_xmin = boxes1[..., 0:1] - (boxes1[..., 2:3] / 2.) # (batch, S, S, 1)
    box1_ymin = boxes1[..., 1:2] - (boxes1[..., 3:4] / 2.) # (batch, S, S, 1)
    box1_xmax = boxes1[..., 0:1] + (boxes1[..., 2:3] / 2.)# (batch, S, S, 1)
    box1_ymax = boxes1[..., 1:2] + (boxes1[..., 3:4] / 2.) # (batch, S, S, 1)

    box2_xmin = boxes2[..., 0:1] - (boxes2[..., 2:3] / 2.) # (batch, S, S, 1)
    box2_ymin = boxes2[..., 1:2] - (boxes2[..., 3:4] / 2.) # (batch, S, S, 1)
    box2_xmax = boxes2[..., 0:1] + (boxes2[..., 2:3] / 2.) # (batch, S, S, 1)
    box2_ymax = boxes2[..., 1:2] + (boxes2[..., 3:4] / 2.) # (batch, S, S, 1)

    inter_xmin = np.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = np.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = np.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = np.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = np.clip((inter_xmax - inter_xmin), 0, (inter_xmax - inter_xmin)) * np.clip((inter_ymax - inter_ymin), 0, (inter_ymax - inter_ymin)) # (batch, S, S, 1)
    box1_area = np.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = np.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


def bbox_iou(boxes1, boxes2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-6):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h] or [x1, y1, x2, y2]
        boxes2 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h] or [x1, y1, x2, y2]
        x1y1x2y2 (Bool): boxes coordinate type if 'True' that means [xmin, ymin, xmax, ymax]
        GIoU (Bool): Calculating Generalized-IoU
        DIoU (Bool): Calculating Distance-IoU
        CIoU (Bool): Calculating Complete-IoU
        eps (Float): epsilon

    Returns:
        Tensor: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
    """

    if x1y1x2y2:
        box1_xmin = boxes1[..., 0:1] # (batch, S, S, 1)
        box1_ymin = boxes1[..., 1:2] # (batch, S, S, 1)
        box1_xmax = boxes1[..., 2:3] # (batch, S, S, 1)
        box1_ymax = boxes1[..., 3:4] # (batch, S, S, 1)

        box2_xmin = boxes2[..., 0:1] # (batch, S, S, 1)
        box2_ymin = boxes2[..., 1:2] # (batch, S, S, 1)
        box2_xmax = boxes2[..., 2:3] # (batch, S, S, 1)
        box2_ymax = boxes2[..., 3:4] # (batch, S, S, 1)

    else:
        box1_xmin = boxes1[..., 0:1] - (boxes1[..., 2:3] / 2.) # (batch, S, S, 1)
        box1_ymin = boxes1[..., 1:2] - (boxes1[..., 3:4] / 2.) # (batch, S, S, 1)
        box1_xmax = boxes1[..., 0:1] + (boxes1[..., 2:3] / 2.) # (batch, S, S, 1)
        box1_ymax = boxes1[..., 1:2] + (boxes1[..., 3:4] / 2.) # (batch, S, S, 1)

        box2_xmin = boxes2[..., 0:1] - (boxes2[..., 2:3] / 2.) # (batch, S, S, 1)
        box2_ymin = boxes2[..., 1:2] - (boxes2[..., 3:4] / 2.) # (batch, S, S, 1)
        box2_xmax = boxes2[..., 0:1] + (boxes2[..., 2:3] / 2.) # (batch, S, S, 1)
        box2_ymax = boxes2[..., 1:2] + (boxes2[..., 3:4] / 2.) # (batch, S, S, 1)

    inter_xmin = torch.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = torch.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = torch.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = torch.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)
    inter_area = torch.clamp((inter_xmax - inter_xmin), 0) * torch.clamp((inter_ymax - inter_ymin), 0) # (batch, S, S, 1)
    
    box1_area = torch.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = torch.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)
    union_area = box1_area + box2_area - inter_area + eps

    iou = inter_area / union_area # (batch, S, S, 1)

    if GIoU or DIoU or CIoU:
        cw = torch.maximum(box1_xmax, box2_xmax) - torch.minimum(box1_xmin, box2_xmin)  # convex (smallest enclosing box) width
        ch = torch.maximum(box1_ymax, box2_ymax) - torch.minimum(box1_ymin, box2_ymin)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((box2_xmin + box2_xmax - box1_xmin - box1_xmax) ** 2 +
                    (box2_ymin + box2_ymax - box1_ymin - box1_ymax) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan((box2_xmax - box2_xmin) / (box2_ymax - box2_ymin)) - \
                    torch.atan((box1_xmax - box1_xmin) / (box1_ymax - box1_ymin)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
    else:
        return iou


def non_max_suppression_numpy(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Numpy Array): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Numpy Array: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = np.take(boxes, np.where(boxes[..., 1] > conf_threshold)[0], axis=0)

    # sort descending by confidence score
    boxes = np.take(boxes, np.argsort(-boxes[..., 1]), axis=0)

    # get boxes after nms
    boxes_after_nms = np.empty(shape=(0, 6))

    while not(np.less(boxes.shape[0], 1)):
        chosen_box = np.expand_dims(boxes[0], axis=0)
        tmp_boxes = np.empty(shape=(0, 6))
        for idx in range(1, boxes.shape[0]):
            tmp_box = np.expand_dims(boxes[idx], axis=0)
            if tmp_box[0][0] != chosen_box[0][0] or bbox_iou(chosen_box[..., 2:], tmp_box[..., 2:]) < iou_threshold:
                tmp_boxes = np.append(tmp_boxes, tmp_box, axis=0)
        boxes = tmp_boxes

        boxes_after_nms = np.append(boxes_after_nms, chosen_box, axis=0)

    return boxes_after_nms


def nms_v1(boxes, conf_threshold=0.25, iou_threshold=0.45):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        conf_threshold (float): threshold to remove predicted boxes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = boxes[torch.where(boxes[..., 1] > conf_threshold)[0]]

    # sort descending by confidence score
    boxes = boxes[torch.argsort(-boxes[..., 1])]
 
    # get boxes after nms
    boxes_after_nms = []

    if boxes.size()[0] == 0:
        return boxes

    while True:
        chosen_box = boxes[:1, ...]
        boxes_after_nms.append(chosen_box[0])
        
        tmp_boxes = []
        for idx in range(1, boxes.shape[0]):
            tmp_box = boxes[idx:idx+1, ...]
            if tmp_box[0][0] != chosen_box[0][0]:
                tmp_boxes.append(tmp_box[0])
            elif torch.lt(bbox_iou(chosen_box[..., 2:], tmp_box[..., 2:]), iou_threshold):
                tmp_boxes.append(tmp_box[0])
                
        if tmp_boxes:
            boxes = torch.stack(tmp_boxes)
        else:
            break

    return torch.stack(boxes_after_nms)


def nms_v2(boxes, conf_threshold=0.25, iou_threshold=0.45):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        conf_threshold (float): threshold to remove predicted boxes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """
    if boxes.is_cuda:
        device = boxes.device
        boxes = boxes.cpu().numpy()
    else:
        device = None
        boxes = boxes.numpy()
    
    # boxes smaller than the conf_threshold are removed
    boxes = boxes[np.where(boxes[..., 1] > conf_threshold)[0]]

    if boxes.shape[0] == 0:
        if device is not None:
            boxes = torch.cuda.FloatTensor(boxes, device=device)
        else:
            boxes = torch.FloatTensor(boxes, device='cpu')
        return boxes

    # get unique class idx
    classes_idx = np.unique(boxes[..., 0])
    
    # get boxes after nms
    boxes_after_nms = np.zeros((0, 6))

    for class_idx in classes_idx:
        # get specific class boxes
        tmp_boxes = boxes[np.where(boxes[..., 0] == class_idx)[0]]
        
        # initialize the list of picked indexes	
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = tmp_boxes[..., 2] - (tmp_boxes[..., 4] / 2)
        y1 = tmp_boxes[..., 3] - (tmp_boxes[..., 5] / 2)
        x2 = tmp_boxes[..., 2] + (tmp_boxes[..., 4] / 2)
        y2 = tmp_boxes[..., 3] + (tmp_boxes[..., 5] / 2)
    
        # compute the area of the bounding boxes and sort the bounding
        # boxes by confidence score
        # area = (x2 - x1 + 1) * (y2 - y1 + 1)
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(tmp_boxes[..., 1])
        
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # compute the width and height of the bounding box
            # w = np.maximum(0, xx2 - xx1 + 1)
            # h = np.maximum(0, yy2 - yy1 + 1)
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # compute the ratio of overlap
            # overlap = (w * h) / area[idxs[:last]]
            overlap = (w * h) / (area[idxs[:last]] + area[i] - (w*h))
            
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > iou_threshold)[0])))

        boxes_after_nms = np.concatenate([boxes_after_nms, tmp_boxes[pick]], axis=0)
    
    if device is not None:
        boxes_after_nms = torch.cuda.FloatTensor(boxes_after_nms, device=device)
    else:
        boxes_after_nms = torch.FloatTensor(boxes_after_nms, device='cpu')
    
    return boxes_after_nms


def nms_v3(boxes, conf_threshold=0.25, iou_threshold=0.45):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        conf_threshold (float): threshold to remove predicted boxes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = boxes[torch.where(boxes[..., 1] > conf_threshold)[0]]

    if boxes.size()[0] == 0:
        return boxes

    x1 = (boxes[..., 2] - (boxes[..., 4] / 2)).unsqueeze(dim=-1)
    y1 = (boxes[..., 3] - (boxes[..., 5] / 2)).unsqueeze(dim=-1)
    x2 = (boxes[..., 2] + (boxes[..., 4] / 2)).unsqueeze(dim=-1)
    y2 = (boxes[..., 3] + (boxes[..., 5] / 2)).unsqueeze(dim=-1)
    x1y1x2y2 = torch.cat([x1, y1, x2, y2], dim=-1)
    
    # get boxes after nms
    boxes_after_nms = batched_nms(x1y1x2y2, boxes[..., 1], boxes[..., 0], iou_threshold)
    
    return boxes[boxes_after_nms]


def decode_predictions_numpy(predictions, num_classes, num_boxes=2):
    """decodes predictions of the YOLO v1 model
    
    Converts bounding boxes output from Yolo with
    an image split size of GRID into entire image ratios
    rather than relative to cell ratios.

    Arguments:
        predictions (Numpy): predictions of the YOLO v1 model with shape  '(1, 7, 7, (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict

    Returns:
        Numpy: boxes after decoding predictions each grid cell with shape '(batch, S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
    """

    # Get class indexes
    class_indexes = np.argmax(predictions[..., :num_classes], axis=-1) # (batch, S, S)
    class_indexes = np.expand_dims(class_indexes, axis=-1) # (batch, S, S, 1)
    class_indexes = class_indexes.astype(np.float32)

    # Get best confidence one-hot
    confidences = []
    for idx in np.arange(num_boxes):
        confidence = predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]
        confidences.append(confidence)
    confidences = np.array(confidences, np.float32) # (num_boxes, batch, S, S, 1)
    best_conf_idx = np.argmax(confidences, axis=0) # (batch, S, S, 1)
    best_conf_one_hot = np.reshape(np.eye(num_boxes)[best_conf_idx.reshape(-1).astype(np.int)], (best_conf_idx.shape[0], best_conf_idx.shape[1], best_conf_idx.shape[2], num_boxes)) # (batch, S, S, num_boxes)

    # Get prediction box & confidence
    pred_box = np.zeros(shape=[1, 7, 7, 4])
    pred_conf = np.zeros(shape=[1, 7, 7, 1])
    for idx in np.arange(num_boxes):
        pred_box += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(1+(5*idx)):num_classes+(1+(5*idx))+4]
        pred_conf += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]

    # Get cell indexes array
    base_arr = np.arange(7).reshape((1, -1)).repeat(7, axis=0)
    x_cell_indexes = np.expand_dims(base_arr, axis=-1)  # (S, S, 1)

    y_cell_indexes = np.transpose(base_arr)
    y_cell_indexes = np.expand_dims(y_cell_indexes, axis=-1)  # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / 7 * (pred_box[..., :1] + x_cell_indexes) # (batch, S, S, 1)
    y = 1 / 7 * (pred_box[..., 1:2] + y_cell_indexes) # (batch, S, S, 1)

    pred_box = np.concatenate([x, y, pred_box[..., 2:4]], axis=-1) # (batch, S, S, 4)
    
    # Concatenate result
    pred_result = np.concatenate([class_indexes, pred_conf, pred_box], axis=-1) # (batch, S, S, 6)

    # Get all bboxes
    pred_result = np.reshape(pred_result, (-1, 7*7, 6)) # (batch, S*S, 6)
    
    return pred_result


def decode_predictions(input, num_classes, num_boxes, input_size):
    """decodes predictions of the YOLO v1 model
    
    Convert predictions to boundig boxes info

    Arguments:
        input (Tensor): predictions of the YOLO v1 model with shape  '(batch, 7, 7, (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict
        input_size: input size of Image

    Returns:
        Tensor: boxes after decoding predictions '(batch, S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
                cx, cy, w, h values are input_size scale
    """
    batch_size, layer_h, layer_w, _ = input.size()
    stride_h = input_size / layer_h
    stride_w = input_size / layer_w
    
    prediction = torch.sigmoid(input)
    
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
    
    return torch.cat([pcls, pconf, pbox], dim=-1).view(-1, layer_h*layer_w, 6)


def get_target_boxes(target, num_classes, num_boxes, input_size):
    """Decode YoloV1 Ground Truth to Bounding Boxes

    Arguments:
        target (Tensor): [batch, 7, 7, (num_boxes*5 + num_classes)]
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict
        input_size: input size of Image
    
    Retruns:
        List: encoded target bounding boxes, specified as [None, 6(class_idx, confidence_score, cx, cy, w, h)]
    """
    batch_size, layer_h, layer_w, _ = target.size()
    target = target.view(batch_size, layer_h*layer_w, -1)
    dst = []

    for b in range(target.size(0)):
        for t in torch.arange(target.size(1)):
            if target[b, t, num_classes] == 0:
                continue
            gx = target[b, t, num_classes+1] * input_size
            gy = target[b, t, num_classes+2] * input_size
            gw = target[b, t, num_classes+3] * input_size
            gh = target[b, t, num_classes+4] * input_size

            dst.append([torch.argmax(target[b, t, :num_classes]), 1., gx, gy, gw, gh])

    return dst


def get_tagged_img(img, boxes, names_path, color):
    """tagging result on img

    Arguments:
        img (Numpy Array): Image array
        boxes (Tensor): boxes after performing NMS (None, 6)
        names_path (String): path of label names file
        color (tuple): boxes color
        
    Returns:
        Numpy Array: tagged image array
    """
    with open(names_path, 'r') as f:
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]
    for bbox in boxes:
        class_name = class_name_list[int(bbox[0])]
        confidence_score = bbox[1]
        x = bbox[2]
        y = bbox[3]
        w = bbox[4]
        h = bbox[5]
        xmin = int((x - (w / 2)))
        ymin = int((y - (h / 2)))
        xmax = int((x + (w / 2)))
        ymax = int((y + (h / 2)))

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=color)

    return img


class DecodeYoloV1(nn.Module):
    '''Decode Yolo V1 Predictions to bunding boxes
    '''
    
    def __init__(self, num_classes, num_boxes, conf_threshold=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.conf_threshold = conf_threshold
        
    def forward(self, x):
        assert x.size(0) == 1
        decode_pred = decode_predictions(x, self.num_classes, self.num_boxes)
        boxes = nms_v1(decode_pred[0], conf_threshold=self.conf_threshold)
        return boxes


if __name__ == '__main__':
    input_shape = 448
    num_classes = 20
    num_boxes = 2
    
    y_true = np.zeros(shape=(1, 7, 7, (num_classes + (5*num_boxes))), dtype=np.float32)
    y_true[:, 0, 0, 0] = 1 # class
    y_true[:, 0, 0, num_classes] = 1 # confidence1
    y_true[:, 0, 0, num_classes+1:num_classes+5] = [0.5, 0.5, 0.1, 0.1] # box1
    
    y_true[:, 3, 3, 1] = 1 # class
    y_true[:, 3, 3, num_classes] = 1 # confidence1
    y_true[:, 3, 3, num_classes+1:num_classes+5] = [0.5, 0.5, 0.1, 0.1] # box1
    
    y_true[:, 6, 6, 2] = 1 # class
    y_true[:, 6, 6, num_classes] = 1 # confidence1
    y_true[:, 6, 6, num_classes+1:num_classes+5] = [0.5, 0.5, 0.1, 0.1] # box1
    
    y_true_tensor = torch.as_tensor(y_true)
    print(f'{y_true_tensor.shape}, {y_true_tensor.dtype}')
    
    y_pred = np.zeros(shape=(1, 7, 7, (num_classes + (5*num_boxes))), dtype=np.float32)
    y_pred[:, 0, 0, :3] = [0.8, 0.5, 0.1] # class
    y_pred[:, 0, 0, num_classes] = 0.6 # confidence1
    y_pred[:, 0, 0, num_classes+1:num_classes+5] = [0.49, 0.49, 0.1, 0.1] # box1
    y_pred[:, 0, 0, num_classes+5] = 0.2 # confidence2
    y_pred[:, 0, 0, num_classes+6:num_classes+10] = [0.45, 0.45, 0.1, 0.1] # box2
    
    y_pred[:, 3, 3, :3] = [0.2, 0.8, 0.1] # class
    y_pred[:, 3, 3, num_classes] = 0.1 # confidence1
    y_pred[:, 3, 3, num_classes+1:num_classes+5] = [0.45, 0.45, 0.1, 0.1] # box1
    y_pred[:, 3, 3, num_classes+5] = 0.9 # confidence2
    y_pred[:, 3, 3, num_classes+6:num_classes+10] = [0.49, 0.49, 0.1, 0.1] # box2
    
    y_pred[:, 6, 6, :3] = [0.1, 0.5, 0.8] # class
    y_pred[:, 6, 6, num_classes] = 0.6 # confidence1
    y_pred[:, 6, 6, num_classes+1:num_classes+5] = [0.49, 0.49, 0.1, 0.1] # box1
    y_pred[:, 6, 6, num_classes+5] = 0.2 # confidence2
    y_pred[:, 6, 6, num_classes+6:num_classes+10] = [0.45, 0.45, 0.1, 0.1] # box2
    
    y_pred_tensor = torch.as_tensor(y_pred)
    print(f'{y_pred_tensor.shape}, {y_pred_tensor.dtype}')


    # IoU Test
    # numpy array
    y_true_bbox = y_true[:, 0, 0, num_classes+1:num_classes+5]
    y_pred_bbox = y_pred[:, 0, 0, num_classes+1:num_classes+5]
    # print(y_true_bbox)
    # print(y_pred_bbox)
    # print(intersection_over_union_numpy(y_true_bbox, y_pred_bbox))
    # torch tensor
    y_true_bbox = y_true_tensor[:, 0, 0, num_classes+1:num_classes+5]
    y_pred_bbox = y_pred_tensor[:, 0, 0, num_classes+1:num_classes+5]
    # print(y_true_bbox)
    # print(y_pred_bbox)
    # print(bbox_iou(y_true_bbox, y_pred_bbox))
    
    
    # Decode Prediction Test
    # numpy array
    decode_pred = decode_predictions_numpy(y_pred, num_classes, num_boxes)
    print(decode_pred)
    # torch tensor
    decode_pred_tensor = decode_predictions(y_pred_tensor, num_classes, num_boxes, input_shape)
    print(decode_pred_tensor)
    
    
    # Non Max Suppression Test
    # numpy array
    bboxes = non_max_suppression_numpy(decode_pred[0])
    print(bboxes)
    # torch tensor
    bboxes_tensor = nms_v3(decode_pred_tensor[0])
    print(bboxes_tensor) 
        
    # print(f'-'*100)
    
    a = torch.FloatTensor([[0, 0, 0.5, 0.5]])
    b = torch.FloatTensor([[0, 0, 0.4, 0.4], [0, 0, 0.5, 0.5]])
    # print(bbox_iou(a, b))