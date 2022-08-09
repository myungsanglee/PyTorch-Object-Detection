import sys
import os
sys.path.append(os.getcwd())

from collections import Counter
import math

import torch
from torch import nn
import numpy as np
import cv2


def collater(data):
    """Data Loader에서 생성된 데이터를 동일한 shape으로 정렬해서 Batch로 전달

    Arguments::
        data (Dict): albumentation Transformed 객체
        'image': list of Torch Tensor len == batch_size, item shape = ch, h, w
        'bboxes': list of list([cx, cy, w, h, cid])

    Returns:
        Dict: 정렬된 batch data.
        'img': list of image tensor, [batch_size, channel, height, width] shape
        'annot': 동일 shape으로 정렬된 tensor, [batch_size, max_num_annots, 5(cx, cy, w, h, cid)] shape
    """
    imgs = [s['image'] for s in data]
    bboxes = [torch.tensor(s['bboxes'])for s in data]
    batch_size = len(imgs)

    max_num_annots = max(annots.shape[0] for annots in bboxes)

    if max_num_annots > 0:
        padded_annots = torch.ones((batch_size, max_num_annots, 5)) * -1
        for idx, annot in enumerate(bboxes):
            if annot.shape[0] > 0:
                padded_annots[idx, :annot.shape[0], :] = annot
                
    else:
        padded_annots = torch.ones((batch_size, 1, 5)) * -1

    return {'img': torch.stack(imgs), 'annot': padded_annots}


# def intersection_over_union_numpy(boxes1, boxes2):
#     """Calculation of intersection-over-union

#     Arguments:
#         boxes1 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]
#         boxes2 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]

#     Returns:
#         Numpy Array: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
#     """

#     box1_xmin = (boxes1[..., 0:1] - boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
#     box1_ymin = (boxes1[..., 1:2] - boxes1[..., 3:4]) / 2. # (batch, S, S, 1)
#     box1_xmax = (boxes1[..., 0:1] + boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
#     box1_ymax = (boxes1[..., 1:2] + boxes1[..., 3:4]) / 2. # (batch, S, S, 1)

#     box2_xmin = (boxes2[..., 0:1] - boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
#     box2_ymin = (boxes2[..., 1:2] - boxes2[..., 3:4]) / 2. # (batch, S, S, 1)
#     box2_xmax = (boxes2[..., 0:1] + boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
#     box2_ymax = (boxes2[..., 1:2] + boxes2[..., 3:4]) / 2. # (batch, S, S, 1)

#     inter_xmin = np.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
#     inter_ymin = np.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
#     inter_xmax = np.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
#     inter_ymax = np.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

#     inter_area = np.clip((inter_xmax - inter_xmin), 0) * np.clip((inter_ymax - inter_ymin), 0) # (batch, S, S, 1)
#     box1_area = np.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
#     box2_area = np.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

#     return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


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


def non_max_suppression_numpy(boxes, iou_threshold=0.5, conf_threshold=0.5):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Numpy Array): All boxes with each grid '(None, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Numpy Array: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = np.take(boxes, np.where(boxes[..., 4] > conf_threshold)[0], axis=0)

    # sort descending by confidence score
    boxes = np.take(boxes, np.argsort(-boxes[..., 4]), axis=0)

    # get boxes after nms
    boxes_after_nms = np.empty(shape=(0, 6))

    while not(np.less(boxes.shape[0], 1)):
        chosen_box = np.expand_dims(boxes[0], axis=0)
        tmp_boxes = np.empty(shape=(0, 6))
        for idx in range(1, boxes.shape[0]):
            tmp_box = np.expand_dims(boxes[idx], axis=0)
            if tmp_box[0][-1] != chosen_box[0][-1] or bbox_iou(chosen_box[..., :4], tmp_box[..., :4]) < iou_threshold:
                tmp_boxes = np.append(tmp_boxes, tmp_box, axis=0)
        boxes = tmp_boxes

        boxes_after_nms = np.append(boxes_after_nms, chosen_box, axis=0)

    return boxes_after_nms


def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.25):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(None, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = boxes[torch.where(boxes[..., 4] > conf_threshold)[0]]

    # sort descending by confidence score
    boxes = boxes[torch.argsort(-boxes[..., 4])]
 
    # get boxes after nms
    boxes_after_nms = []

    if boxes.size(0) == 0:
        return boxes

    while True:
        chosen_box = boxes[:1, ...]
        boxes_after_nms.append(chosen_box[0])
        
        tmp_boxes = []
        for idx in range(1, boxes.size(0)):
            tmp_box = boxes[idx:idx+1, ...]
            if tmp_box[0][-1] != chosen_box[0][-1]:
                tmp_boxes.append(tmp_box[0])
            elif torch.lt(bbox_iou(chosen_box[..., :4], tmp_box[..., :4]), iou_threshold):
                tmp_boxes.append(tmp_box[0])
                
        if tmp_boxes:
            boxes = torch.stack(tmp_boxes)
        else:
            break

    return torch.stack(boxes_after_nms)


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
    """
    batch_size, _, layer_h, layer_w = input.size()
    num_anchors = len(scaled_anchors)
    stride_h = input_size / layer_h
    stride_w = input_size / layer_w
    # [batch_size, num_anchors, 5+num_classes, layer_h, layer_w] to [batch_size, num_anchors, layer_h, layer_w, 5+num_classes]
    prediction = input.view(batch_size, num_anchors, -1, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()

    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    conf = torch.sigmoid(prediction[..., 4])
    pred_cls = torch.sigmoid(prediction[..., 5:])

    pred_cls = pred_cls.view(batch_size, -1, num_classes) # [batch_size, num_anchors*layer_h*layer_w, num_classes]
    pred_cls = torch.argmax(pred_cls, dim=-1, keepdim=True) # [batch_size, num_anchors*layer_h*layer_w, 1]

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    # Calculate offsets for each grid
    grid_x = torch.linspace(0, layer_w-1, layer_w).repeat(layer_w, 1).repeat(
        batch_size * num_anchors, 1, 1).view(x.size()).type(FloatTensor)
    grid_y = torch.linspace(0, layer_h-1, layer_h).repeat(layer_h, 1).t().repeat(
        batch_size * num_anchors, 1, 1).view(y.size()).type(FloatTensor)
    # Calculate anchor w, h
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, layer_h * layer_w).view(w.size())
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, layer_h * layer_w).view(h.size())
    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction[..., :4].size())
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h
    # pred_boxes[..., 2] = ((torch.sigmoid(w) * 2) ** 2) * anchor_w
    # pred_boxes[..., 3] = ((torch.sigmoid(h) * 2) ** 2) * anchor_h

    # Results
    _scale = FloatTensor([stride_w, stride_h] * 2)
    output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale, conf.view(batch_size, -1, 1), pred_cls), -1)
    
    return output


def get_target_boxes(target, input_size):
    """Decode YoloV2 Ground Truth to Bounding Boxes

    Arguments:
        target (Tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
        input_size (int): Input Size of Image
    
    Retruns:
        List: encoded target bounding boxes, specified as [None, 6(cx, cy, w, h, confidence, class_idx)]
    """
    dst = []

    for b in range(target.size(0)):
        for t in torch.arange(target.size(1)):
            if target[b, t].sum() <= 0:
                continue
            gx = target[b, t, 0] * input_size
            gy = target[b, t, 1] * input_size
            gw = target[b, t, 2] * input_size
            gh = target[b, t, 3] * input_size

            dst.append([gx, gy, gw, gh, 1., target[b, t, 4]])

    return dst
    

def get_target_boxes_for_map(target, input_size):
    """Decode YoloV2 Ground Truth to Bounding Boxes

    Arguments:
        target (Tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
        input_size (int): Input Size of Image
    
    Retruns:
        Dict: encoded target bounding boxes, specified as [None, 6(cx, cy, w, h, confidence, class_idx)]
    """
    dst = dict()

    for b in range(target.size(0)):
        tmp = []
        for t in torch.arange(target.size(1)):
            if target[b, t].sum() <= 0:
                continue
            gx = target[b, t, 0] * input_size
            gy = target[b, t, 1] * input_size
            gw = target[b, t, 2] * input_size
            gh = target[b, t, 3] * input_size

            tmp.append([gx, gy, gw, gh, 1., target[b, t, 4]])

        dst[b] = torch.FloatTensor(tmp)

    return dst


def mean_average_precision(true_boxes, pred_boxes, num_classes, iou_threshold=0.5):
    """Calculates mean average precision

    Arguments:
        true_boxes (Tensor): Tensor of all boxes with all images (None, 7), specified as [img_idx, cx, cy, w, h, confidence_score, class_idx]
        pred_boxes (Tensor): Similar as true_bboxes
        num_classes (int): number of classes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in torch.arange(num_classes, dtype=torch.float32):
        # print('\nCalculating AP: ', int(c), ' / ', num_classes)

        # detections, ground_truths variables in specific class
        detections = pred_boxes[torch.where(pred_boxes[..., -1] == c)[0]]
        ground_truths = true_boxes[torch.where(true_boxes[..., -1] == c)[0]]

        # If none exists for this class then we can safely skip
        total_true_boxes = len(ground_truths)
        if total_true_boxes == 0:
            average_precisions.append(torch.tensor(0))
            continue

        # print(c, ' class ground truths size: ', ground_truths.size()[0])
        # print(c, ' class detections size: ', detections.size()[0])

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([int(gt[0]) for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # sort by confidence score
        detections = detections[torch.sort(detections[..., -2], descending=True)[1]]
        true_positive = torch.zeros((len(detections)))
        false_positive = torch.zeros((len(detections)))

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = ground_truths[torch.where(ground_truths[..., 0] == detection[0])[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = bbox_iou(detection[1:5], gt[1:5])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[int(detection[0])][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    true_positive[detection_idx] = 1
                    amount_bboxes[int(detection[0])][best_gt_idx] = 1
                else:
                    false_positive[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                false_positive[detection_idx] = 1

        tf_cumsum = torch.cumsum(true_positive, dim=0)
        fp_cumsum = torch.cumsum(false_positive, dim=0)
        recalls = tf_cumsum / (total_true_boxes + epsilon)
        precisions = torch.divide(tf_cumsum, (tf_cumsum + fp_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return torch.mean(torch.stack(average_precisions))


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
        class_name = class_name_list[int(bbox[-1])]
        confidence_score = bbox[4]
        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]
        xmin = int((cx - (w / 2)))
        ymin = int((cy - (h / 2)))
        xmax = int((cx + (w / 2)))
        ymax = int((cy + (h / 2)))

        print(f'{class_name}: {xmin}, {ymin}, {xmax}, {ymax}')

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color)
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=color)
    print(f'')
    return img

def get_tagged_img_2(img, boxes, names_path, color):
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
        class_name = class_name_list[int(bbox[-1])]
        confidence_score = bbox[4]
        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]
        xmin = int((cx - (w / 2)))
        ymin = int((cy - (h / 2)))
        xmax = int((cx + (w / 2)))
        ymax = int((cy + (h / 2)))

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=color)

    return img


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
        boxes = non_max_suppression(decode_pred[0], conf_threshold=self.conf_threshold)
        return boxes


class MeanAveragePrecision:
    def __init__(self, num_classes, scaled_anchors, input_size):
        self._all_true_boxes_variable = 0
        self._all_pred_boxes_variable = 0
        self._img_idx = 0
        self._num_classes = num_classes
        self._scaled_anchors = scaled_anchors
        self._input_size = input_size

    def reset_states(self):
        self._all_true_boxes_variable = 0
        self._all_pred_boxes_variable = 0
        self._img_idx = 0

    def update_state(self, y_true, y_pred):
        true_boxes = get_target_boxes_for_map(y_true, self._input_size)

        pred_boxes = decode_predictions(y_pred, self._num_classes, self._scaled_anchors, self._input_size)

        for idx in torch.arange(y_true.size(0)):
            pred_nms = non_max_suppression(pred_boxes[idx], conf_threshold=0.25)
            pred_img_idx = torch.zeros([pred_nms.size(0), 1], dtype=torch.float32) + self._img_idx
            if pred_nms.is_cuda:
                pred_img_idx = pred_img_idx.cuda()
            pred_concat = torch.concat([pred_img_idx, pred_nms], dim=1)

            true_nms = true_boxes[int(idx)]
            true_nms = true_nms.cuda()

            true_img_idx = torch.zeros([true_nms.size(0), 1], dtype=torch.float32) + self._img_idx
            if true_nms.is_cuda:
                true_img_idx = true_img_idx.cuda()
            true_concat = torch.concat([true_img_idx, true_nms], dim=1)
            
            if self._img_idx == 0.:
                self._all_true_boxes_variable = true_concat
                self._all_pred_boxes_variable = pred_concat
            else:
                self._all_true_boxes_variable = torch.concat([self._all_true_boxes_variable, true_concat], axis=0)
                self._all_pred_boxes_variable = torch.concat([self._all_pred_boxes_variable, pred_concat], axis=0)

            self._img_idx += 1

    def result(self):
        return mean_average_precision(self._all_true_boxes_variable, self._all_pred_boxes_variable, self._num_classes)


if __name__ == '__main__':
    num_classes = 20
    scaled_anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    num_anchors = len(scaled_anchors)
    input_size = 416
    
    y_true = np.zeros(shape=(1, 3, 5), dtype=np.float32)
    y_true[:, 0, :] = [0.047, 0.047, 0.112, 0.147, 0.] # [cx, cy, w, h, class_idx]
    y_true[:, 1, :] = [0.277, 0.277, 0.112, 0.147, 1.] # [cx, cy, w, h, class_idx]
    y_true[:, 2, :] = [0.51, 0.51, 0.112, 0.147, 2.] # [cx, cy, w, h, class_idx]
    
    y_true_tensor = torch.as_tensor(y_true)
    print(f'{y_true_tensor.size()}, {y_true_tensor.dtype}')
    
    y_pred = np.zeros(shape=(1, num_anchors*(5 + num_classes), 13, 13), dtype=np.float32)
    y_pred[:, 5:8, 0, 0] = [0.8, 0.5, 0.1] # class
    y_pred[:, 4, 0, 0] = 0.6 # confidence1
    y_pred[:, :4, 0, 0] = [0.49, 0.49, 0.1, 0.1] # box1
    
    y_pred[:, 5:8, 3, 3] = [0.2, 0.8, 0.1] # class
    y_pred[:, 4, 3, 3] = 0.1 # confidence1
    y_pred[:, :4, 3, 3] = [0.45, 0.45, 0.1, 0.1] # box1
    
    y_pred[:, 5:8, 6, 6] = [0.1, 0.5, 0.8] # class
    y_pred[:, 4, 6, 6] = 0.6 # confidence1
    y_pred[:, :4, 6, 6] = [0.49, 0.49, 0.1, 0.1] # box1
    
    y_pred_tensor = torch.as_tensor(y_pred)
    print(f'{y_pred_tensor.size()}, {y_pred_tensor.dtype}')
    
    # encode = encode_target(y_true_tensor, num_classes, scaled_anchors, input_size)
    # decode_true = decode_predictions(encode, num_classes, scaled_anchors, input_size)
    # print(f'Decode Truth\n{decode_true}\n{decode_true.size()}\n')

    # Decode Prediction Test
    decode_pred = decode_predictions(y_pred_tensor, num_classes, scaled_anchors, input_size)
    print(f'Decode Pred\n{decode_pred}\n{decode_pred.size()}\n')

    # NMS Test
    boxes_pred = non_max_suppression(decode_pred[0], conf_threshold=0.5)
    print(f'NMS Pred\n{boxes_pred}\n{boxes_pred.size()}\n')

    # boxes_true = non_max_suppression(decode_true[0], conf_threshold=0.5)
    # print(f'NMS Truth\n{boxes_true}\n{boxes_true.size()}\n')

    # iou = bbox_iou(boxes_pred[0, :4], boxes_true[0, :4])
    # print(iou)

    # mAP Test
    map_metric = MeanAveragePrecision(num_classes, scaled_anchors, input_size)
    map_metric.update_state(y_true_tensor, y_pred_tensor)
    map = map_metric.result()    
    print(map)
    map_metric.reset_states()
    
    map_metric.update_state(y_true_tensor, y_pred_tensor)
    map = map_metric.result()    
    print(map)
    