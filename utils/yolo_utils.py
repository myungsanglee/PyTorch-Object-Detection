from collections import Counter
import math

import torch
from torchvision.ops import batched_nms
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image


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
    bboxes = [torch.tensor(s['bboxes']) for s in data]
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

#     inter_area = np.clip((inter_xmax - inter_xmin), 0, (inter_xmax - inter_xmin)) * np.clip((inter_ymax - inter_ymin), 0, (inter_ymax - inter_ymin)) # (batch, S, S, 1)
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


def nms_v1(boxes, conf_threshold=0.25, iou_threshold=0.45):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(None, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
        conf_threshold (float): threshold to remove predicted boxes
        iou_threshold (float): threshold where predicted boxes is correct
        
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
                # if torch.lt(bbox_iou(chosen_box[..., :4], tmp_box[..., :4]), iou_threshold):
                tmp_boxes.append(tmp_box[0])
            elif torch.lt(bbox_iou(chosen_box[..., :4], tmp_box[..., :4]), iou_threshold):
                tmp_boxes.append(tmp_box[0])
                
        if tmp_boxes:
            boxes = torch.stack(tmp_boxes)
        else:
            break

    return torch.stack(boxes_after_nms)


def nms_v2(boxes, conf_threshold=0.25, iou_threshold=0.45):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(None, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]\
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
    boxes = boxes[np.where(boxes[..., 4] > conf_threshold)[0]]

    if boxes.shape[0] == 0:
        if device is not None:
            boxes = torch.cuda.FloatTensor(boxes, device=device)
        else:
            boxes = torch.FloatTensor(boxes, device='cpu')
        return boxes
    
    # get unique class idx
    classes_idx = np.unique(boxes[..., 5])
    
    # get boxes after nms
    boxes_after_nms = np.zeros((0, 6))
    
    for class_idx in classes_idx:
        # get specific class boxes
        tmp_boxes = boxes[np.where(boxes[..., 5] == class_idx)[0]]
        
        # initialize the list of picked indexes	
        pick = []
        
        # grab the coordinates of the bounding boxes
        x1 = tmp_boxes[..., 0] - (tmp_boxes[..., 2] / 2)
        y1 = tmp_boxes[..., 1] - (tmp_boxes[..., 3] / 2)
        x2 = tmp_boxes[..., 0] + (tmp_boxes[..., 2] / 2)
        y2 = tmp_boxes[..., 1] + (tmp_boxes[..., 3] / 2)
                        
        # compute the area of the bounding boxes and sort the bounding
        # boxes by confidence score
        # area = (x2 - x1 + 1) * (y2 - y1 + 1)
        area = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(tmp_boxes[..., 4])
        
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
            overlap = (w * h) / area[idxs[:last]]
            # overlap = (w * h) / (area[idxs[:last]] + area[i] - (w*h))
            
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
        boxes (Tensor): All boxes with each grid '(None, 6)', specified as [cx, cy, w, h, confidence_score, class_idx]
        conf_threshold (float): threshold to remove predicted boxes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = boxes[torch.where(boxes[..., 4] > conf_threshold)[0]]

    if boxes.shape[0] == 0:
        return boxes
    
    x1 = (boxes[..., 0] - (boxes[..., 2] / 2)).unsqueeze(dim=-1)
    y1 = (boxes[..., 1] - (boxes[..., 3] / 2)).unsqueeze(dim=-1)
    x2 = (boxes[..., 0] + (boxes[..., 2] / 2)).unsqueeze(dim=-1)
    y2 = (boxes[..., 1] + (boxes[..., 3] / 2)).unsqueeze(dim=-1)
    x1y1x2y2 = torch.cat([x1, y1, x2, y2], dim=-1)
    
    # get boxes after nms
    boxes_after_nms = batched_nms(x1y1x2y2, boxes[..., 4], boxes[..., 5], iou_threshold)
    
    return boxes[boxes_after_nms]


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
            average_precisions.append(torch.zeros(1))
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

        tp_cumsum = torch.cumsum(true_positive, dim=0)
        fp_cumsum = torch.cumsum(false_positive, dim=0)
        recalls_cumsum = tp_cumsum / (total_true_boxes + epsilon)
        precisions_cumsum = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))
        precisions_cumsum = torch.cat((torch.tensor([1]), precisions_cumsum))
        recalls_cumsum = torch.cat((torch.tensor([0]), recalls_cumsum))
        
        # torch.trapz for numerical integration
        # average_precisions.append(torch.trapz(precisions, recalls))
        
        for i in torch.arange(precisions_cumsum.size(0) - 1, 0, -1):
            precisions_cumsum[i-1] = torch.max(precisions_cumsum[i-1], precisions_cumsum[i])

        ii = []
        for i in torch.arange(recalls_cumsum.size(0) - 1):
            if recalls_cumsum[i+1] != recalls_cumsum[i]:
                ii.append(i+1)
            
        ap = torch.zeros(1)    
        for i in ii:
            ap += (recalls_cumsum[i] - recalls_cumsum[i - 1]) * precisions_cumsum[i]
        
        average_precisions.append(ap)

    return torch.mean(torch.stack(average_precisions))

def metrics_per_class(true_boxes, pred_boxes, num_classes, iou_threshold=0.5):
    """Calculate metrics per class: average precision, true positive, false positive, false negative

    Arguments:
        true_boxes (Tensor): Tensor of all boxes with all images (None, 7), specified as [img_idx, cx, cy, w, h, confidence_score, class_idx]
        pred_boxes (Tensor): Similar as true_bboxes
        num_classes (int): number of classes
        iou_threshold (float): threshold where predicted boxes is correct

    Returns:
        Tensor: metrics per class (num_classes, 4), specified as [average precision, true positive, false positive, false negative]
    """

    # list storing all AP for respective classes
    AP = []
    TP = []
    FP = []
    FN = []

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
            AP.append(torch.zeros(1))
            TP.append(torch.zeros(1))
            FP.append(torch.zeros(1))
            FN.append(torch.zeros(1))
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

        tp_cumsum = torch.cumsum(true_positive, dim=0)
        fp_cumsum = torch.cumsum(false_positive, dim=0)
        recalls_cumsum = tp_cumsum / (total_true_boxes + epsilon)
        precisions_cumsum = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))
        precisions_cumsum = torch.cat((torch.tensor([1]), precisions_cumsum))
        recalls_cumsum = torch.cat((torch.tensor([0]), recalls_cumsum))
        
        for i in torch.arange(precisions_cumsum.size(0) - 1, 0, -1):
            precisions_cumsum[i-1] = torch.max(precisions_cumsum[i-1], precisions_cumsum[i])

        ii = []
        for i in torch.arange(recalls_cumsum.size(0) - 1):
            if recalls_cumsum[i+1] != recalls_cumsum[i]:
                ii.append(i+1)
            
        ap = torch.zeros(1)    
        for i in ii:
            ap += (recalls_cumsum[i] - recalls_cumsum[i - 1]) * precisions_cumsum[i]
        
        AP.append(ap)
        TP.append(torch.sum(true_positive, dim=0, keepdim=True))
        FP.append(torch.sum(false_positive, dim=0, keepdim=True))
        FN.append(torch.tensor([total_true_boxes - torch.sum(true_positive)]))

    AP = torch.stack(AP) # (num_classes, 1)
    TP = torch.stack(TP) # (num_classes, 1)
    FP = torch.stack(FP) # (num_classes, 1)
    FN = torch.stack(FN) # (num_classes, 1)

    return torch.cat([AP, TP, FP, FN], dim=1)


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

        # print(f'{class_name}: {xmin}, {ymin}, {xmax}, {ymax}')

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
        img = cv2.putText(img, 
                          f'{class_name}, {confidence_score:.2f}',
                          (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=color)
    # print(f'')
    return img


def get_tagged_lpr_img(img, boxes, names_path, color):
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
    
    font = ImageFont.truetype('malgun.ttf', 20)
    
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    
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

        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=1)
        draw.text(((xmin, ymin)), f'{class_name}', font=font, fill=color)

    return np.array(img)


def get_target_boxes(target, input_size):
    """Decode Yolo Ground Truth to Bounding Boxes

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
    """Decode Yolo Ground Truth to Bounding Boxes

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

        if not tmp:
            tmp = np.zeros((0, 6))
        dst[b] = torch.FloatTensor(tmp)

    return dst