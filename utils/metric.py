import sys
import os
sys.path.append(os.getcwd())

from collections import Counter

import torch
import numpy as np

from dataset.detection.yolov1_utils import non_max_suppression, decode_predictions, intersection_over_union


class MeanAveragePrecision:
    def __init__(self, num_classes, num_boxes):
        self._all_true_boxes_variable = 0
        self._all_pred_boxes_variable = 0
        self._img_idx = 0
        self._num_classes = num_classes
        self._num_boxes = num_boxes

    def reset_states(self):
        self._all_true_boxes_variable = 0
        self._all_pred_boxes_variable = 0
        self._img_idx = 0

    def update_state(self, y_true, y_pred):
        true_boxes = decode_predictions(y_true, self._num_classes, self._num_boxes)
        pred_boxes = decode_predictions(y_pred, self._num_classes, self._num_boxes)

        for idx in torch.arange(y_true.size()[0]):
            pred_nms = non_max_suppression(pred_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            pred_img_idx = torch.zeros([pred_nms.size()[0], 1], dtype=torch.float32) + self._img_idx
            if pred_nms.is_cuda:
                pred_img_idx = pred_img_idx.cuda()
            pred_concat = torch.concat([pred_img_idx, pred_nms], dim=1)

            true_nms = non_max_suppression(true_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            true_img_idx = torch.zeros([true_nms.size()[0], 1], dtype=torch.float32) + self._img_idx
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
    
    
def mean_average_precision(true_boxes, pred_boxes, num_classes, iou_threshold=0.5):
    """Calculates mean average precision

    Arguments:
        true_boxes (Tensor): Tensor of all boxes with all images (None, 7), specified as [img_idx, class_idx, confidence_score, cx, cy, w, h]
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
        detections = pred_boxes[torch.where(pred_boxes[..., 1] == c)[0]]
        ground_truths = true_boxes[torch.where(true_boxes[..., 1] == c)[0]]

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
        detections = detections[torch.sort(detections[..., 2], descending=True)[1]]
        true_positive = torch.zeros((len(detections)))
        false_positive = torch.zeros((len(detections)))

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = ground_truths[torch.where(ground_truths[..., 0] == detection[0])[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(detection[3:], gt[3:])

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


if __name__ == '__main__':
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
    
    # mAP Test
    map_metric = MeanAveragePrecision(num_classes, num_boxes)
    map_metric.update_state(y_true_tensor, y_pred_tensor)
    map = map_metric.result()    
    print(map)
    map_metric.reset_states()
    
    map_metric.update_state(y_true_tensor, y_pred_tensor)
    map = map_metric.result()    
    print(map)
    