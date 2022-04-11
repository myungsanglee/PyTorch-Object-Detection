import torch
import numpy as np

from dataset.detection.utils import non_max_suppression, decode_predictions


class MeanAveragePrecision:
    def __init__(self, num_classes, num_boxes):
        self.all_true_boxes_variable = 0
        self.all_pred_boxes_variable = 0
        self.img_idx = 0
        self._num_classes = num_classes
        self._num_boxes = num_boxes

    def reset_states(self):
        self.img_idx = 0

    def update_state(self, y_true, y_pred):
        true_boxes = decode_predictions(y_true, self._num_classes, self._num_boxes)
        pred_boxes = decode_predictions(y_pred, self._num_classes, self._num_boxes)

        for idx in torch.arange(y_true.size()[0]):
            pred_nms = non_max_suppression(pred_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            pred_img_idx = torch.zeros([pred_nms.size()[0], 1], torch.float32) + self.img_idx
            pred_concat = torch.concat([pred_img_idx, pred_nms], dim=1)

            true_nms = non_max_suppression(true_boxes[idx], iou_threshold=0.5, conf_threshold=0.4)
            true_img_idx = torch.zeros([true_nms.size()[0], 1], torch.float32) + self.img_idx
            true_concat = torch.concat([true_img_idx, true_nms], dim=1)

            if self.img_idx == 0.:
                self.all_true_boxes_variable = true_concat
                self.all_pred_boxes_variable = pred_concat
            else:
                self.all_true_boxes_variable = torch.concat([self.all_true_boxes_variable, true_concat], axis=0)
                self.all_pred_boxes_variable = torch.concat([self.all_pred_boxes_variable, pred_concat], axis=0)

            self.img_idx += 1

    def result(self):
        return mean_average_precision(self.all_true_boxes_variable, self.all_pred_boxes_variable, self._num_classes)
   
    
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
        print('Calculating AP: ', c, ' / ', num_classes)

        # detections, ground_truths variables in specific class
        detections = pred_boxes[torch.where(pred_boxes[..., 1] == c)[0]]
        ground_truths = true_boxes[torch.where(true_boxes[..., 1] == c)[0]]

        # If none exists for this class then we can safely skip
        total_true_boxes = torch.tensor(ground_truths.size()[0], dtype=torch.float32)
        if total_true_boxes == 0.:
            average_precisions = average_precisions.append(torch.tensor(0, dtype=torch.float32))
            continue

        # tf.print(c, ' class ground truths size: ', tf.shape(ground_truths)[0])
        # tf.print(c, ' class detections size: ', tf.shape(detections)[0])

        # Get the number of true boxes by image index
        img_idx, idx, count = tf.unique_with_counts(ground_truths[..., 0])
        img_idx = tf.cast(img_idx, dtype=tf.int32)

        # Convert idx to idx tensor for find num of true boxes by img idx
        idx_tensor = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        for i in tf.range(tf.math.reduce_max(idx) + 1):
            idx_tensor = idx_tensor.write(idx_tensor.size(), i)
        idx_tensor = idx_tensor.stack()

        # Get hash table: key - img_idx, value - idx_tensor
        table = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int32,
            value_dtype=tf.int32,
            default_value=-1,
            empty_key=-1,
            deleted_key=-2
        )
        table.insert(img_idx, idx_tensor)

        # Get true boxes num array
        ground_truth_num = tf.TensorArray(tf.int32, size=tf.math.reduce_max(idx) + 1, dynamic_size=True, clear_after_read=False)
        for i in tf.range(tf.math.reduce_max(idx) + 1):
            ground_truth_num = ground_truth_num.write(i, tf.zeros(tf.math.reduce_max(count), dtype=tf.int32))

        # sort by confidence score
        detections = tf.gather(detections, tf.argsort(detections[..., 2], direction='DESCENDING'))
        true_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())
        false_positive = tf.TensorArray(tf.float32, size=tf.shape(detections)[0], element_shape=())

        detections_size = tf.shape(detections)[0]
        detection_idx = tf.constant(0, dtype=tf.int32)
        for detection in detections:
            # tf.print('progressing of detection: ', detection_idx, ' / ', detections_size)
            # tf.print('detection_img_idx: ', detection[0])
            # tf.print('detection_confidence: ', detection[2])

            ground_truth_img = tf.gather(ground_truths, tf.reshape(tf.where(ground_truths[..., 0] == detection[0]),
                                                                   shape=(-1,)))
            # tf.print('ground_truth_img: ', tf.shape(ground_truth_img)[0])

            best_iou = tf.TensorArray(tf.float32, size=1, element_shape=(1,), clear_after_read=False)
            best_gt_idx = tf.TensorArray(tf.int32, size=1, element_shape=(), clear_after_read=False)

            gt_idx = tf.constant(0, dtype=tf.int32)
            for gt_img in ground_truth_img:
                iou = intersection_over_union(detection[3:], gt_img[3:])

                if iou > best_iou.read(0):
                    best_iou = best_iou.write(0, iou)
                    best_gt_idx = best_gt_idx.write(0, gt_idx)

                gt_idx += 1

            if best_iou.read(0) > iou_threshold:
                # Get current detections img_idx
                cur_det_img_idx = tf.cast(detection[0], dtype=tf.int32)

                # Get row idx of ground_truth_num array
                gt_row_idx = table.lookup(cur_det_img_idx)

                # Get 'current img ground_truth_num tensor'
                cur_gt_num_tensor = ground_truth_num.read(gt_row_idx)

                # Get idx of current best ground truth
                cur_best_gt_idx = best_gt_idx.read(0)

                if cur_gt_num_tensor[cur_best_gt_idx] == 0:
                    true_positive = true_positive.write(detection_idx, 1)

                    # change cur_gt_num_tensor[cur_best_gt_idx] to 1
                    cur_gt_num_tensor = change_tensor(cur_gt_num_tensor, cur_best_gt_idx)

                    # update ground_truth_num array
                    ground_truth_num = ground_truth_num.write(gt_row_idx, cur_gt_num_tensor)

                else:
                    false_positive = false_positive.write(detection_idx, 1)

            # if IOU is lower then the detection is a false positive
            else:
                false_positive = false_positive.write(detection_idx, 1)

            # ground_truth_img.close()
            best_iou.close()
            best_gt_idx.close()
            detection_idx += 1

        # Compute the cumulative sum of the tensor
        tp_cumsum = tf.math.cumsum(true_positive.stack(), axis=0)
        fp_cumsum = tf.math.cumsum(false_positive.stack(), axis=0)

        # Calculate recalls and precisions
        recalls = tf.math.divide(tp_cumsum, (total_true_boxes + epsilon))
        precisions = tf.math.divide(tp_cumsum, (tp_cumsum + fp_cumsum + epsilon))

        # Append start point value of precision-recall graph
        precisions = tf.concat([tf.constant([1], dtype=tf.float32), precisions], axis=0)
        recalls = tf.concat([tf.constant([0], dtype=tf.float32), recalls], axis=0)
        # tf.print(precisions)
        # tf.print(recalls)

        # Calculate area of precision-recall graph
        average_precision_value = tf.py_function(func=np.trapz,
                                                 inp=[precisions, recalls],
                                                 Tout=tf.float32)
        average_precisions = average_precisions.write(average_precisions.size(), average_precision_value)
        # tf.print('average precision: ', average_precision_value)

        ground_truth_num.close()
        true_positive.close()
        false_positive.close()

    # tf.print(average_precisions.stack())
    # tf.print('mAP: ', tf.math.reduce_mean(average_precisions.stack()))
    return tf.math.reduce_mean(average_precisions.stack())