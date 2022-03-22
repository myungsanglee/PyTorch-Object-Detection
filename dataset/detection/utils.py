import time

import torch
import numpy as np
import cv2


def collater(data):
    """Data Loader에서 생성된 데이터를 동일한 shape으로 정렬해서 Batch로 전달

    Args:
        data ([dict]): albumentation Transformed 객체
        'image': list of Torch Tensor len == batch_size, item shape = ch, h, w
        'bboxes': list of list([x1, y1, w, h, cid])

    Returns:
        [dict]: 정렬된 batch data.
        'img': list of image tensor
        'annot': 동일 shape으로 정렬된 tensor [x1,y1,x2,y2] format
    """
    imgs = [s['image'] for s in data]
    bboxes = [torch.tensor(s['bboxes'])for s in data]
    batch_size = len(imgs)

    max_num_annots = max(annots.shape[0] for annots in bboxes)

    if max_num_annots > 0:
        padded_annots = torch.ones((batch_size, max_num_annots, 5)) * -1
        for idx, annot in enumerate(bboxes):
            if annot.shape[0] > 0:
                # To x1, y1, x2, y2
                annot[:, 2] += annot[:, 0]
                annot[:, 3] += annot[:, 1]
                padded_annots[idx, :annot.shape[0], :] = annot
    else:
        padded_annots = torch.ones((batch_size, 1, 5)) * -1

    return {'img': torch.stack(imgs), 'annot': padded_annots}


def visualize(images, bboxes, batch_idx=0):
    """batch data를 opencv로 visualize

    Args:
        images ([list]): list of img tensor
        bboxes ([tensor]): tensor data of annotations
                        shape == [batch, max_annots, 5(x1,y1,x2,y2,cid)]
                        max_annots 은 batch sample 중 가장 많은 bbox 갯수.
                        다른 sample 은 -1 로 패딩된 데이터가 저장됨.
        batch_idx (int, optional): [description]. Defaults to 0.
    """
    img = images[batch_idx].numpy()
    img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

    for b in bboxes[batch_idx]:
        x1, y1, x2, y2, cid = b.numpy()
        if cid > -1:
            img = cv2.rectangle(img, (int(x1), int(y1)),
                                (int(x2), int(y2)), (0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def get_tagged_img(img, boxes, names_path):
    """tagging result on img

    Arguments:
        img (Numpy Array): Image array
        boxes (Tensor): boxes after performing NMS (None, 6)
        names_path (String): path of label names file

    Returns:
        Numpy Array: tagged image array
    """

    width = img.shape[1]
    height = img.shape[0]
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
        xmin = int((x - (w / 2)) * width)
        ymin = int((y - (h / 2)) * height)
        xmax = int((x + (w / 2)) * width)
        ymax = int((y + (h / 2)) * height)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 255, 0))

    return img


def intersection_over_union_numpy(boxes1, boxes2):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]
        boxes2 (Numpy Array): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]

    Returns:
        Numpy Array: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
    """

    box1_xmin = (boxes1[..., 0:1] - boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymin = (boxes1[..., 1:2] - boxes1[..., 3:4]) / 2. # (batch, S, S, 1)
    box1_xmax = (boxes1[..., 0:1] + boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymax = (boxes1[..., 1:2] + boxes1[..., 3:4]) / 2. # (batch, S, S, 1)

    box2_xmin = (boxes2[..., 0:1] - boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymin = (boxes2[..., 1:2] - boxes2[..., 3:4]) / 2. # (batch, S, S, 1)
    box2_xmax = (boxes2[..., 0:1] + boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymax = (boxes2[..., 1:2] + boxes2[..., 3:4]) / 2. # (batch, S, S, 1)

    inter_xmin = np.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = np.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = np.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = np.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = np.clip((inter_xmax - inter_xmin), 0, 1) * np.clip((inter_ymax - inter_ymin), 0, 1) # (batch, S, S, 1)
    box1_area = np.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = np.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


def intersection_over_union(boxes1, boxes2):
    """Calculation of intersection-over-union

    Arguments:
        boxes1 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]
        boxes2 (Tensor): boxes with shape '(batch, S, S, 4) or (batch, num_boxes, 4) or (num_boxes, 4)', specified as [cx, cy, w, h]

    Returns:
        Tensor: IoU with shape '(batch, S, S, 1) or (batch, num_boxes, 1) or (num_boxes, 1)'
    """

    box1_xmin = (boxes1[..., 0:1] - boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymin = (boxes1[..., 1:2] - boxes1[..., 3:4]) / 2. # (batch, S, S, 1)
    box1_xmax = (boxes1[..., 0:1] + boxes1[..., 2:3]) / 2. # (batch, S, S, 1)
    box1_ymax = (boxes1[..., 1:2] + boxes1[..., 3:4]) / 2. # (batch, S, S, 1)

    box2_xmin = (boxes2[..., 0:1] - boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymin = (boxes2[..., 1:2] - boxes2[..., 3:4]) / 2. # (batch, S, S, 1)
    box2_xmax = (boxes2[..., 0:1] + boxes2[..., 2:3]) / 2. # (batch, S, S, 1)
    box2_ymax = (boxes2[..., 1:2] + boxes2[..., 3:4]) / 2. # (batch, S, S, 1)

    inter_xmin = torch.maximum(box1_xmin, box2_xmin) # (batch, S, S, 1)
    inter_ymin = torch.maximum(box1_ymin, box2_ymin) # (batch, S, S, 1)
    inter_xmax = torch.minimum(box1_xmax, box2_xmax) # (batch, S, S, 1)
    inter_ymax = torch.minimum(box1_ymax, box2_ymax) # (batch, S, S, 1)

    inter_area = torch.clamp((inter_xmax - inter_xmin), 0, 1) * torch.clamp((inter_ymax - inter_ymin), 0, 1) # (batch, S, S, 1)
    box1_area = torch.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = torch.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


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
            if tmp_box[0][0] != chosen_box[0][0] or intersection_over_union_numpy(chosen_box[..., 2:], tmp_box[..., 2:]) < iou_threshold:
                tmp_boxes = np.append(tmp_boxes, tmp_box, axis=0)
        boxes = tmp_boxes

        boxes_after_nms = np.append(boxes_after_nms, chosen_box, axis=0)

    return boxes_after_nms


def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """Does Non Max Suppression given boxes

    Arguments:
        boxes (Tensor): All boxes with each grid '(S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
        iou_threshold (float): threshold where predicted boxes is correct
        conf_threshold (float): threshold to remove predicted boxes

    Returns:
        Tensor: boxes after performing NMS given a specific IoU threshold '(None, 6)'
    """

    # boxes smaller than the conf_threshold are removed
    boxes = boxes[torch.where(boxes[..., 1] > conf_threshold)[0]]

    # sort descending by confidence score
    boxes = boxes[torch.argsort(-boxes[..., 1])]
 
    # get boxes after nms
    boxes_after_nms = []

    while True:
        chosen_box = boxes[:1, ...]
        boxes_after_nms.append(chosen_box[0])
        
        tmp_boxes = []
        for idx in range(1, boxes.shape[0]):
            tmp_box = boxes[idx:idx+1, ...]
            if tmp_box[0][0] != chosen_box[0][0]:
                tmp_boxes.append(tmp_box[0])
            elif torch.lt(intersection_over_union(chosen_box[..., 2:], tmp_box[..., 2:]), iou_threshold):
                tmp_boxes.append(tmp_box[0])
                
        if tmp_boxes:
            boxes = torch.stack(tmp_boxes)
        else:
            break

    return torch.stack(boxes_after_nms)


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


def decode_predictions(predictions, num_classes, num_boxes=2):
    """decodes predictions of the YOLO v1 model
    
    Converts bounding boxes output from Yolo with
    an image split size of GRID into entire image ratios
    rather than relative to cell ratios.

    Arguments:
        predictions (Tensor): predictions of the YOLO v1 model with shape  '(1, 7, 7, (num_boxes*5 + num_classes))'
        num_classes: Number of classes in the dataset
        num_boxes: Number of boxes to predict

    Returns:
        Tensor: boxes after decoding predictions each grid cell with shape '(batch, S*S, 6)', specified as [class_idx, confidence_score, cx, cy, w, h]
    """

    # Get class indexes
    class_indexes = torch.argmax(predictions[..., :num_classes], dim=-1, keepdim=True) # (batch, S, S, 1)
    class_indexes = class_indexes.type(torch.float32)
    
    # Get best confidence one-hot
    confidences = []
    for idx in torch.arange(num_boxes):
        confidence = predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]
        confidences.append(confidence)
    confidences = torch.stack(confidences) # (num_boxes, batch, S, S, 1)
    best_conf_idx = torch.argmax(confidences, dim=0) # (batch, S, S, 1)
    best_conf_one_hot = torch.nn.functional.one_hot(best_conf_idx, num_boxes).view(-1, 7, 7, num_boxes) # (batch, S, S, num_boxes)

    # Get prediction box & confidence
    pred_box = torch.zeros([1, 7, 7, 4])
    pred_conf = torch.zeros([1, 7, 7, 1])
    for idx in torch.arange(num_boxes):
        pred_box += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(1+(5*idx)):num_classes+(1+(5*idx))+4]
        pred_conf += best_conf_one_hot[..., idx:idx+1] * predictions[..., num_classes+(5*idx):num_classes+(5*idx)+1]

    # Get cell indexes array
    base_arr = torch.arange(7).reshape((1, -1)).repeat(7, 1)
    x_cell_indexes = torch.unsqueeze(base_arr, dim=-1)  # (S, S, 1)

    y_cell_indexes = np.transpose(base_arr)
    y_cell_indexes = torch.unsqueeze(y_cell_indexes, dim=-1)  # (S, S, 1)

    # Convert x, y ratios to YOLO ratios
    x = 1 / 7 * (pred_box[..., :1] + x_cell_indexes) # (batch, S, S, 1)
    y = 1 / 7 * (pred_box[..., 1:2] + y_cell_indexes) # (batch, S, S, 1)

    pred_box = torch.cat([x, y, pred_box[..., 2:4]], dim=-1) # (batch, S, S, 4)

    # Concatenate result
    pred_result = torch.cat([class_indexes, pred_conf, pred_box], dim=-1) # (batch, S, S, 6)

    # Get all bboxes
    pred_result = pred_result.view(-1, 7*7, 6) # (batch, S*S, 6)
    
    return pred_result


def get_tagged_img(img, boxes, names_path):
    """tagging result on img

    Arguments:
        img (Numpy Array): Image array
        boxes (Tensor): boxes after performing NMS (None, 6)
        names_path (String): path of label names file

    Returns:
        Numpy Array: tagged image array
    """

    width = img.shape[1]
    height = img.shape[0]
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
        xmin = int((x - (w / 2)) * width)
        ymin = int((y - (h / 2)) * height)
        xmax = int((x + (w / 2)) * width)
        ymax = int((y + (h / 2)) * height)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
        img = cv2.putText(img, "{:s}, {:.2f}".format(class_name, confidence_score), (xmin, ymin + 20),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,
                          fontScale=1,
                          color=(0, 255, 0))

    return img


if __name__ == '__main__':
    num_classes = 3
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
    y_pred[:, 0, 0, :num_classes] = [0.8, 0.5, 0.1] # class
    y_pred[:, 0, 0, num_classes] = 0.6 # confidence1
    y_pred[:, 0, 0, num_classes+1:num_classes+5] = [0.49, 0.49, 0.1, 0.1] # box1
    y_pred[:, 0, 0, num_classes+5] = 0.2 # confidence2
    y_pred[:, 0, 0, num_classes+6:num_classes+10] = [0.45, 0.45, 0.1, 0.1] # box2
    
    y_pred[:, 3, 3, :num_classes] = [0.2, 0.8, 0.1] # class
    y_pred[:, 3, 3, num_classes] = 0.1 # confidence1
    y_pred[:, 3, 3, num_classes+1:num_classes+5] = [0.45, 0.45, 0.1, 0.1] # box1
    y_pred[:, 3, 3, num_classes+5] = 0.9 # confidence2
    y_pred[:, 3, 3, num_classes+6:num_classes+10] = [0.49, 0.49, 0.1, 0.1] # box2
    
    y_pred[:, 6, 6, :num_classes] = [0.1, 0.5, 0.8] # class
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
    print(y_true_bbox)
    print(y_pred_bbox)
    print(intersection_over_union_numpy(y_true_bbox, y_pred_bbox))
    # torch tensor
    y_true_bbox = y_true_tensor[:, 0, 0, num_classes+1:num_classes+5]
    y_pred_bbox = y_pred_tensor[:, 0, 0, num_classes+1:num_classes+5]
    print(y_true_bbox)
    print(y_pred_bbox)
    print(intersection_over_union(y_true_bbox, y_pred_bbox))
    
    
    # Decode Prediction Test
    # numpy array
    decode_pred = decode_predictions_numpy(y_pred, num_classes, num_boxes)
    print(decode_pred)
    # torch tensor
    decode_pred_tensor = decode_predictions(y_pred_tensor, num_classes, num_boxes)
    print(decode_pred_tensor)
    
    
    # Non Max Suppression Test
    # numpy array
    bboxes = non_max_suppression_numpy(decode_pred[0])
    print(bboxes)
    # torch tensor
    bboxes_tensor = non_max_suppression(decode_pred_tensor[0])
    print(bboxes_tensor) 