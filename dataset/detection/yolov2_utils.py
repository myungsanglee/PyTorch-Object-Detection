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


def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.25):
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
            elif torch.lt(intersection_over_union(chosen_box[..., 2:], tmp_box[..., 2:]), iou_threshold):
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
        Tensor: boxes after decoding predictions '(batch, num_anchors*13*13, 5+num_classes)', specified as [cx, cy, w, h, confidence_score, class_idx]
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

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    # Calculate offsets for each grid
    grid_x = torch.linspace(0, layer_w-1, layer_w).repeat(layer_w, 1).repeat(
        batch_size * num_anchors, 1, 1).view(x.shape).type(FloatTensor)
    grid_y = torch.linspace(0, layer_h-1, layer_h).repeat(layer_h, 1).t().repeat(
        batch_size * num_anchors, 1, 1).view(y.shape).type(FloatTensor)
    # Calculate anchor w, h
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, layer_h * layer_w).view(w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, layer_h * layer_w).view(h.shape)
    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h
    # Results
    _scale = torch.FloatTensor([stride_w, stride_h] * 2)
    output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                        conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)), -1)
    
    return output
    


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

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color)
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
        decode_pred = decode_predictions(x, self.num_classes, self.num_boxes)
        boxes = non_max_suppression(decode_pred[0], conf_threshold=self.conf_threshold)
        return boxes


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
        
    print(f'-'*100)
    
    a = torch.FloatTensor([[0, 0, 0.5, 0.5]])
    b = torch.FloatTensor([[0, 0, 0.4, 0.4], [0, 0, 0.5, 0.5]])
    print(intersection_over_union(a, b))