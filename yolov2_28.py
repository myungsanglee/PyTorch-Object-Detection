import platform
import os
import time
from collections import Counter
from bisect import bisect_left

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, _LRScheduler
import torchsummary
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


######################################################################################################################
# Set Hyperparameter
######################################################################################################################
def get_cfg():
    cfg = dict()

    cfg['model'] = 'yolov2'
    cfg['backbone'] = 'darknet19'
    cfg['dataset_name'] = 'voc'
    cfg['input_size'] = 416
    cfg['in_channels'] = 3
    cfg['num_classes'] = 20
    cfg['scaled_anchors'] = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    cfg['epochs'] = 365
    
    cfg['train_list'] = '/home/fssv2/myungsang/datasets/voc/yolo_format/train.txt'
    cfg['val_list'] = '/home/fssv2/myungsang/datasets/voc/yolo_format/val.txt'
    cfg['names'] = '/home/fssv2/myungsang/datasets/voc/yolo_format/voc.names'
    cfg['workers'] = 32
    cfg['batch_size'] = 64
    
    cfg['save_dir'] = './saved'
    cfg['save_freq'] = 5

    cfg['trainer_options'] = {
        'check_val_every_n_epoch': 5,
        'num_sanity_val_steps': 0
    }

    cfg['accelerator'] = 'gpu'
    cfg['devices'] = [0]

    cfg['optimizer'] = 'sgd'
    cfg['optimizer_options'] = {
        'lr': 1e-4,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True
    }

    cfg['scheduler'] = 'yolo_lr'
    cfg['scheduler_options'] = {
        'burn_in': 1000,
        'steps': [40000],
        'scales': [0.1]
    }

    return cfg


######################################################################################################################
# Utility
######################################################################################################################
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


def get_model(model_name):
    model_dict = {
        'darknet19': darknet19
    }
    return model_dict.get(model_name)


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {
        'sgd': optim.SGD, 
        'adam': optim.Adam,
        'radam': optim.RAdam,
        'adamw': optim.AdamW
    }
    optimizer = optim_dict.get(optimizer_name)
    if optimizer:
        return optimizer(params, **kwargs)


def get_scheduler(scheduler_name, optim, **kwargs):
    scheduler_dict = {
        'multi_step': MultiStepLR, 
        'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
        'yolo_lr': YoloLR
    }
    scheduler = scheduler_dict.get(scheduler_name)
    if scheduler:
        return scheduler(optim, **kwargs)


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

    inter_area = torch.clamp((inter_xmax - inter_xmin), 0) * torch.clamp((inter_ymax - inter_ymin), 0) # (batch, S, S, 1)
    box1_area = torch.abs((box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)) # (batch, S, S, 1)
    box2_area = torch.abs((box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)) # (batch, S, S, 1)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6) # (batch, S, S, 1)


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
            elif torch.lt(intersection_over_union(chosen_box[..., :4], tmp_box[..., :4]), iou_threshold):
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
                iou = intersection_over_union(detection[1:5], gt[1:5])

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

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color)
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
            if pred_nms.is_cuda:
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


class YoloLR(_LRScheduler):
    def __init__(self, optimizer, burn_in, steps, scales, last_epoch=-1):
        self.burn_in = burn_in
        self.steps = steps
        self.scales = scales
        self.scale = 1.
        super(YoloLR, self).__init__(optimizer, last_epoch)     
    
    def get_lr(self):
        if self.last_epoch < self.burn_in:
            return [base_lr * pow(self.last_epoch/self.burn_in, 4) for base_lr in self.base_lrs]
        else:
            if self.last_epoch  < self.steps[0]:
                return self.base_lrs
            else:
                if self.last_epoch in self.steps:
                    self.scale *= self.scales[bisect_left(self.steps, self.last_epoch)]
                return [base_lr * self.scale for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


######################################################################################################################
# Data Module
######################################################################################################################
class YoloV2Dataset(Dataset):
    def __init__(self, transforms, files_list):
        super().__init__()

        with open(files_list, 'r') as f:
            self.imgs = f.read().splitlines()
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self._get_boxes(img_file.replace('.jpg', '.txt'))
        
        transformed = self.transforms(image=img, bboxes=boxes)
        
        return transformed

    def _get_boxes(self, label_path):
        boxes = np.zeros((0, 5))
        with open(label_path, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                class_id, cx, cy, w, h = map(float, annot.split(' '))
                annotation = np.array([[cx, cy, w, h, class_id]])
                boxes = np.append(boxes, annotation, axis=0)

        return boxes


class YoloV2DataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, input_size, batch_size):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.workers = workers
        self.input_size = input_size
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        train_transforms = A.Compose([
            A.HorizontalFlip(),
            A.CLAHE(),
            A.ColorJitter(
                brightness=0.5,
                contrast=0.2,
                saturation=0.5,
                hue=0.1
            ),
            A.RandomResizedCrop(self.input_size, self.input_size, (0.3, 1)),
            A.Normalize(0, 1),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

        valid_transform = A.Compose([
            A.Resize(self.input_size, self.input_size, always_apply=True),
            A.Normalize(0, 1),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
        
        self.train_dataset = YoloV2Dataset(
            train_transforms, 
            self.train_list
        )
        
        self.valid_dataset = YoloV2Dataset(
            valid_transform, 
            self.val_list
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
            collate_fn=collater
        )


######################################################################################################################
# Darknet19 Model
######################################################################################################################
class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, padding_mode='zeros'):
        super(Conv2dBnRelu, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, False,
                              padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return self.relu(y)


class _Darknet19(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(_Darknet19, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # config out_channels, kernel_size
        stem = [
            [32, 3]
        ]
        layer1 = [
            'M',
            [64, 3]
        ]
        layer2 = [
            'M',
            [128, 3],
            [64, 1],
            [128, 3]
        ]
        layer3 = [
            'M',
            [256, 3],
            [128, 1],
            [256, 3]
        ]
        layer4 = [
            'M',
            [512, 3],
            [256, 1],
            [512, 3],
            [256, 1],
            [512, 3]
        ]
        layer5 = [
            'M',
            [1024, 3],
            [512, 1],
            [1024, 3],
            [512, 1],
            [1024, 3],
        ]

        self.stem = self._make_layers(stem)
        self.layer1 = self._make_layers(layer1)
        self.layer2 = self._make_layers(layer2)
        self.layer3 = self._make_layers(layer3)
        self.layer4 = self._make_layers(layer4)
        self.layer5 = self._make_layers(layer5)

        self.classifier = nn.Sequential(
            Conv2dBnRelu(1024, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.classifier(x)
        return x

    def _make_layers(self, layer_cfg):
        layers = []

        for cfg in layer_cfg:
            if cfg == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(Conv2dBnRelu(self.in_channels, cfg[0], cfg[1]))
                self.in_channels = cfg[0]

        return nn.Sequential(*layers)


def darknet19(num_classes=1000, in_channels=3):
    model = _Darknet19(num_classes, in_channels)
    return model


######################################################################################################################
# Yolo v2 Model
######################################################################################################################
class YoloV2(nn.Module):
    def __init__(self, backbone, num_classes, num_anchors):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.b4_layer = nn.Sequential(
            Conv2dBnRelu(512, 64, 1)
        )

        self.b5_layer = nn.Sequential(
            Conv2dBnRelu(1024, 1024, 3),
            Conv2dBnRelu(1024, 1024, 3)
        )
        
        self.yolov2_head = nn.Sequential(
            Conv2dBnRelu(1280, 1024, 3),
            
            nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        b1 = self.backbone.layer1(x)
        b2 = self.backbone.layer2(b1)
        b3 = self.backbone.layer3(b2)
        b4 = self.backbone.layer4(b3)
        b5 = self.backbone.layer5(b4)

        b4 = self.b4_layer(b4)
        bs, _, h, w = b4.size()
        b4 = b4.view(bs, -1, h//2, w//2)

        b5 = self.b5_layer(b5)

        x = torch.cat((b4, b5), 1)

        x = self.dropout(x)

        # prediction
        predictions = self.yolov2_head(x)

        return predictions


######################################################################################################################
# Yolo v2 Loss Function
######################################################################################################################
class YoloV2Loss(nn.Module):
    """YoloV2 Loss Function

    Arguments:
      num_classes (int): Number of classes in the dataset
      scaled_anchors (List): Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
    """
    
    def __init__(self, num_classes, scaled_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.scaled_anchors = scaled_anchors

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_obj = 5
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.lambda_class = 1
        
        self.ignore_threshold = 0.5
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, input, target):
        """
        Arguments:
            input (tensor): [batch, num_anchors*(5+num_classes), 13, 13]
            target (tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            
        Returns:
            loss (float): total loss values
        """
        batch_size, _, layer_h, layer_w = input.size()
        # [batch_size, num_anchors, 5+num_classes, layer_h, layer_w] to [batch_size, num_anchors, layer_h, layer_w, 5+num_classes]
        prediction = input.view(batch_size, len(self.scaled_anchors), -1, layer_h, layer_w).permute(0, 1, 3, 4, 2).contiguous()
        
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.exp(prediction[..., 2])
        h = torch.exp(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.encode_target(target, self.num_classes, self.scaled_anchors, layer_w, layer_h, self.ignore_threshold)
        if prediction.is_cuda:
            mask = mask.cuda()
            noobj_mask = noobj_mask.cuda()
            tx = tx.cuda()
            ty = ty.cuda()
            tw = tw.cuda()
            th = th.cuda()
            tconf = tconf.cuda()
            tcls = tcls.cuda()

        # ============================ #
        #   FOR BOX COORDINATES Loss   #
        # ============================ #
        loss_x = self.mse_loss(x * mask, tx * mask)
        loss_y = self.mse_loss(y * mask, ty * mask)
        loss_w = self.mse_loss(w * mask, tw * mask)
        loss_h = self.mse_loss(h * mask, th * mask)
        box_loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        object_loss = self.lambda_obj * self.mse_loss(conf * mask, tconf)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.lambda_noobj * self.mse_loss(conf * noobj_mask, noobj_mask * 0.0)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.lambda_class * self.bce_loss(pred_cls[mask==1], tcls[mask==1])

        loss = (box_loss + object_loss + no_object_loss + class_loss) / batch_size

        return loss


    def encode_target(self, target, num_classes, scaled_anchors, layer_w, layer_h, ignore_threshold):
        """YoloV2 Loss Function

        Arguments:
            target (Tensor): [batch, max_num_annots, 5(cx, cy, w, h, cid)]
            num_classes (int): Number of classes in the dataset
            scaled_anchors (List): Scaled Anchors of a specific dataset, [num_anchors, 2(scaled_w, scaled_h)]
            layer_w (int): size of predictions width
            layer_h (int): size of predictions height
            ignore_threshold (float): float value of ignore iou
        
        Retruns:
            mask (Tensor): Objectness Mask Tensor, [batch_size, num_anchors, layer_h, layer_w]
            noobj_mask (Tensor): No Objectness Mask Tensor, [batch_size, num_anchors, layer_h, layer_w]
            tx (Tensor): Ground Truth X, [batch_size, num_anchors, layer_h, layer_w]
            ty (Tensor): Ground Truth Y, [batch_size, num_anchors, layer_h, layer_w]
            tw (Tensor): Ground Truth W, [batch_size, num_anchors, layer_h, layer_w]
            th (Tensor): Ground Truth H, [batch_size, num_anchors, layer_h, layer_w]
            tconf (Tensor): Ground Truth Confidence Score, [batch_size, num_anchors, layer_h, layer_w]
            tcls (Tensor): Ground Truth Class index, [batch_size, num_anchors, layer_h, layer_w, num_classes]
        """
        batch_size = target.size(0)
        num_anchors = len(scaled_anchors)
        
        mask = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        noobj_mask = torch.ones(batch_size, num_anchors, layer_h, layer_w)
        tx = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        ty = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tw = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        th = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tconf = torch.zeros(batch_size, num_anchors, layer_h, layer_w)
        tcls = torch.zeros(batch_size, num_anchors, layer_h, layer_w, num_classes)

        for b in torch.arange(batch_size):
            for t in torch.arange(target.size(1)):
                if target[b, t].sum() <= 0:
                    continue
                gx = target[b, t, 0] * layer_w
                gy = target[b, t, 1] * layer_h
                gw = target[b, t, 2] * layer_w
                gh = target[b, t, 3] * layer_h
                gi = int(gx)
                gj = int(gy)

                gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0) # [1, 4]
                anchors_box = torch.cat([torch.zeros((num_anchors, 2), dtype=torch.float32), torch.FloatTensor(scaled_anchors)], 1) # [num_anchors, 4]

                calc_iou = intersection_over_union(gt_box, anchors_box) # [num_anchors, 1]
                calc_iou = calc_iou.squeeze(dim=-1) # [num_anchors]
                
                noobj_mask[b, calc_iou > ignore_threshold, gj, gi] = 0
                best_n = torch.argmax(calc_iou)
                mask[b, best_n, gj, gi] = 1
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = gw/scaled_anchors[best_n][0]
                th[b, best_n, gj, gi] = gh/scaled_anchors[best_n][1]
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1
                
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls    


######################################################################################################################
# Model Module
######################################################################################################################
class YoloV2Detector(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = YoloV2Loss(cfg['num_classes'], cfg['scaled_anchors'])
        self.map_metric = MeanAveragePrecision(cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'])
        self.tmp_num = 0

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        pred = self.model(batch['img'])
        loss = self.loss_fn(pred, batch['annot'])

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_start(self):
        self.map_metric.reset_states()

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch['img'])
        loss = self.loss_fn(pred, batch['annot'])

        self.log('val_loss', loss, prog_bar=True, logger=True)

        self.map_metric.update_state(batch['annot'], pred)        

    def on_validation_epoch_end(self):
        map = self.map_metric.result()
        self.log('val_mAP', map, prog_bar=True, logger=True)

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        try:
            scheduler = get_scheduler(
                cfg['scheduler'],
                optim,
                **cfg['scheduler_options']
            )

            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        
        except KeyError:
            return optim


######################################################################################################################
# Train Function
######################################################################################################################
def train(cfg):
    data_module = YoloV2DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )

    backbone = get_model(cfg['backbone'])(num_classes=200)

    # Load pretrained weights
    ckpt_path = os.path.join(os.getcwd(), 'ckpt/darknet19-tiny-imagenet.ckpt')
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))

    state_dict = checkpoint["state_dict"]
    for key in list(state_dict):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)

    backbone.load_state_dict(state_dict)
    
    model = YoloV2(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )
    
    torchsummary.summary(model, (cfg['in_channels'], cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = YoloV2Detector(
        model=model, 
        cfg=cfg
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True,
            every_n_epochs=cfg['save_freq']
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=True
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], cfg['model'] + '_' + cfg['dataset_name'], default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=True) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )
    
    trainer.fit(model_module, data_module)


######################################################################################################################
# Test Function
######################################################################################################################
def test(cfg, ckpt):
    data_module = YoloV2DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=cfg['batch_size']
    )

    backbone = get_model(cfg['backbone'])()
    
    model = YoloV2(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )
    
    torchsummary.summary(model, (cfg['in_channels'], cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = YoloV2Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )

    trainer = pl.Trainer(
        logger=False,
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=True) if platform.system() != 'Windows' else None
    )
    
    trainer.validate(model_module, data_module)


######################################################################################################################
# Inference Function
######################################################################################################################
def inference(cfg, ckpt):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = YoloV2DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=1
    )
    data_module.prepare_data()
    data_module.setup()

    backbone = get_model(cfg['backbone'])()
    
    model = YoloV2(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )

    if torch.cuda.is_available:
        model = model.cuda()
    
    model_module = YoloV2Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )
    model_module.eval()

    yolov2_decoder = DecodeYoloV2(cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'], conf_threshold=0.5)

    # Inference
    for sample in data_module.val_dataloader():
        batch_x = sample['img']
        batch_y = sample['annot']

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()
        
        before = time.time()
        with torch.no_grad():
            predictions = model_module(batch_x)
        boxes = yolov2_decoder(predictions)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')
        
        # batch_x to img
        if torch.cuda.is_available:
            img = batch_x.cpu()[0].numpy()   
        else:
            img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # true_boxes = get_target_boxes(batch_y, 416)

        tagged_img = get_tagged_img(img, boxes, cfg['names'], (0, 255, 0))
        # tagged_img = get_tagged_img(tagged_img, true_boxes, cfg['names'], (0, 0, 255))

        cv2.imshow('test', tagged_img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()


######################################################################################################################
# Make Prediction Result Txt File Function for checking public map metric
######################################################################################################################
from tqdm import tqdm
def make_pred_result(cfg, ckpt):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = YoloV2DataModule(
        train_list=cfg['train_list'], 
        val_list=cfg['val_list'],
        workers=cfg['workers'], 
        input_size=cfg['input_size'],
        batch_size=1
    )
    data_module.prepare_data()
    data_module.setup()

    backbone = get_model(cfg['backbone'])()
    
    model = YoloV2(
        backbone=backbone,
        num_classes=cfg['num_classes'],
        num_anchors=len(cfg['scaled_anchors'])
    )

    if torch.cuda.is_available:
        model = model.cuda()
    
    model_module = YoloV2Detector.load_from_checkpoint(
        checkpoint_path=ckpt,
        model=model, 
        cfg=cfg
    )
    model_module.eval()

    yolov2_decoder = DecodeYoloV2(cfg['num_classes'], cfg['scaled_anchors'], cfg['input_size'], conf_threshold=0.25)

    with open(cfg['names'], 'r') as f:
        class_name_list = f.readlines()
    class_name_list = [x.strip() for x in class_name_list]

    img_num = 0
    # Inference
    for sample in tqdm(data_module.val_dataloader()):
        batch_x = sample['img']
        batch_y = sample['annot']

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()

        with torch.no_grad():
            predictions = model_module(batch_x)
        boxes = yolov2_decoder(predictions)

        img_num += 1
        pred_txt_fd = open(os.path.join('/home/fssv2/myungsang/my_projects/mAP/input/detection-results', f'{img_num:05d}.txt'), 'w')
        # true_txt_fd = open(os.path.join(os.getcwd(), 'tmp_gt', f'{img_num:05d}.txt'), 'w')

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

            pred_txt_fd.write(f'{class_name} {confidence_score} {xmin} {ymin} {xmax} {ymax}\n')
        pred_txt_fd.close()
        
        # true_boxes = get_target_boxes(batch_y, 416)

        # for bbox in true_boxes:
        #     class_name = class_name_list[int(bbox[-1])]
        #     confidence_score = bbox[4]
        #     cx = bbox[0]
        #     cy = bbox[1]
        #     w = bbox[2]
        #     h = bbox[3]
        #     xmin = int((cx - (w / 2)))
        #     ymin = int((cy - (h / 2)))
        #     xmax = int((cx + (w / 2)))
        #     ymax = int((cy + (h / 2)))

        #     true_txt_fd.write(f'{class_name} {xmin} {ymin} {xmax} {ymax}\n')
        # true_txt_fd.close()


######################################################################################################################
# Main
######################################################################################################################
if __name__ == '__main__':
    cfg = get_cfg()

    train(cfg)

    # ckpt = './saved/yolov2_voc/version_26/checkpoints/epoch=184-step=40699.ckpt'
    # test(cfg, ckpt)
    # inference(cfg, ckpt)
    # make_pred_result(cfg, ckpt)
