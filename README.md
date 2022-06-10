# PyTorch Object Detection
PyTorch 기반 Object Detection 모델 구조 및 학습 기법을 테스트하기 위한 프로젝트

## Implementations
 * YOLO v1
 * YOLO v2

## TODOs
- [x] ~~YOLO v1~~
- [x] ~~YOLO v2~~
- [ ] YOLO v3
- [ ] YOLO v4
- [ ] RetinaNet

## Requirements
* `PyTorch >= 1.8.1`
* `PyTorch Lightning`
* `Albumentations`
* `PyYaml`

## Train Detector
```python
python train_yolov1.py --cfg configs/yolov1-voc.yaml
```

## Test Detector
```python
python test_yolov1.py --cfg configs/yolov1-voc.yaml
```

## YOLO v2 Training Experiment
### 개요
- Darknet을 벗어나기 위한 프로젝트
- YOLO V2를 처음부터 만들어서 훈련까지 PyTorch로 구현하는 것이 목표
- Darknet으로 훈련한 모델과 mAP가 비슷하게 나오게 하는 것이 목표
### 훈련
- **Version 0**
    - **개요**
        - 실험 기준이 되는 모델
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
    - **기본 세팅**
        - Hyperparameter
            
            ```python
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
                    'lr': 1e-3,
                    'momentum': 0.9,
                    'weight_decay': 1e-5
                }
            
                cfg['scheduler'] = 'yolo_lr'
                cfg['scheduler_options'] = {
                    'burn_in': 1000,
                    'steps': [40000, 60000],
                    'scales': [0.1, 0.1]
                }
            
                return cfg
            ```
            
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1,
                    always_apply=True
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            
            valid_transform = A.Compose([
                A.Resize(self.input_size, self.input_size, always_apply=True),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - Loss Function
            
            ```python
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
                    self.lambda_obj = 1
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
                    w = torch.sqrt(torch.exp(prediction[..., 2]))
                    h = torch.sqrt(torch.exp(prediction[..., 3]))
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
                    box_loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h) * (1/batch_size)
            
                    # ==================== #
                    #   FOR OBJECT LOSS    #
                    # ==================== #
                    object_loss = self.lambda_obj * self.bce_loss(conf * mask, tconf) * (1/batch_size)
            
                    # ======================= #
                    #   FOR NO OBJECT LOSS    #
                    # ======================= #
                    no_object_loss = self.lambda_noobj * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0) * (1/batch_size)
            
                    # ================== #
                    #   FOR CLASS LOSS   #
                    # ================== #
                    class_loss = self.lambda_class * self.bce_loss(pred_cls[mask==1], tcls[mask==1]) * (1/batch_size)
            
                    loss = box_loss + object_loss + no_object_loss + class_loss
            
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
                            tw[b, best_n, gj, gi] = torch.sqrt(gw/scaled_anchors[best_n][0])
                            th[b, best_n, gj, gi] = torch.sqrt(gh/scaled_anchors[best_n][1])
                            tconf[b, best_n, gj, gi] = calc_iou[best_n]
                            tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1
                            
                    return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
            ```
            
    - **결과**
        - val_loss가 epoch=19에서 증가로 Overfitting 발생
        - mAP@0.5 =  12.79%
- **Version 1**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
    - **결과**
        - val_loss가 epoch=34에서 증가로 Overfitting 발생
        - mAP@0.5 =  17.00%
        - Overfitting이 조금 늦게 발생하긴 했지만 아직 너무 빨리 발생- **Version 2**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
    - **결과**
        - val_loss가 epoch=34에서 증가로 Overfitting 발생
        - mAP@0.5 =  18.59%
        
- **Version 3**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Learning Rate 1e-3 → 5e-4 변경
    - **결과**
        - val_loss가 epoch=39에서 증가로 Overfitting 발생
        - mAP@0.5 =  15.10%
- **Version 4**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 1
            self.lambda_coord = 1
            self.lambda_class = 1
            ```
            
    - **결과**
        - val_loss가 epoch=39에서 증가로 Overfitting 발생
        - mAP@0.5 =  22.88%
- **Version 5**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 1
            self.lambda_class = 1
            ```
            
    - **결과**
        - val_loss가 epoch=39에서 증가로 Overfitting 발생
        - mAP@0.5 =  25.69%
- **Version 6**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            ```
            
    - **결과**
        - val_loss가 epoch=34에서 증가로 Overfitting 발생
        - mAP@0.5 =  28.74%
- **Version 7**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            class_loss에 Focal Loss 적용
            ```
            
    - **결과**
        - val_loss가 epoch=34에서 증가로 Overfitting 발생
        - mAP@0.5 =  15.32%
- **Version 8**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            class_loss에 Cross Entropy Loss 적용
            ```
            
    - **결과**
        - val_loss가 epoch=34에서 증가로 Overfitting 발생
        - mAP@0.5 =  24.72%
- **Version 9**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
    - **결과**
        - val_loss가 epoch=54에서 증가로 Overfitting 발생
        - mAP@0.5 =  32.11%
- **Version 10**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
    - **결과**
        - val_loss가 epoch=54에서 증가로 Overfitting 발생
        - mAP@0.5 =  32.61%
- **Version 11**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            w, h에 torch.sqrt 제거
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
    - **결과**
        - val_loss가 epoch=89에서 증가로 Overfitting 발생
        - mAP@0.5 =  34.95%
- **Version 12**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            w, h에 torch.sqrt 제거
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
        - Optimizer SGD → AdamW 변경
    - **결과**
        - val_loss가 epoch=94에서 증가로 Overfitting 발생
        - mAP@0.5 =  33.45%
- **Version 13**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            w, h에 torch.sqrt 제거
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
        - Scheduler 변경
            - yolo_lr → cosine_annealing_warm_up_restarts 변경
                
                ```python
                cfg['scheduler'] = 'cosine_annealing_warm_up_restarts'
                cfg['scheduler_options'] = {
                    'T_0': 5000,
                    'T_mult': 2,
                    'eta_max': 1e-3,
                    'T_up': 1000,
                    'gamma': 0.96
                }
                ```
                
    - **결과**
        - val_loss가 epoch=59에서 증가로 Overfitting 발생
        - mAP@0.5 =  24.29%
- **Version 14**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            w, h에 torch.sqrt 제거
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
        - Scheduler ‘burn_in’ 1000 → 2000 변경
    - **결과**
        - val_loss가 epoch=104에서 증가로 Overfitting 발생
        - mAP@0.5 =  25.09%
- **Version 15**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            w, h에 torch.sqrt 제거
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
        - Scheduler ‘burn_in’ 1000 → 500 변경
    - **결과**
        - val_loss가 epoch=164에서 증가로 Overfitting 발생
        - mAP@0.5 =  33.03%
- **Version 16**
    - **개요**
        - Pretrained된 Darknet19 사용하지 않고 처음부터 훈련
        - Overfitting 예방하기 위해 Image Augmentation 증가
    - **변경 사항**
        - Image Augmentation
            
            ```python
            train_transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Cutout(),
                A.Blur(),
                A.CLAHE(),
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.2,
                    saturation=0.5,
                    hue=0.1
                ),
                A.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
                A.Normalize(0, 1),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))
            ```
            
        - weight_initialize 미사용
        - Loss Function 변경
            
            ```python
            self.lambda_obj = 5
            self.lambda_noobj = 0.5
            self.lambda_coord = 5
            self.lambda_class = 1
            
            w, h에 torch.sqrt 제거
            ```
            
        - Yolo Head layer 전에 Dropout(0.5) Layer 추가
        - Yolo 모델 Conv2d → ConvBnRelu(bias=False) 변경, 마지막 Conv2d(bias=False) 변경
        - Learning Rate 1e-3 → 5e-3 변경
    - **결과**
        - val_loss가 epoch=79에서 증가로 Overfitting 발생
        - mAP@0.5 =  33.59%
