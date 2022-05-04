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
