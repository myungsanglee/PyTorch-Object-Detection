# PyTorch Object Detection
PyTorch 기반 Object Detection 모델 구조 및 학습 기법을 테스트하기 위한 프로젝트

## Implementations
 * YOLO v1
 * YOLO v2
 * YOLO v3
 
## TODOs
- [x] ~~YOLO v1~~
- [x] ~~YOLO v2~~
- [x] ~~YOLO v3~~
- [ ] YOLO v4
- [ ] RetinaNet

## Requirements
* `PyTorch >= 1.8.1`
* `PyTorch Lightning`
* `Albumentations`
* `PyYaml`

## Train Detector
```python
python train_yolov2.py --cfg configs/yolov2-voc.yaml
```

## Test Detector
```python
python test_yolov2.py --cfg configs/yolov2-voc.yaml
```

## Inference Detector
```python
python inference_yolov2.py --cfg configs/yolov2-voc.yaml --ckpt path/to/ckpt_file
```

## YOLO Training Experiments
### 개요
- Darknet으로 훈련한 모델과 mAP가 비슷하게 나오게 하는 것이 목표
- Pretrained Weights 사용하지 않고 처음부터 훈련

### 데이터셋
| Train Dataset | Validation Dataset |
| --- | --- |
| VOC2007 Train + VOC2012 Train/Validation | VOC2007 Validation |
| 14041 images | 2510 images |

### 결과
- Validation Dataset 결과
- Confidence Threshold = 0.25

| Method | Backbone | Input size | DL Frameworks | mAP@.5 | Public mAP@.5 |
| --- | --- | --- | --- | --- | --- |
| Yolo v2 | darknet19 | 416 | PyTorch | 51.56% | 51.82% |
| Yolo v2 | darknet19 | 416 | Darknet | 51.18% | 51.29% |
| Yolo v3 | darknet19 | 416 | PyTorch | 56.02% | 56.34% |
| Yolo v3 | darknet19 | 416 | Darknet | 55.91% | 55.94% |

## References
- [Public mAP Calculator](https://github.com/Cartucho/mAP)
