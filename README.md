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

## YOLO v2 Training Experiment
### 개요
- Darknet을 벗어나기 위한 프로젝트
- YOLO V2를 처음부터 만들어서 훈련까지 PyTorch로 구현하는 것이 목표
- Darknet으로 훈련한 모델과 mAP가 비슷하게 나오게 하는 것이 목표
- Pretrained Weights 사용하지 않고 처음부터 훈련

### 데이터셋
| Train Dataset | Validation Dataset |
| --- | --- |
| VOC2007 Train + VOC2012 Train/Validation | VOC2007 Validation |
| 14041 images | 2510 images |

### 결과
|  | My mAP Calculator | Public mAP Calculator |
| --- | --- | --- |
| Verson 165 (Conf > 0.25) | <span style="color.red">52.94%</span> | 53.34% |
| Darknet (Conf > 0.25) | 51.18% | 51.29% |
| Verson 165 (Conf > 0.5) | 51.83% | 52.20% |
| Darknet (Conf > 0.5) | 44.64% | 44.70% |

> **Darknet 이기기 성공!!!**
> 
- Public mAP Calculator
    
    [GitHub - Cartucho/mAP: mean Average Precision - This code evaluates the performance of your neural net for object recognition.](https://github.com/Cartucho/mAP)
