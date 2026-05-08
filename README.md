# Towards Generalized Spaceborne SAR Ship Detection via Fourier-based Perturbation Augmentation and Quality-Aware Invariance Learning
Official PyTorch implementation accompanying our submitted manuscript to *ISPRS Journal of Photogrammetry and Remote Sensing*.
## Abstract
Synthetic aperture radar (SAR) ship detection is essential for maritime situational awareness. As spaceborne SAR platforms continue to expand, there is an urgent demand for detectors that can generalize to new or unseen platforms. However, existing methods are typically tailored to specific platforms and suffer significant performance degradation on unseen data due to inherent distribution shifts and limited generalization ability. To alleviate this, we propose a generalized SAR ship detection framework through Fourier-based perturbation Augmentation and Quality-Aware invariance learning, termed FAQA. FAQA aims to integrate multiple platform SAR data to learn generalized ship representations, boosting detection on previously unseen data. Specifically, our framework consists of three key components:

- **FPA (Fourier-based Perturbation Augmentation)** : Simulates potential unknown domains by applying controlled random perturbations to the low-frequency amplitude regions, while preserving semantic integrity.

- **QAIL (Quality-Aware Invariance Learning)**: Jointly evaluates the location-classification quality of each proposal as guidance, prompting the model to focus on invariance learning of semantically reliable ship regions while suppressing SAR-specific background interference.

- **APS (Adaptive Parameter Search)**: Leverages gradient alignment between original and augmented images as the supervisory signal, adaptively steering optimization toward more generalized solutions.
---
## News
- [2026.05] Code repository released.
- [2025.11] Manuscript submitted to ISPRS.
---
## Environment Setup

This project is implemented based on:

- [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

- [HTCN](https://github.com/chaoqichen/HTCN)

Please follow the environment setup instructions in the Faster R-CNN and HTCN repository.

### Requirements
- Python 3.8+
- PyTorch 1.8.0
- CUDA 12.4
- torchvision
- numpy
- scipy
- opencv-python
### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
Directory structure:
```
data/
├── VOCdevkit/
│   ├── VOC2007/
│   │   ├── Annotations/         # XML annotation files
│   │   ├── ImageSets/
│   │   │   └── Main/
│   │   │       ├── train.txt
│   │   │       ├── val.txt
│   │   │       └── test.txt
│   │   ├── JPEGImages/          # SAR images
│   │ 
│   └── ...
```
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.
## Models
### Pre-trained Models
In our experiments, we used two pretrained_models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)
* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)
Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.
## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python trainval_net.py \
       --dataset source_dataset\
       --net vgg16/resnet101
```
## Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python evaluate_net.py \
       --dataset source_dataset --dataset_t target_dataset \
       --net vgg16/resnet101  \
       --load_name path_to_model
```










