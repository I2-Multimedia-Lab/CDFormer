# CDFormer: When Degradation Prediction Embraces Diffusion Model for Blind Image Super-Resolution

Created by [Qingguo Liu](https://github.com/users/zbhfc712), [Chenyi Zhuang](https://github.com/Sheryl-Z), [Pan Gao]()\*, [Jie Qin]()\*

[[arXiv]]() [[supp]]()

This repository contains PyTorch implementation for __CDFormer: When Degradation Prediction Embraces Diffusion Model for Blind Image Super-Resolution__ (Accepted by CVPR 2024).
## 🔥Abstract
> Existing Blind image Super-Resolution (BSR) methods focus on estimating either kernel or degradation information, but have long overlooked the essential content details. In this paper, we propose a novel BSR approach, Content-aware Degradation-driven Transformer (CDFormer), to capture both degradation and content representations. However, low-resolution images cannot provide enough content details, and thus we introduce a diffusion-based module $CDFormer_{diff}$ to first learn Content Degradation Prior (CDP) in both low- and high-resolution images, and then approximate the real distribution given only low-resolution information. Moreover, we apply an adaptive SR network $CDFormer_{SR}$ that effectively utilizes CDP to refine features. Compared to previous diffusion-based SR methods, we treat the diffusion model as an estimator that can overcome the limitations of expensive sampling time and excessive diversity. Experiments show that CDFormer can outperform existing methods, establishing a new SOTA performance on various benchmarks under blind settings.
## 🔥News
- **2024-02-27** CDFormer is accepted by CVPR 2024.

## 🔥Results
![intro](fig/iamge_040.gif)

## 🔥Environment
Python 3.8.8 and Pytorch 2.0.1. Details can be found in `requirements.txt`. 

## 🔥Train
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `your_data_path/DF2K/HR` to build the DF2K dataset. 

### 2. Begin to train
Run `python main.py` to train on the DF2K dataset. Please update `dir_data` in the bash file as `your_data_path`.

## Test
### 1. Prepare test data 
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `your_data_path/benchmark`.
### 2. Prepare pretrained model 
Download [pretrained model](https://drive.google.com/drive/folders/1zWAPqE23VBBy7bpTyM7omTERrn6bXq0x?usp=sharing)  in `your_data_path`(e.g., for `x2` scale, download `experiment\cdformer_x2_bicubic_iso\model\model_1200pt` in `your_data_path\experiment\cdformer_x2_bicubic_iso\model\model_1200pt`).

### 3. Begin to test
Run `python test_x2.py` to test scale 2 on benchmark datasets. 
Run `python test_x3.py` to test scale 3 on benchmark datasets.
Run `python test_x4.py` to test scale 4 on benchmark datasets.
Please update `dir_data` in the bash file as `your_data_path` and selection parameter.
# Acknowledgements
This code is built on [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR), [DAT](https://github.com/zhengchen1999/DAT) and [DiffIR](https://github.com/Zj-BinXia/DiffIR). We thank the authors for sharing the excellent codes.

## Citation
If you find our work useful in your research, please consider citing: 
```

```
