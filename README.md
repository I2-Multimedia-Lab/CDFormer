# CDFormer: When Degradation Prediction Embraces Diffusion Model for Blind Image Super-Resolution

Created by [Qingguo Liu](https://github.com/users/zbhfc712), [Chenyi Zhuang](https://github.com/Sheryl-Z), [Pan Gao]()\*, [Jie Qin]()\*

[[arXiv]]() [[supp]]()

This repository contains PyTorch implementation for __CDFormer: When Degradation Prediction Embraces Diffusion Model for Blind Image Super-Resolution__ (Accepted by CVPR 2024).

Existing Blind image Super-Resolution (BSR) methods focus on estimating either kernel or degradation information, but have long overlooked the essential content details. In this paper, we propose a novel BSR approach, Content-aware Degradation-driven Transformer (CDFormer), to capture both degradation and content representations. However, low-resolution images cannot provide enough content details, and thus we introduce a diffusion-based module $CDFormer_{diff}$ to first learn Content Degradation Prior (CDP) in both low- and high-resolution images, and then approximate the real distribution given only low-resolution information. Moreover, we apply an adaptive SR network $CDFormer_{SR}$ that effectively utilizes CDP to refine features. Compared to previous diffusion-based SR methods, we treat the diffusion model as an estimator that can overcome the limitations of expensive sampling time and excessive diversity. Experiments show that CDFormer can outperform existing methods, establishing a new SOTA performance on various benchmarks under blind settings.

## 🔥News
- **2024-02-27** CDFormer is accepted by CVPR 2024.

![intro](fig/network.pdf)

# Acknowledgements
This code is built on [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR) and [DiffIR](https://github.com/Zj-BinXia/DiffIR). We thank the authors for sharing the excellent codes.

## Citation
If you find our work useful in your research, please consider citing: 
```

```
