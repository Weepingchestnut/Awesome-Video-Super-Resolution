# This repo is built for paper: Video Super Resolution Based on Deep Learning: A Comprehensive Survey【[paper](https://arxiv.org/abs/2007.12928)】

![image](./imgs/classification.png)



**Citing this work**

If this repository is helpful to you, please cite our [survey](https://arxiv.org/abs/2007.12928).

```
@article{liu2020video,
  title={Video super resolution based on deep learning: A comprehensive survey},
  author={Liu, Hongying and Ruan, Zhubo and Zhao, Peng and Dong, Chao and Shang, Fanhua and Liu, Yuanyuan and Yang, Linlin},
  journal={arXiv preprint arXiv:2007.12928},
  year={2020}
}
```

🔥 (citations > 200)  

## Methods with aligment

### MEMC

| Paper                                                        | Model         | Code                                                         | Published                                                    |
| ------------------------------------------------------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Video Super-Resolution via Deep Draft-Ensemble Learning 🔥    | Deep-DE       | /                                                            | [ICCV2015](https://openaccess.thecvf.com/content_iccv_2015/html/Liao_Video_Super-Resolution_via_ICCV_2015_paper.html) |
| Video Super-Resolution With Convolutional Neural Networks 🔥  | VSRnet        | [PyTorch](https://github.com/usstdqq/vsrnet_pytorch)         | [TCI2016](https://ieeexplore.ieee.org/document/7444187)      |
| Real-Time Video Super-Resolution With Spatio-Temporal Networks and Motion Compensation 🔥 | VESPCN        | [PyTorch](https://github.com/JuheonYi/VESPCN-PyTorch), [TensorFlow](https://github.com/JuheonYi/VESPCN-tensorflow) | [CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.html), [arXiv](http://arxiv.org/abs/1611.05250v2) |
| Detail-Revealing Deep Video Super-Resolution 🔥               | DRVSR         | [TensorFlow](https://github.com/JuheonYi/VESPCN-tensorflow)  | [ICCV2017](https://openaccess.thecvf.com/content_iccv_2017/html/Tao_Detail-Revealing_Deep_Video_ICCV_2017_paper.html), [arXiv](https://arxiv.org/abs/1704.02738v1) |
| Robust Video Super-Resolution with Learned Temporal Dynamics | RVSR          | /                                                            | [ICCV2017](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Robust_Video_Super-Resolution_ICCV_2017_paper.html), [arXiv]() |
| Frame-Recurrent Video Super-Resolution 🔥                     | FRVSR         | [GitHub](https://github.com/msmsajjadi/FRVSR)                | [CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sajjadi_Frame-Recurrent_Video_Super-Resolution_CVPR_2018_paper.html), [arXiv](https://arxiv.org/abs/1801.04590) |
| Spatio-Temporal Transformer Network for Video Restoration    | STTN          | [PyTorch](https://github.com/alpErenSari/spatioTemporalTransformer) | [ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Tae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper.html), [arXiv]() |
| Learning for Video Super-Resolution Through HR Optical Flow Estimation (ACCV), Deep Video Super-Resolution using HR Optical Flow Estimation (TIP) | SOFVSR        | [PyTorch](https://github.com/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR) | [ACCV2018](https://link.springer.com/chapter/10.1007/978-3-030-20887-5_32), [TIP2020](http://arxiv.org/abs/2001.02129) |
| Video Enhancement with Task-Oriented Flow 🔥                  | TOFlow        | [PyTorch](https://github.com/anchen1011/toflow)              | [IJCV2019](https://link.springer.com/article/10.1007/s11263-018-01144-2), [arXiv](https://arxiv.org/abs/1711.09078) |
| Multi-Memory Convolutional Neural Network for Video Super-Resolution | MMCNN         | [TensorFlow](https://github.com/psychopa4/MMCNN)             | [TIP2019](https://ieeexplore.ieee.org/document/8579237)      |
| MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement | MEMC-Net      | [PyTorch](https://github.com/baowenbo/MEMC-Net)              | [TPAMI2021](https://ieeexplore.ieee.org/abstract/document/8840983/), [arXiv](https://arxiv.org/abs/1810.08768) |
| Video Super-Resolution Using Non-Simultaneous Fully Recurrent Convolutional Network | RRCN          | /                                                            | [TIP2019](https://ieeexplore.ieee.org/abstract/document/8501928/) |
| Real-time video super-resolution via motion convolution kernel estimation | RTVSR         | /                                                            | [NEUCOM](https://www.sciencedirect.com/science/article/abs/pii/S0925231219311063) |
| Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation | TecoGAN       | [TensorFlow](https://github.com/thunil/TecoGAN), [PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch) | [TG2020](https://dl.acm.org/doi/abs/10.1145/3386569.3392457), [arXiv](https://arxiv.org/abs/1811.09393) |
| MultiBoot VSR: Multi-Stage Multi-Reference Bootstrapping for Video Super-Resolution | MultiBoot VSR | /                                                            | [CVPRW2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Kalarot_MultiBoot_Vsr_Multi-Stage_Multi-Reference_Bootstrapping_for_Video_Super-Resolution_CVPRW_2019_paper.html) |
| MuCAN: Multi-Correspondence Aggregation Network for Video Super-Resolution | MuCAN         | [PyTorch](https://github.com/dvlab-research/Simple-SR)       | [ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550341.pdf), [arXiv](https://arxiv.org/pdf/2007.11803) |
| BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond | BasicVSR      | [PyTorch](https://github.com/open-mmlab/mmediting)           | [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.html), [arXiv](https://arxiv.org/abs/2012.02181) |



### DC



## Method without aligment

### 2D Conv



### 3D Conv



### RCNN



### Non-Local



### Other





