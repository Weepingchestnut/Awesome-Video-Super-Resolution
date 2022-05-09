# This repo is built for paper: Video Super Resolution Based on Deep Learning: A Comprehensive Survey【[paper](https://arxiv.org/abs/2007.12928)】

![image](./imgs/classification.png)



**Citing this work**

If this repository is helpful to you, please cite our [survey](https://arxiv.org/abs/2007.12928).

```
@article{liu2022video,
  title={Video super-resolution based on deep learning: a comprehensive survey},
  author={Liu, Hongying and Ruan, Zhubo and Zhao, Peng and Dong, Chao and Shang, Fanhua and Liu, Yuanyuan and Yang, Linlin and Timofte, Radu},
  journal={Artificial Intelligence Review},
  pages={1--55},
  year={2022},
  publisher={Springer}
}

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
| Video Super-Resolution via Deep Draft-Ensemble Learning [[Project Page]](http://www.cse.cuhk.edu.hk/leojia/projects/DeepSR/index.html) 🔥 | Deep-DE       | [MATLAB](http://www.cse.cuhk.edu.hk/leojia/projects/DeepSR/data/DeepSR_code.zip) | [ICCV2015](https://openaccess.thecvf.com/content_iccv_2015/html/Liao_Video_Super-Resolution_via_ICCV_2015_paper.html) |
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
| Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation [[Project Page]](https://ge.in.tum.de/publications/2019-tecogan-chu/) | TecoGAN       | [TensorFlow](https://github.com/thunil/TecoGAN), [PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch) | [TG2020](https://dl.acm.org/doi/abs/10.1145/3386569.3392457), [arXiv](https://arxiv.org/abs/1811.09393) |
| MultiBoot VSR: Multi-Stage Multi-Reference Bootstrapping for Video Super-Resolution | MultiBoot VSR | /                                                            | [CVPRW2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Kalarot_MultiBoot_Vsr_Multi-Stage_Multi-Reference_Bootstrapping_for_Video_Super-Resolution_CVPRW_2019_paper.html) |
| MuCAN: Multi-Correspondence Aggregation Network for Video Super-Resolution | MuCAN         | [PyTorch](https://github.com/dvlab-research/Simple-SR)       | [ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550341.pdf), [arXiv](https://arxiv.org/pdf/2007.11803) |
| BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond [[Project Page]](https://ckkelvinchan.github.io/projects/BasicVSR/) | BasicVSR      | [PyTorch](https://github.com/open-mmlab/mmediting)           | [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chan_BasicVSR_The_Search_for_Essential_Components_in_Video_Super-Resolution_and_CVPR_2021_paper.html), [arXiv](https://arxiv.org/abs/2012.02181) |



### DC

| Paper                                                        | Model    | Code                                                        | Published                                                    |
| ------------------------------------------------------------ | -------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| EDVR: Video Restoration With Enhanced Deformable Convolutional Networks [[Project Page]](https://xinntao.github.io/projects/EDVR) 🔥 | EDVR     | [PyTorch](https://github.com/xinntao/EDVR)                  | [CVPR2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.html), [arXiv](http://arxiv.org/abs/1905.02716) |
| Deformable Non-Local Network for Video Super-Resolution      | DNLN     | [PyTorch](https://github.com/wh1h/DNLN)                     | [ACCESS2019](https://ieeexplore.ieee.org/abstract/document/8926405) |
| TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution | TDAN     | [PyTorch](https://github.com/YapengTian/TDAN-VSR-CVPR-2020) | [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Tian_TDAN_Temporally-Deformable_Alignment_Network_for_Video_Super-Resolution_CVPR_2020_paper.html), [arXiv](https://arxiv.org/abs/1812.02898) |
| Deformable 3D Convolution for Video Super-Resolution         | D3Dnet   | [PyTorch](https://github.com/XinyiYing/D3Dnet)              | [SPL2020](https://ieeexplore.ieee.org/abstract/document/9153920), [arXiv](https://arxiv.org/abs/2004.02803) |
| VESR-Net: The Winning Solution to Youku Video Enhancement and Super-Resolution Challenge | VESR-Net | /                                                           | [arXiv](https://arxiv.org/abs/2003.02115)                    |



## Method without aligment

### 2D Conv

| Paper                                                        | Model         | Code                                               | Published                                                    |
| ------------------------------------------------------------ | ------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Generative Adversarial Networks and Perceptual Losses for Video Super-Resolution | VSRResFeatGAN | /                                                  | [TIP2019](https://ieeexplore.ieee.org/document/8629024), [arXiv](https://arxiv.org/abs/1806.05764) |
| Frame and Feature-Context Video Super-Resolution             | FFCVSR        | [TensorFlow](https://github.com/linchuming/FFCVSR) | [AAAI2019](https://aaai.org/ojs/index.php/AAAI/article/view/4502) |

### 3D Conv

| Paper                                                        | Model   | Code                                            | Published                                                    |
| ------------------------------------------------------------ | ------- | ----------------------------------------------- | ------------------------------------------------------------ |
| Deep Video Super-Resolution Network Using Dynamic Upsampling FiltersWithout Explicit Motion Compensation 🔥 | DUF     | [TensorFlow](https://github.com/yhjo09/VSR-DUF) | [CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.html) |
| Fast Spatio-Temporal Residual Network for Video Super-Resolution | FSTRN   | [TensorFlow](https://github.com/lsmale/FSTRN)   | [CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Fast_Spatio-Temporal_Residual_Network_for_Video_Super-Resolution_CVPR_2019_paper.html), [arXiv](https://arxiv.org/abs/1904.02870) |
| 3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks | 3DSRnet | [MATLAB](https://github.com/sooyekim/3DSRnet)   | ICIP2019, [arXiv](https://arxiv.org/abs/1812.09079)          |
| Large Motion Video Super-Resolution with Dual Subnet and Multi-Stage Communicated Upsampling | DSMC    | PyTorch                                         | [AAAI2021](https://ojs.aaai.org/index.php/AAAI/article/view/16310/16117), [arXiv](https://arxiv.org/abs/2103.11744) |

### RCNN

| Paper                                                        | Model | Code                                           | Published                                                    |
| ------------------------------------------------------------ | ----- | ---------------------------------------------- | ------------------------------------------------------------ |
| Bidirectional Recurrent Convolutional Networks for Multi-Frame Super-Resolution(NeurIPS)<br />Video Super-Resolution via Bidirectional Recurrent Convolutional Networks (TPAMI) | BRCN  | [MATLAB](https://github.com/linan142857/BRCN)  | [NeurIPS2015](http://cognn.com/papers/24%20NIPS%202015%20Yan%20bidirectional-recurrent-convolutional-networks-for-multi-frame-super-resolution-Paper.pdf), [TPAMI2017](https://ieeexplore.ieee.org/abstract/document/7919264) |
| Building an End-to-End Spatial-Temporal Convolutional Network for Video Super-Resolution | STCN  | /                                              | [AAAI2017](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14733) |
| Residual Invertible Spatio-Temporal Network for Video Super-Resolution | RISTN | [PyTorch](https://github.com/lizhuangzi/RISTN) | [AAAI2019](https://ojs.aaai.org/index.php/AAAI/article/view/4550) |
| Efficient Video Super-Resolution through Recurrent Latent Space Propagation | RLSP  | [PyTorch](https://github.com/dariofuoli/RLSP)  | [ICCVW2019](https://ieeexplore.ieee.org/abstract/document/9022159), [arXiv](https://arxiv.org/abs/1909.08080) |
| Video Super-Resolution with Recurrent Structure-Detail Network | RSDN  | [PyTorch](https://github.com/junpan19/RSDN)    | [ECCV2020](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_38), [arXiv](https://arxiv.org/abs/2008.00455) |

### Non-Local

| Paper                                                        | Model | Code                                            | Published                                                    |
| ------------------------------------------------------------ | ----- | ----------------------------------------------- | ------------------------------------------------------------ |
| Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations | PFNL  | [TensorFlow](https://github.com/psychopa4/PFNL) | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Yi_Progressive_Fusion_Video_Super-Resolution_Network_via_Exploiting_Non-Local_Spatio-Temporal_Correlations_ICCV_2019_paper.html) |

### Other

| Paper                                                        | Model   | Code                                                 | Published                                                    |
| ------------------------------------------------------------ | ------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Recurrent Back-Projection Network for Video Super-Resolution [[Project Page]](Project page) | RBPN    | [PyTorch](https://github.com/alterzero/RBPN-PyTorch) | [CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Haris_Recurrent_Back-Projection_Network_for_Video_Super-Resolution_CVPR_2019_paper.html), [arXiv](https://arxiv.org/abs/1903.10128) |
| Space-Time-Aware Multi-Resolution Video Enhancement          | STARnet | [PyTorch](https://github.com/alterzero/STARnet)      | [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Haris_Space-Time-Aware_Multi-Resolution_Video_Enhancement_CVPR_2020_paper.html), [arXiv](https://arxiv.org/abs/2003.13170) |
| Video super-resolution via dense non-local spatial-temporal convolutional network | DNSTNet | /                                                    | [NEUCOM2020](https://www.sciencedirect.com/science/article/abs/pii/S0925231220306056) |



## New methods

Some new methods that were not categorized.

| Paper                                                        | Model   | Code                                                      | Published                                                    |
| ------------------------------------------------------------ | ------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| Omniscient Video Super-Resolution                            | OVSR    | [PyTorch](https://github.com/psychopa4/OVSR)              | [ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yi_Omniscient_Video_Super-Resolution_ICCV_2021_paper.html) |
| Learning interlaced sparse Sinkhorn matching network for video super-resolution | ISSM    | /                                                         | [PATCOG2021](https://www.sciencedirect.com/science/article/pii/S0031320321006518) |
| MEGAN: Memory Enhanced Graph Attention Network for Space-Time Video Super-Resolution | MEGAN   | /                                                         | [WACV2022](https://openaccess.thecvf.com/content/WACV2022/html/You_MEGAN_Memory_Enhanced_Graph_Attention_Network_for_Space-Time_Video_Super-Resolution_WACV_2022_paper.html) |
| Plug-and-Play video super-resolution using edge-preserving filtering | /       | /                                                         | [CVIU2022](https://www.sciencedirect.com/science/article/pii/S1077314222000029) |
| Video super-resolution using a hierarchical recurrent multireceptive-field integration network | RMRIN   | /                                                         | [DSP2021](https://www.sciencedirect.com/science/article/pii/S1051200421003912) |
| Improved EDVR Model for Robust and Efficient Video Super-Resolution | /       | /                                                         | [WACV2022](https://openaccess.thecvf.com/content/WACV2022W/VAQ/html/Huang_Improved_EDVR_Model_for_Robust_and_Efficient_Video_Super-Resolution_WACVW_2022_paper.html) |
| Video super-resolution network using detail component extraction and optical flow enhancement algorithm | /       | /                                                         | [Appl Intell2022](https://link.springer.com/article/10.1007/s10489-021-02882-6) |
| Deeply feature fused video super-resolution network using temporal grouping | /       | /                                                         | [Supercomput2022](https://link.springer.com/article/10.1007/s11227-021-04299-x) |
| Frame Attention Recurrent Back-Projection Network for Accurate Video Super-Resolution | /       | /                                                         | [ICCE2022](https://ieeexplore.ieee.org/abstract/document/9730760) |
| Semi-Supervised Super-Resolution                             | /       | /                                                         | [arXiv](https://arxiv.org/abs/2204.08192)                    |
| STDAN: Deformable Attention Network for Space-Time Video Super-Resolution | STDAN   | /                                                         | [arXiv](https://arxiv.org/abs/2203.06841)                    |
| Fast Online Video Super-Resolution with Deformable Attention Pyramid | DAP-128 | /                                                         | [arXiv](https://arxiv.org/abs/2202.01731)                    |
| Self-Supervised Deep Blind Video Super-Resolution            | /       | [PyTorch](https://github.com/csbhr/Deep-Blind-VSR)        | [arXiv](https://arxiv.org/abs/2201.07422)                    |
| Video Super-Resolution Transformer                           | VSRT    | [PyTorch](https://github.com/caojiezhang/VSR-Transformer) | [arXiv](https://arxiv.org/abs/2106.06847)                    |
| VRT: A Video Restoration Transformer                         | VRT     | [PyTorch](https://github.com/JingyunLiang/VRT)            | [arXiv](https://arxiv.org/abs/2201.12288)                    |

