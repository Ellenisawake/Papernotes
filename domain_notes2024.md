## Parameter Efficient Self-Supervised Geospatial Domain Adaptation
- CVPR 2024, 
- [link](

## Domain-Agnostic Mutual Prompting for Unsupervised Domain Adaptation
- CVPR 2024, 
- [link](https://github.com/TL-UESTC/DAMP)
- CLIP embedding

## Universal Semi-Supervised Domain Adaptation by Mitigating Common-Class Bias
- CVPR 2024, ASTAT & NUS
- [link](https://arxiv.org/pdf/2403.11234)
- prior-guided pseudo-label refinement strategy


## Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation
- CVPR 2024, University of Electronic Science and Technology of China
- [link](https://arxiv.org/pdf/2403.06946)
- modality separation network
- disentangles CLIP’s features into languageassociated and vision-associated components
- modality discriminator

## Continual-MAE: Adaptive Distribution Masked Autoencoders for Continual Test-Time Adaptation
- CVPR 2024, Peking University
- [link](https://sites.google.com/view/continual-mae/home)
- Distribution-aware Masking
- consistency constraints between the masked target samples and the original target samples

## Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models
- CVPR 2024, HKPolyU
- [link](https://arxiv.org/pdf/2403.17589)
- 

## Efficient Test-Time Adaptation of Vision-Language Models
- CVPR 2024, Mohamed bin Zayed University of Artificial Intelligence, Nanyang Technological University
- [link](https://arxiv.org/pdf/2403.18293)
- 

## Unsupervised Video Domain Adaptation with Masked Pre-Training and Collaborative Self-Training
- CVPR 2024, Johns Hopkins University
- [link](https://arxiv.org/pdf/2312.02914)
- teacher model, self-supervised learning,  pseudo-labeling, self-training

## Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation
- CVPR 2024, South China University of Technology
- [link](https://arxiv.org/pdf/2312.03502)
- 

## Test time adaptation via self-training with nearest neighbor information
- ICLR 2023, KAIST
- [link](https://openreview.net/pdf?id=EzLtB4M1SbM)
- 

## GeoMultiTaskNet: remote sensing unsupervised domain adaptation using geographical coordinates
- CVPR 2023 workshop, Sapienza University of Rome & Univ Gustave Eiffel
- [link](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Marsocci_GeoMultiTaskNet_Remote_Sensing_Unsupervised_Domain_Adaptation_Using_Geographical_Coordinates_CVPRW_2023_paper.pdf)
- 

## Gradual Domain Adaptation via Gradient Flow
- ICLR 2024, Southern University of Science and Technology
- [link](https://openreview.net/pdf?id=iTTZFKrlGV)
- Self-training, pseudo-labeling --> Gradual SelfTraining (GST) as the baseline algorithm for GDA
- Wasserstein Gradient Flow, internal, (external) potential, and interaction energy
- use the Wasserstein-2 distance to quantify the domain shift

## A Good Learner can Teach Better: Teacher-Student Collaborative Knowledge Distillation
- ICLR 2024, IIT Delhi
- [link](https://openreview.net/pdf?id=Ixi4j6LtdX)

## Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks
- ICLR 2024, CMU
- [link](https://arxiv.org/pdf/2309.17002)

## Spurious Feature Diversification Improves Out-of-distribution Generalization
- ICLR 2024, The Hong Kong University of Science and Technology
- [link](https://arxiv.org/pdf/2309.17230)
- 70+ pages

## Nevis'22: A Stream of 100 Tasks Sampled from 30 Years of Computer Vision Research
- ICLR 2024, DeepMind
- [link](https://arxiv.org/pdf/2211.11747)
- 70+ pages
- resulting stream reflects what the research community thought was meaningful at any point in time

## AUGCAL: Improving Sim2Real Adaptation by Uncertainty Calibration on Augmented Synthetic Images
- ICLR 2024, Georgia Tech
- [link](https://arxiv.org/pdf/2312.06106)
- replace training sim images with strongly augmented views, calibration loss
- GTAV --> Cityscapes

## Patch-Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective
- CVPR 2023, HKUST
- [link](https://arxiv.org/pdf/2303.13434v2)
- [pytorch 1.7 code (apex)](https://github.com/JinjingZhu/PMTrans)
- ViT-based PatchMix module to build up an intermediate domain, by learning to sample patches from both source and target domains based on the game-theoretical models
- loss: maximize cross entropy + feature/label space mixup loss
- use attention mao from ViT to re-weight label of each patch
- using pre-trained Swin-B model
- experiments on: Office-Home, Office-31, DomainNet and VisDA17

## MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation (MIC)
- CVPR 2023, ETH Zurich & Max Planck Institute for Informatics
- [link](https://arxiv.org/pdf/2212.01322)
- [code](https://github.com/lhoyer/mic)
- learning spatial context relations of the target domain: enforces the
consistency between predictions of masked target images
- pseudo-labels that are generated based on the complete image by an exponential moving average teacher
- experiments on: GTA5 to Cityscapes, and VisDA17
- benchmark with: ADVENT, ProDA, DAFormer, HRDA, DANNet
  - DAFormer: Improving network architectures and training strategies for domain-adaptive semantic segmentation, CVPR22
  - Advent: Adversarial entropy minimization for domain adaptation in semantic segmentation, CVPR19
- follow up work:
  - [HRDA](https://github.com/lhoyer/HRDA): Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation, ECCV2022
  - DAFormer/HRDA extension, TPAMI 2023
  - EDAPS, ICCV 2023

## DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation
- CVPR 2022, ETH Zurich
- [DAFormer paper](https://arxiv.org/pdf/2111.14887)
- [HRDA paper](https://arxiv.org/pdf/2304.13615)
- [code](https://github.com/lhoyer/DAFormer)
- Transformer encoder, multi-level context-aware feature fusion decoder
- Rare Class Sampling on the source domain
- Thing-Class ImageNet Feature Distance
- learning rate warmup
- experiments on GTA/Synthia --> Cityscapes

## Hyperbolic Active Learning for Semantic Segmentation under Domain Shift
- ICML 2024, Sapienza University of Rome & UC Berkeley
- [link](https://arxiv.org/pdf/2306.11180v4)
- [code](https://github.com/paolomandica/HALO)
- novel interpretation of the hyperbolic radius as an indicator of data scarcity
- using either CNN or transformer backbones
- experiments on: GTA5/SYNTHIA to Cityscapes, Cityscapes to ACDC
- benchmark with: source only, CBST, MADA, RIPU (active learning methods using portion of labelled target)

## Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation (DDB)
- NeurIPS 2022, University of Science and Technology of China
- [link](https://arxiv.org/pdf/2209.07695v3), [link2](https://proceedings.neurips.cc/paper_files/paper/2022/file/61aa557643ae8709b6a4f41140b2234a-Paper-Conference.pdf)
- [code](https://github.com/xiaoachen98/DDB)
- generating two intermediate domains using the coarse-wise and the fine-wise data mixing techniques
- cross-path knowledge distillation (multi-teacher distillation)
- benchmark with: CycleGAN, FDA, Mixup, CutMix, ClassMix, (source only or pseudo labeling methods)
  - Class-balanced pixel-level self-labeling for domain adaptive semantic segmentation, CVPR22
  - Undoing the damage of label shift for cross-domain semantic segmentation, CVPR22
  - Prototypical pseudo label denoising and target structure learning for domain adaptive semantic segmentation, CVPR21
- experiments on: GTA5 to Cityscapes, GTA5+Synscapes to Cityscapes, GTA5 to Cityscapes + Mapillary
