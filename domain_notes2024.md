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

## MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation
- CVPR 2023, ETH Zurich & Max Planck Institute for Informatics
- [link](https://arxiv.org/pdf/2212.01322v2)
- [code](https://github.com/lhoyer/mic)
- learning spatial context relations of the target domain: enforces the
consistency between predictions of masked target images
- pseudo-labels that are generated based on the complete image by an exponential moving average teacher
- experiments on: GTA5 to Cityscapes, and VisDA17

## Hyperbolic Active Learning for Semantic Segmentation under Domain Shift
- ICML 2024, Sapienza University of Rome & UC Berkeley
- [link](https://arxiv.org/pdf/2306.11180v4)
- [code](https://github.com/paolomandica/HALO)
- novel interpretation of the hyperbolic radius as an indicator of data scarcity
- using either CNN or transformer backbones
- experiments on: GTA5/SYNTHIA to Cityscapes, Cityscapes to ACDC

## Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation
- NeurIPS 2022, University of Science and Technology of China
- [link](https://arxiv.org/pdf/2209.07695v3)
- [code](https://github.com/xiaoachen98/DDB)
- generating two intermediate domains using the coarse-wise and the fine-wise data mixing techniques
- cross-path knowledge distillation (multi-teacher distillation)
- benchmark with: CycleGAN, FDA, Mixup, CutMix, ClassMix, (source only or pseudo labeling methods)
- experiments on: GTA5 to Cityscapes, GTA5+Synscapes to Cityscapes, GTA5 to Cityscapes + Mapillary