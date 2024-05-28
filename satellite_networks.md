

## Unsupervised Domain Adaptation for the Semantic Segmentation of Remote Sensing Images via One-Shot Image-to-Image Translation
- IEEE Geoscience and Remote Sensing Letters 2023
- [link](https://arxiv.org/abs/2212.03826)
- [code](https://github.com/Sarmadfismael/RSOS_I2I)
- image-to-image translation, encoder-decoder principle where latent content representations are mixed across domains
- a perceptual network module


## Unsupervised Domain Adaptation for Remote Sensing Semantic Segmentation with Transformer
- Remote Sensing 2022
- [link](https://www.mdpi.com/2072-4292/14/19/4942)
- [code](https://github.com/Levantespot/UDA_for_RS), based on DAFormer, MMSegmentation, SegFormer and DACS
- Gradual Class Weights (GCW) to stabilize the model on the source domain by addressing the class-imbalance problem
- Local Dynamic Quality (LDQ) to improve the quality of the pseudo-labels via distinguishing the discrete and clustered pseudo-labels on the target domain
- Potsdam --> Vaihingen, Vaihingen --> Potsdam

## Unsupervised domain adaptation semantic segmentation of high-resolution remote sensing imagery with invariant domain-level Prototype memory
- [link](https://arxiv.org/pdf/2208.07722)
- output space adversarial learning
- invariant feature memory module to store invariant domain-level prototype information
- category attention-driven invariant domain-level memory aggregation module
- entropy-based pseudo label filtering


## Remote sensing image domain adaptation segmentation model combining scale discriminator and attention
- International Conference on Geology, Mapping and Remote Sensing (ICGMRS 2023), SPIE
- Shaanxi Normal University
- [link](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12978/129781H/Remote-sensing-image-domain-adaptation-segmentation-model-combining-scale-discriminator/10.1117/12.3019644.short#_=_)
- common feature discriminator, adversarial-based discrimination network
- proposed: a scale discriminator for different remote sensing image resolutions in different domains
- visual attention network (VAN) for spatial and channel attention
- FocalLoss

## GeoMultiTaskNet: remote sensing unsupervised domain adaptation using geographical coordinates
- CVPR 2023 workshop, Sapienza University of Rome & Univ Gustave Eiffel
- [link](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Marsocci_GeoMultiTaskNet_Remote_Sensing_Unsupervised_Domain_Adaptation_Using_Geographical_Coordinates_CVPRW_2023_paper.pdf)
- 

## Modernized Training of U-Net for Aerial Semantic Segmentation
- WACV2024 workshop
- [link to paper](https://openaccess.thecvf.com/content/WACV2024W/CV4EO/papers/Straka_Modernized_Training_of_U-Net_for_Aerial_Semantic_Segmentation_WACVW_2024_paper.pdf)
- French National Institute of Geographical and Forest Information (IGN)
- [link to repo](https://github.com/strakaj/U-Net-for-remote-sensing)
- French Land cover from Aerospace ImageRy (FLAIR)


## An Empirical Study of Remote Sensing Pretraining
- IEEE
- [link to paper](https://ieeexplore.ieee.org/document/9782149)
- MillionAID dataset, 1000848 nonoverlapping scenes, RGB, agriculture, land, commercial land, industrial land, public service land, residential land, transportation land, unutilized land, and water area
- fMoW 132716,  BigEarthNet 590326 scenes
- ResNet-50, ViTAEv2 model, DeiT-S, PVT-S, and Swin-T, ViT-B
