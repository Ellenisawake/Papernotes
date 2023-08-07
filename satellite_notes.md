# Papers

## Automatically Mapping Urban Green Space Using Sentinel-2 Imagery and Deep Learning Methods in Multiple Cities Worldwide: A Convolutional Neural Network Approach
- [Master thesis](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/42573/Automatically%20Mapping%20Urban%20Green%20Space%20Using%20Sentinel-2%20Imagery%20and%20Deep%20Learning%20Methods%20in%20Multiple%20Cities%20Worldwide%20A%20Convolutional%20Neural%20Network%20Approach.pdf?sequence=1&isAllowed=y) of Applied Data Science in Utrecht University
- semantic segmentation of urban green spaces from Sentinel-2 imagery for multiple cities around the globe
  - such as golf courses and cemeteries
- U-Net model from base level and U-Net model with ResNet-50 and VGG-16 backbones pretrained on ImageNet
- average OA, IoU, F-score, and AUC
- 13 cities for model training, 3 external cities for testing and prediction
 - San Francisco, Seattle, Denver, Philadelphia, Greater Manchester, Dublin, Amsterdam, Ghent, Dhaka, Vancouver, Dallas, London, and Buffalo – six from the USA, five from West Europe, one in Canada, and one in Asia
 - Washington D.C., Kampala, and Tel Aviv
- EarthExplorer, blue (B2), green (B3), red (B4), near-infrared (NIR, B8), and short-wave infrared (SWIR, B12),
- Spatial resolution for B2-B4 and B8 is 10m, and for B12 it is 20m

## Continental-Scale Building Detection from High Resolution Satellite Imagery
- https://sites.research.google/open-buildings/
- https://arxiv.org/pdf/2107.12283.pdf
- detecting buildings across the entire continent of Africa, South Asia, South-East Asia, Latin America and the Caribbean
- 50cm satellite imagery, a dataset of 100k satellite images, 1.75M manually labelled
building instances; further datasets for pre-training and self-training
- U-Net model,
- study variations in architecture, loss functions, regularization, pre-training,
self-training and post-processing that increase instance segmentation performance
  - use of mixup
  - self-training with soft KL loss
- Open Buildings dataset: 1.8 billion buildings (building polygons, building points and score thresholds)
- Other resources:
  - [DeepGlobe 2018 Challenge](http://deepglobe.org/)
  - [EarthVision 2019](http://www.classic.grss-ieee.org/earthvision2019/)

## ViTs for SITS: Vision Transformers for Satellite Image Time Series
- CVPR2023 https://arxiv.org/abs/2301.04944
- Temporo-Spatial Vision Transformer (TSViT)
- https://github.com/michaeltrs/DeepSatModels
- general Satellite Image Time Series (SITS) processing
- acquisition-time-specific temporal positional encodings
- multiple learnable class tokens

## Change-Aware Sampling and Contrastive Learning for Satellite Images
- [CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)
- use the temporal signal to contrast images with long-term and shortterm differences
- a new loss contrastive loss called ChangeAware Contrastive (CACo) Loss
- a novel method of sampling different geographical regions
- semantic segmentation, change detection

## Self-supervised Learning in Remote Sensing: A Review
- [Link to paper](https://arxiv.org/abs/2206.13188)
- [Benchmarking code, dataset and pre-trained models in PyTorch](https://github.com/zhu-xlab/SSL4EO-S12)
- Dataset: BigEarthNet, SEN12MS, So2Sat-LCZ42
- SSL for Earth Observation (SSL4EO)

## A generalizable and accessible approach to machine learning with global satellite imagery
- Nature Communications, 2021
- Multi-task Observation using Satellite Imagery & Kitchen Sinks (MOSAIKS)
- one-time unsupervised image featurization using random convolutional features

## Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data
https://arxiv.org/abs/2103.16607

## SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation
https://arxiv.org/abs/2211.07044

# Benchmark
## BigEarthNet
https://bigearth.net/
https://arxiv.org/pdf/2105.07921.pdf
https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models (TensorFlow)
- Remote Sensing Image Analysis (RSiM) Group and the Database Systems and Information Management (DIMA) Group at the Technische Universität Berlin
- 590,326 pairs of Sentinel-1 and Sentinel-2 image patches
- first version (v1.0-beta) of BigEarthNet includes only Sentinel 2 images
- 125 Sentinel-2 tiles acquired between June 2017 and May 2018 over the 10 countries (Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland)
- All the tiles were atmospherically corrected by the Sentinel-2 Level 2A product generation and formatting tool (sen2cor v2.5.5)
- divided into 590,326 non-overlapping image patches
- Each image patch was annotated by the multiple land-cover classes (i.e., multi-labels) that were provided from the CORINE Land Cover database of the year 2018 (CLC 2018)
- 66GB
### BigEarthNet-S1
- 321 Sentinel-1 scenes acquired between June 2017 and May 2018
- BigEarthNet-S1 consists of 590,326 preprocessed Sentinel-1 image patches - one for each Sentinel-2 patch
- 55GB
### BigEarthNet-MM
- multi-modal, Sentinel-1


# Datasets
## DeepGlobe Land Cover Classification Challenge dataset
- 1,146 satellite images of size 2448 x 2448 pixels
- training [dataset](https://www.kaggle.com/datasets/geoap96/deepglobe2018-landcover-segmentation-traindataset) with 803 images 
- 7 classes: Urban land,Agriculture land,Rangeland,Forest land,Water,Barren land,Unknow
- RGB images, masks are RGB image with with unique RGB values representing the class
- image size: 2,448x2,448
- resolution: 0.5m

## LandCover.ai
- central Poland
- resolution: 25 cm
- 4 classes: building,woodland,water,road

## LoveDA
- Nanjing, Changzhou, Wuhan
- background–1, building–2, road–3, water–4, barren–5,forest–6, agriculture–7, ignore-0
- Pre-trained urls for HRNet
- 0.3m, 1024x1024, segmentation masks are single-channel pngs
https://github.com/open-mmlab/mmsegmentation
https://github.com/Junjue-Wang/LoveDA

## Agriculture-Vision Challenge Dataset
https://github.com/SHI-Labs/Agriculture-Vision#agriculture-vision-challenge-dataset
- 7 classes: 0-background,1-cloud_shadow,2-double_plant,3-planter_skip,4-standing_water,5-waterway,6-weed_cluster
- 21,061 aerial farmland images captured throughout 2019 across the US
- 512x512x4, RGB+NIR
- 

## iSAID
https://captain-whu.github.io/DOTA/dataset.html
- oriented bounding box
- Google Earth, GF-2 and JL-1 satellite provided by the China Centre for Resources Satellite Data and Application, and aerial images provided by CycloMedia B.V. DOTA consists of RGB images and grayscale images
- object categories: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool, container crane, airport and helipad

# Libraries

## segment-geospatial (SAMGeo)
- https://samgeo.gishub.org/
- https://github.com/aliaksandr960/segment-anything-eo
- https://github.com/facebookresearch/segment-anything


## Segmentation models PyTorch
https://github.com/qubvel/segmentation_models.pytorch
- UNet, FPN, PSPNet, DeepLab...

## MMSegmentation
https://github.com/open-mmlab/mmsegmentation
- a lot of backbone and seg network architectures
- LoveDA, Potsdam, iSAID

## Mask2Former
https://github.com/facebookresearch/Mask2Former
- detectron2, for instance/semantic/panoptic/video segmentation
- dataset structure from COCO, json for annotation
- ResNet and Swin-Transformer backbones

# Terminology
orthoimage/orthophoto
orthorectified - geometrically corrected remote sensing image, scale is uniform, accurate representation of the Earth's surface
