# Papers
## A generalizable and accessible approach to machine learning with global satellite imagery
- Nature Communications, 2021
- Multi-task Observation using Satellite Imagery & Kitchen Sinks (MOSAIKS)
- one-time unsupervised image featurization using random convolutional features

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


# Libraries
## Segmentation models PyTorch
https://github.com/qubvel/segmentation_models.pytorch
- UNet, FPN, PSPNet, DeepLab...

## MMSegmentation
https://github.com/open-mmlab/mmsegmentation
- a lot...


# Terminology
orthoimage/orthophoto
orthorectified - geometrically corrected remote sensing image, scale is uniform, accurate representation of the Earth's surface
