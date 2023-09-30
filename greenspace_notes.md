# Papers
## Mapping Public Urban Green Spaces Based on OpenStreetMap and Sentinel-2 Imagery Using Belief Functions
- [link to paper](https://www.mdpi.com/2220-9964/10/4/251)


## The potentials of Sentinel-2 and LandSat-8 data in green infrastructure extraction, using object based image analysis (OBIA) method
- [link to paper]([https://www.mdpi.com/2220-9964/10/4/251](https://www.tandfonline.com/doi/full/10.1080/22797254.2017.1419441)https://www.tandfonline.com/doi/full/10.1080/22797254.2017.1419441)

## A 10 m resolution urban green space map for major Latin American cities from Sentinel-2 remote sensing images and OpenStreetMap
- [link to resources](https://figshare.com/articles/dataset/A_10_m_resolution_urban_green_space_map_for_major_Latin_American_cities_from_Sentinel-2_remote_sensing_images_and_OpenStreetMap/19803790)
- 371 major Latin American cities as of 2017
- [github](https://github.com/yangju-90/urban_greenspace_classification/tree/main)
- supervised classification of Sentinel-2 satellite imagery and UGS samples derived from OpenStreetMap
- binary UGS maps at 10 m spatial resolution in GEOTIFF format
- a shapefile of mapped boundaries
- .prj files containing projection information


## UGS-1m: fine-grained urban green space mapping of 31 major cities in China based on the deep learning framework
- [link to paper](https://essd.copernicus.org/articles/15/555/2023/)
- [additional github](https://github.com/liumency/UGS-1m)
- [link2](https://www.scidb.cn/en/detail?dataSetId=36af2aed281e4c82aa8a3cd3f1211a37#p2)
- UGS: parks, green buffers, square green spaces, attached green spaces, and other green spaces
- urban green space dataset UGSet: 4544 samples, 512x512 (142 sample areas in Guangdong, Gaofen-2 (GF2) satellite)
  - spatial resolution of about 1 m, which is equipped with two high-resolution, 1 m panchromatic and 4 m multispectral cameras  
- a generator: fully convolutional network designed for UGS extraction (UGSNet) with attention, pre-trained on UGSet
- discriminator: fully connected network aiming to deal with the domain shift between images
- fine-tuning: 2179 Google Earth images, 1.1m spatial resolution


## Automatically Mapping Urban Green Space Using Sentinel-2 Imagery and Deep Learning Methods in Multiple Cities Worldwide: A Convolutional Neural Network Approach
- [Master thesis](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/42573/Automatically%20Mapping%20Urban%20Green%20Space%20Using%20Sentinel-2%20Imagery%20and%20Deep%20Learning%20Methods%20in%20Multiple%20Cities%20Worldwide%20A%20Convolutional%20Neural%20Network%20Approach.pdf?sequence=1&isAllowed=y) of Applied Data Science in Utrecht University
- Model
  - semantic segmentation of urban green spaces from Sentinel-2 imagery for multiple cities around the globe
    - such as golf courses and cemeteries
  - U-Net model from base level and U-Net model with ResNet-50 and VGG-16 backbones pretrained on ImageNet
  - Binary Focal Loss
  - average OA, IoU, F-score, and AUC
  - batch size of 16, total epochs for training were set to be 50
  - 256 pixels×256 pixels input, 
- Data
  - 4299 training image chip pairs, number of pixels of green space 71.9 million, and backgrounds 209.9 million
  - 13 cities for model training, 3 external cities for testing and prediction
    - San Francisco, Seattle, Denver, Philadelphia, Greater Manchester, Dublin, Amsterdam, Ghent, Dhaka, Vancouver, Dallas, London, and Buffalo – six from the USA, five from West Europe, one in Canada, and one in Asia
   - Washington D.C., Kampala, and Tel Aviv
  - EarthExplorer, blue (B2), green (B3), red (B4), near-infrared (NIR, B8), and short-wave infrared (SWIR, B12),
  - Spatial resolution for B2-B4 and B8 is 10m, and for B12 it is 20m, overlapping of chips to let each training city has a similar contribution
  - ground truth datasets for UGS
    - [PAD-US 2.1 dataset](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download)
    - OS [Mastermap Greenspace Layer](https://www.ordnancesurvey.co.uk/products/os-mastermap-greenspace-layer#technical), Ordnance Survey UK
    - [WorldCover](https://esa-worldcover.org/en), 10m, Sentinel-1 and Sentinel-2
  - Indices: NDVI, NDWI, Normalized Difference Built-up Index (NDBI)
  - removed the image chips whose proportion of backgrounds (non-UGS pixels) is more than a certain threshold (86%)
- Data augmentation
  - random rotation within an angle of 45°
  - random width and height shift within a range of 20%
  - randomly horizontal and vertical flip
  - randomly zooming in and out within a range of 20%
- Results:
  - additional UGS identified: golf courses, national parks, and cemeteries



# Code
## Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI
- [github]([https://www.mdpi.com/2220-9964/10/4/251](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/tree/main/Jiawei%27s-approach)https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/tree/main/Jiawei%27s-approach)
