# Multiple Objects Tracking/Video Instance Segmentation
Methods: tracking-by-detection, end-to-end
Metrics: HOTA, MOTA, MOTP, IDF1

## MOT
### BoostTrack++
- SOTA on MOT20
- Improved similarity score calculation: shape+ Mahalanobis distance+soft BIoU, on top of BoostTrack+

### Tracking any object dataset and benchmark (TAO)
- cat, dog, swan

### YouTube VIS dataset
- with animal class

### Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown
- Latest on Arxiv

### MCBLT: Multi-Camera Multi-Object 3D Tracking in Long Videos
- TUM
- aggregates multi-view images with necessary camera calibration parameters to obtain 3D object detections
- hierarchical graph neural networks (GNNs) to track these 3D detections in bird's eye view
- AICity'24 dataset, WildTrack dataset

## VIS
### VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement
- [GitHub](https://github.com/KimHanjung/VISAGE?tab=readme-ov-file)
- ECCV24

