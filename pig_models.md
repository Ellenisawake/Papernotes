
## Tracking
### CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online Multi-Object Tracking
- [github](https://github.com/TrackingLaboratory/CAMELTrack?tab=readme-ov-file)
- employs two transformer-based modules
- Association-Centric Training
- built on top of [TrackLab](https://github.com/TrackingLaboratory/tracklab)
- evaluated on MOT17, MOT20, DanceTrack, BEE24, PoseTrack21
- model weights availabel

### TrackLab
- [TrackLab](https://github.com/TrackingLaboratory/tracklab)
- support YOLO, YOLOX, RTMDet, RTDetr
- 


### NetTrack: Tracking Highly Dynamic Objects with a Net
- CVPR24
- [project](https://george-zhuang.github.io/nettrack/)
- []()

### Multiple Object Tracking as ID Prediction
- [paper](https://arxiv.org/pdf/2403.16848)

### PKL-Track: A Keypoint-Optimized approach for piglet tracking and activity measurement
- [paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169925006842#f0010)
- Optimized YOLOv11s-Pose for precise keypoint localization
- Kpt-IoU and Norm-L2 matching strategies

### Object Concepts Emerge from Motion
- [paper](https://arxiv.org/pdf/2505.21635)

## Detection
### Multi-Object Keypoint Detection and Pose Estimation for Pigs
- 2025, Eindhoven University of Technology
- [paper](https://www.scitepress.org/Papers/2025/131701/131701.pdf)

## Pose
### MT-SRNet: A Transferable Multi-Task Super-Resolution Network for Pig Keypoint Detection, Segmentation, and Posture Estimation
- Tomas Norton, Catholic University of Leuven, accepted May 2025
- [Computers and Electronics paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169925006398?via%3Dihub)
- a Rotated Bounding Box (RBB) detector combined with an Auto-Visual Prompt strategy
- a Multi-task Super-Resolution Network (MT-SRnet) simultaneously predicting pig keypoints, masks, and postures from low-resolution inputs
- [project page](https://gitlab.kuleuven.be/m3-biores/public/m3pig)

### YOLOPose
- https://github.com/ultralytics/ultralytics/issues/1915
- https://docs.ultralytics.com/tasks/pose/#models
- https://docs.ultralytics.com/datasets/pose/tiger-pose/#usage
- https://blog.roboflow.com/train-a-custom-yolov8-pose-estimation-model/#step-4-train-a-yolov8-keypoint-detection-model


### MMPose
- https://github.com/open-mmlab/mmpose/blob/main/docs/en/dataset_zoo/2d_animal_keypoint.md
- https://github.com/jgraving/DeepPoseKit-Data/tree/master/datasets/zebra
- https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_animal_demo.md

## Multimodal methods
### Transformer-based audio-visual multimodal fusion for fine-grained recognition of individual sow nursing behaviour
- South China Agricultural University
- Audio-visual fusion framework for sow nursing sound and behaviour recognition
- Sound source localisation system
- [paper in Artificial Intelligence in Agriculture 2025](https://www.sciencedirect.com/science/article/pii/S2589721725000376)

## Dataset
### PigTrack
- Benchmarking pig detection and tracking under diverse and challenging conditions
- University of Gottingen Germany, 2025
- 80 video sequences, conventional pig farming, top view cameras, axis-aligned boxes, 24.3GB
- Model weights of MOTRv2 and MOTIP trained for pig tracking
- built on top of TrackEval, MOTR, ByteTrack, Deformable DETR, YOLOX, OC-SORT, DanceTrack, BDD100K, mmdetection
- end-to-end models show better association performance
- [homepage](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/P7VQTP)
- [github](https://github.com/jonaden94/PigBench)
- [2025 paper](https://arxiv.org/pdf/2507.16639)


### PigLife
- UIUC, 2023, 16GB
- detection, segmentation, posture & behavior labels, occlusion,
- [dataset page](https://data.aifarms.org/view/piglife)
