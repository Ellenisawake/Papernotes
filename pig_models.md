
## Tracking
### Zero-Shot Multi-Animal Tracking in the Wild
- We modify SAM2MOT for multi-animal tracking
- adaptive detection threshold
- [paper](https://arxiv.org/pdf/2511.02591)
- [github](https://github.com/ecker-lab/SAM2-Animal-Tracking)


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
### Towards automatic farrowing monitoring—A Noisy Student approach for improving detection performance of newborn piglets
- Germany
- a Noisy Student approach to automatically generate annotation information, teacher-student model
- transform the image structure and generate pseudo-labels for the object classes piglet and tail
- a unique dataset for the detection and evaluation of newborn piglets and sow body parts
- [2024 paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310818)

### Multistage pig identification using a sequential ear tag detection pipeline
- detect pigs, localize ear tags, perform rotation correction via pin detection, and recognize digits
- generating a reliable ID proposal
- publicly proposing three custom datasets for ear tag, pin, and digit detection
- [2025 Scientific Reports paper](https://www.nature.com/articles/s41598-025-05283-8)

## Pose
### Multi-Object Keypoint Detection and Pose Estimation for Pigs
- 2025, Eindhoven University of Technology
- [paper](https://www.scitepress.org/Papers/2025/131701/131701.pdf)

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
### OinkTrack
- over five hours of annotated video across sixteen sequences, ABBs
- sequence ranges from one minute to one hour
- ACM Multimedia 2025
- [paper](https://dl.acm.org/doi/10.1145/3746027.3758189)
- [project page](https://leohuang0511.github.io/oinktrack-page/)


### SPMF-YOLO-Tracker: A Method for Quantifying Individual Activity Levels and Assessing Health in Newborn Piglets
- Nanjing Agricultural University, 2025, Agriculture
- enhances small-object detection by: SPDConv, MFM module, NWD loss function into YOLOv11
  - spatial–depth feature conversion module (SPDConv): transforms spatial structural information from layer P2 into depth-dimensional signals, thereby enriching low-level structural features
  - Modulation Fusion Module (MFM): performs attention-weighted fusion of feature maps across different resolutions
  - Normalized Wasserstein Distance as loss function instead of iou, improves localization for small or ambiguous targets
- ByteTrack
- quantified the cumulative movement distance of each newborn piglet within 30 min after birth
- Data: April 2025, Nanshang Nongke Pig Farm in Nanyang City, Henan Province
  - sows were of the Large White breed
  - Each pen housed one sow and a variable number of newborn piglets, ranging from 7 to 14, and was fitted with plastic slatted flooring and iron guardrails
  - Heat lamps were installed in the piglet activity areas, while natural daylight provided overall illumination
  - hemispherical camera (DS-2CD3326WDV3-I, Hikvision was mounted 1.8 m
  - 25 fps, 1920 × 1080
  - from the onset of farrowing to 30 min after delivery
  - For object detection, one frame was sampled every two seconds from the surveillance videos. We then used optical flow to compute the average motion magnitude between adjacent frames and the structural similarity index (SSIM) to remove redundant high-similarity frames
- Detection dataset: 1780 images, 1246 training, 267 validation, 267 test
  - annotated with the open-source software LabelMe 3.16.2
- Tracking dataset: 33 2.5 min clips from 10 farrowing pens
  - covered different stages from the sow’s farrowing process to the post-farrowing period
  - eight clips were selected from two low-light pens
  - sample one image every 25 frames, 4950 images in total
  - X-AnyLabeling 2.5.4, 
- input resolution of 640 × 640, 300 epochs, batch size 24, 
- [paper](https://www.mdpi.com/2077-0472/15/19/2087#)

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

## Review
Harnessing contactless monitoring technology for sows and piglets within farrowing pens: A critical review
- City University of Hong Kong, Smart Agricultural Technology 2025
- optical, infrared, audio, and radio-based sensing techniques
- applications in farrowing pens are then presented, which span from daily monitoring of sows and piglets to event-specific tasks (e.g., farrowing prediction, lactation behavior analysis, crushing detection, and health indicator monitoring)
- [Science Direct Paper](https://www.sciencedirect.com/science/article/pii/S2772375525005520?ref=cra_js_challenge&fr=RR-1)

