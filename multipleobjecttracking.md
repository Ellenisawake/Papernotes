# Multiple Objects Tracking/Video Instance Segmentation
Methods: tracking-by-detection, end-to-end

Metrics: HOTA, MOTA, MOTP, IDF1
## Overview
#### Deep Learning-Based Multi-Object Tracking
- [2025 review paper](https://arxiv.org/pdf/2506.13457)


## MOT
### Benchmarks
#### AI City Challenge Multi-Camera People Tracking
- tracking people across multiple cameras using an expanded synthetic dataset
- 1300 cameras, 3400 people, HOTA metric in 3D
- [website](https://www.aicitychallenge.org/2024-challenge-tracks/)

#### Tracking any object dataset and benchmark (TAO)
- cat, dog, swan

#### AnimalTrack
- IJCV2023
- 10 animal classes, including pig (rabbit, chicken, goose, duck, horse)
- 9-11 seconds of videos of multiple fast-moving animals
- [website](https://hengfan2010.github.io/projects/AnimalTrack/))
- [paper](https://arxiv.org/pdf/2205.00158)

### Algorithms
#### Multiple Object Tracking as ID Prediction (MOTIP)
- CVPR25, Nanjing University
- leveraging object-level features as tracking cues, decodes the ID labels for current detections
  - trainable ID Decoder head
- built on top of Deforamable DETR, MOTR
- [paper](https://arxiv.org/pdf/2403.16848)
- [github](https://github.com/MCG-NJU/MOTIP)

#### MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors
- CVPR23, Megvii
- improvement over MOTR (end-to-end tracking with transformer): use separate strong pre-trained detector (YOLOX) to provide more accurate boxes as anchors for MOTR
- DanceTrack, BDD100K
- [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MOTRv2_Bootstrapping_End-to-End_Multi-Object_Tracking_by_Pretrained_Object_Detectors_CVPR_2023_paper.pdf)
- [github](https://github.com/megvii-research/MOTRv2)

#### CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online Multi-Object Tracking
- UCLouvain, 2025 ArXiv
- an association method which can utlise different types of cues (box, keypoints etc) for detection-tracklet association
- employs two transformerbased modules 
- association-centric training scheme 
- model the complex interactions between tracked targets and their various association cues
- a set of Temporal Encoders (TE) that aggregate each tracking cue into tracklet-level representations
- a Group-Aware Feature-Fusion Encoder (GAFFE) that jointly transforms all cues into unified disentangled representations for each tracklet and detection
- ReId model/embeddings: BPBReID, KPR (ECCV24)
  - ReId dataset using train/val splits of MOT dataset
  - swin transformer backbone with SOLIDER person foundation model
  - 34.4\% mAP on DanceTrack
  - split training to two halve, use reid model trained on one half to produce embeddings for the other half to avoid overfitting embeddings
  - val set embeddings using reid model trained on full train set
- [paper](https://arxiv.org/pdf/2505.01257)
- [github](https://github.com/TrackingLaboratory/CAMELTrack)


#### BoostTrack++
- SOTA on MOT20
- Improved similarity score calculation: shape+ Mahalanobis distance+soft BIoU, on top of BoostTrack+

#### Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown
- Latest on Arxiv

#### MCBLT: Multi-Camera Multi-Object 3D Tracking in Long Videos
- TUM
- aggregates multi-view images with necessary camera calibration parameters to obtain 3D object detections
- hierarchical graph neural networks (GNNs) to track these 3D detections in bird's eye view
- ReID features
- AICity'24 dataset, WildTrack dataset

#### LTTrack: Rethinking the Tracking Framework for Long-Term Multi-Object Tracking
- Sichuan University, TCSVT2024
- Position-Based Association module, encodes relative and absolute positions as interaction and motion features
- long-lost target re-identification
- Zombie Track Re-Match
- experiments on MOT17, MOT20, and DanceTrack
- [github](https://github.com/Lin-Jiaping/LTTrack)

#### An HMM-based framework for identity-aware long-term multi-object tracking from sparse and uncertain identification: use case on long-term tracking in livestock
- CV4Animals workshop at CVPR24, Laval university Canada
- [paper](https://drive.google.com/file/d/1_-6oLD4X2FHp3bo-Qp4PDtcpEMWr0kIL/view)
- combine uncertainty identity with tracking information
- 10-minute pig tracking dataset with 21 id at the feeding station of pen
- [github](https://github.com/ngobibibnbe/uncertain-identity-aware-tracking)
- [cv4animals 2024 archive](https://www.cv4animals.com/2024-home)

#### Simple Online and Realtime Tracking -based methods (SORT)
- tracking-by-detection, robust for limited data
- Kalman Filters for motion prediction
- Hungarian algorithm for data association via IoU-based cost matrix
- real-time, minimal computational overhead
- **DeepSORT**: add appearance embeddings
  - Mahalanobis distance + cosine distance of appearance features
  - Matching is based on a combined motion + appearance distance metric
  - same Kalman Filter structure for motion modeling
- other extensions
  - ByteTrack: Uses low-confidence detections in association, improving recall
  - OC-SORT: Adds a non-linear observer model to reduce ID switches and improve long-term tracking
  - BoT-SORT: Integrates stronger Re-ID models and IoU-based association strategies
  - FairMOT: Combines detection and Re-ID feature learning in a single network for efficiency 

## VIS
### Benchmarks
#### YouTube-VIS
- animal class
  - eagle, shark, horse, cow, person, ape, giant panda, parrot, lizard, dog, monkey, cat, rabbit, snake, duck, fox, bear, turtle, leopard, fish, deer, zebra, owl, giraffe, elephant, frog, tiger, mouse, seal
- 2019 version: 2238 train + 302 val + 343 test, 40 catefories, 4883 instances, 131k annotations
- 2021 version: 2985 train + 421 val + 453 test, improved 40 catefories, 8171 instances, 232k annotations
- 2022 version: additional long videos for val/test
- evaluation metrics: AP, AR, 
- [website](https://youtube-vos.org/dataset/vis/)
- [tech report](https://arxiv.org/pdf/1905.04804)
- [leaderboard](https://youtube-vos.org/challenge/2022/leaderboard/)
- [first place on VIS](https://youtube-vos.org/assets/challenge/2022/reports/VIS_1st.pdf)
- [2nd place on VIS](https://arxiv.org/pdf/2206.07011)
- [LSVOS challenge 2024](https://lsvos.github.io/)

#### LVOS: long-term video object segmentation dataset
- part of LSVOS challenge 2024
- Each sequence lasting 1.14 minutes on average (!)
- 12 animal classes: elephant, sheep, giraffe, goldfish, lion, bear, zebra, tiger, monkey, kangaroo, shark, gorilla
- evaluation metrics: Region Similarity, Contour Accuracy, and Standard Deviation
- workshop and challenge at ICCV2023
- [website](https://lingyihongfd.github.io/lvos.github.io/)
- [paper](https://arxiv.org/pdf/2211.10181)

#### VOST: Video Object Segmentation under Transformations
- focus on object transformations
  
#### DAVIS
- diverse categories, including dogs/cats/cows etc.
- semi-supervised and unsupervised segmentation
- [website](https://davischallenge.org/index.html)

### Algorithms
#### SeqFormer
- ECCV22, based on Deformable DETR and VisTR
- [model zoo](https://github.com/wjf5203/SeqFormer)
#### SAM 2: Segment Anything in Images and Videos
- Meta, [github](https://github.com/facebookresearch/sam2)
#### Cutie: Putting the Object Back
- a follow-up work of XMem, support interactive segmentation
- [github](https://github.com/hkchengrex/Cutie?tab=readme-ov-file)
- UIUC & Adobe
#### DEVA: Tracking Anything with Decoupled Video Segmentation
- long-term, open-vocabulary video segmentation with text-prompts out-of-the-box
- task-specific image-level segmentation and class/task-agnostic bi-directional temporal propagation
- [github](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
- UIUC & Adobe, ICCV23
#### Grounded SAM 2: Ground and Track Anything in Videos
- Grounding DINO and SAM 2
-   Transformer-based detector DINO with grounded pre-training
- [github](https://github.com/IDEA-Research/Grounded-SAM-2)
- [grounded-sam](https://github.com/IDEA-Research/Grounded-Segment-Anything)

#### Grounding DINO
- DETR with Improved deNoising anchOr boxes
- [paper](https://arxiv.org/pdf/2303.05499)

#### VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement
- [GitHub](https://github.com/KimHanjung/VISAGE?tab=readme-ov-file)
- ECCV24
- explicitly extract embeddings from backbone features and drive queries to capture the appearances of objects


#### 1st place on YouTube-VIS 2022
- [first place on VIS](https://youtube-vos.org/assets/challenge/2022/reports/VIS_1st.pdf)
- ByteDance
- Deformable DETR, dynamic mask head, simulateneously predict box+class+mask (SeqFormer)
- contrastive learning between a key frame and a reference frame
  - reference frame is selected from temporal neighborhood of key frame
  - what if no positive sample on the reference frame, e.g., object instance not found???
- empty memory bank on initialization
- CNN backbone (ResNet-50) extracts multi-scale feature maps
- deformable DETR module takes the feature maps with additional fixed positional encodings and N learnable object
queries as input
- a light-weighted FFN as a contrastive head to decode the contrastive embeddings from the instance features
- object queries are first transformed into output embeddings by the transformer decoder, then decoded into box coordinates, class labels, and instance masks
- $L=L_cls + \lambda1 * L_box + \lambda2 * L_mask + \lambda3 * L_emb$
- number of object queries is set to 300
- pre-trained on the MS COCO 2017 for segmentation head, random crops from COCO to pretrain constrastive learning head
- multi-crops during training at 11 scales (320-640), test at both single/multi-scales
- 4 pairs per GPU, 8 GPU, 12k iterations
- ensemble Swin-L and ConvNext-L

#### Consistent Video Instance Segmentation with Inter-Frame Recurrent Attention
- [2nd place on YouTube-VIS 2022](https://arxiv.org/pdf/2206.07011)
- Microsoft


#### DVIS, 1st place on YouTube-VIS 2023 :fire:
- decouple tasks into segmentation, tracking and refinement
  - focus on robustly associating objects across adjacent frames, refiner to improve both segmentation & tracking
- based on Mask2Former
- tracker and refiner operate exclusively on the instance representations output by the segmenter
- R50/Swin-L backbone
- tracker and refiner can be trained separately while keeping other components frozen
  - trained on a single GPU with 11G memory
- Wuhan University & Kuaishou Technology 
- [model zoo](https://github.com/zhang-tao-whu/DVIS/tree/main)
- [tech report on arXiv with further improvements to DVIS for the challenge](https://arxiv.org/pdf/2308.14392)
- [DVIS paper](https://arxiv.org/pdf/2306.03413)

#### CSS-Segment, 2nd place on LSVOS challenge 2024 :fire:
-  CSS-Segment by efficiently integrating the advantageous modules of Cutie, SAM, and SAM2
- Xidian University
- [paper](https://arxiv.org/pdf/2408.13582)

#### 3rd place on LSVOS challenge 2024
- combines Cutie and SAM2
- Xidian University
- [paper](https://arxiv.org/pdf/2408.10469)


- []()
