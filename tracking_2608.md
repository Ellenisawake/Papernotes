# Multiple visually similar objects

### History-Aware Transformation of ReID Features for Multiple Object Tracking
- ECCV26, [paper](https://arxiv.org/pdf/2503.12562), Nanjing University
- a tailored Fisher Linear Discriminant (FLD) to project the raw ReID features into a sequence-specific representation space
- DanceTrack, SportsMOT

### BMW: Bidirectionally Memory bank reWriting for Unsupervised Person Re-Identification
- NeurIPS25
- memory banks should be rewritten with both intra-class and inter-class constraints


### No Train Yet Gain: Towards Generic Multi-Object Tracking in Sports and Beyond
- CVPR25 CV in Sports workshop
- [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.pdf)
- McByte, integrates temporally propagated segmentation mask as an association cue
- Evaluated on SportsMOT, DanceTrack, SoccerNet-tracking 2022 and MOT17
- Each tracklet gets its own mask, which is then propagated across frames
- Cutie for mask propagator, MeMOTR for first frame segmentation
- Ambiguity occurs when the IoU-based costs are low and similar for more than one entry in a row (or column) of the cost matrix
  - IoU-based matches when tracked objects are close
- Isolation occurs when relevant cost matrix entries contain too high values, not allowing for the association, and at the same time there is no ambiguity
- condition 1: check if  temporally propagated mask is visible
- condition 2: check if mask confidence above threshold
- condition 3: check if the mask fill ratio of the bounding box mf is sufficiently high
  - low could mean noisy or wrong masks
- condition 4: check if the bounding box coverage of the mask mc is sufficiently high
  - slightly below 1.0 is allowed

### When Fish Look Alike: Tracking Identities with Dual-branch Elasticity
- [paper](https://arxiv.org/abs/2607.26412)
- Adaptive Geometric Correspondence IoU, an association mechanism leveraging spatial and structural consistency to robustly handle complex morphological variations
- Lightweight L-branch HOTA 28.43 on MFT-Edge benchmark
- Scalable S-branch achieves HOTA 29.98
- cascaded two-stage matching
  - first stage, high-confidence matching utilizing the embedded appearance features
  - A cosine distance module evaluates the semantic similarity between the newly extracted appearance embeddings and the existing tracklet features
  - second stage: abandons appearance embeddings, performs robust geometric matching using our Adaptive Geometric Correspondence IoU (AGCIoU) metric; first associates remaining unmatched tracks with high confidence detections; secondary matching pass utilizing lowconfidence detections


### An appearance-independent multi-object tracking framework for group-housed pigs
- Computers and Electronics in Agriculture, October 2026
- Spatiotemporal Association Enhanced Pig Tracking
- reformulating identity maintenance in unmarked pig tracking as a two stage trajectory continuity problem under weak appearance cues
- nonlinear short term motion uncertainty and long term trajectory fragmentation are treated as two distinct but connected sources of identity inconsistency
- Unscented Kalman Filter (UKF) to better handle nonlinear motion such as sudden accelerations and sharp turns
- To address identity loss from long-term occlusion and trajectory fragmentation, an AFLink module constructs a global cost matrix from spatiotemporal trajectory continuity, combined with the Hungarian algorithm to recover fragmented identities without relying on appearance embeddings
  - AFlink proposed in StrongSORT paper
- Evaluated on both a self-collected and a public dataset
- baseline SORT, DeepSORT, BotSORT, and OCSORT

# Long term
### LTTrack: Rethinking the Tracking Framework for Long-Term Multi-Object Tracking
- Sichuan University, TCSVT2024
- Position-Based Association module, encodes relative and absolute positions as interaction and motion features
- long-lost target re-identification
- Zombie Track Re-Match
- experiments on MOT17, MOT20, and DanceTrack
- [github](https://github.com/Lin-Jiaping/LTTrack)

### An HMM-based framework for identity-aware long-term multi-object tracking from sparse and uncertain identification: use case on long-term tracking in livestock
- CV4Animals workshop at CVPR24, Laval university Canada
- [paper](https://drive.google.com/file/d/1_-6oLD4X2FHp3bo-Qp4PDtcpEMWr0kIL/view)
- combine uncertainty identity with tracking information
- 10-minute pig tracking dataset with 21 id at the feeding station of pen
- [github](https://github.com/ngobibibnbe/uncertain-identity-aware-tracking)
- [cv4animals 2024 archive](https://www.cv4animals.com/2024-home)

### SIPTrack: Reliability-Aware Identity Prediction for Sparse-Interval Pig Multi-Object Tracking with a New Benchmark
- Expert Systems with Applications, Aug 2026, Huazhong Agricultural University, Wuhan, China
- [paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417426028186#da01)
- PigMOT, a sparse-interval pig multi-object tracking benchmark
  - 46 video sequences (794.4min), including 2 night-time sequences, 25 FPS, 2560 × 1440
  - 3-second annotation interval
  - pigs were grouped into three categories: cough24, foot pain16, and healthy6
  - open-source DarkLabel tool for annotation, and the annotation format follows DanceTrack
  - 5,888 annotated frames, 69,732 bounding boxes, and 298 trajectories
- SIPTrack, a reliability-aware spatio-temporal identity prediction framework
  - Sparse-Interval Identity Memory, Temporally Constrained Spatio-Temporal Fusion
  - Reliability-Aware Identity Prediction and Recovery
  - Trajectory-level augmentation
- also benchmarked on PigTrack

### Long-term identity-consistent multi-pig tracking in group-housed pens
- Computers and Electronics in Agriculture, October 2026
- data association is performed by jointly exploiting appearance features and motion information, with Re-identification-based appearance modeling
- identity memory bank and a trajectory relinking mechanism
- Outperforms BOT-SORT and ByteTrack baselines in MOTA and IDF1 scores
- partial generalization in cross-scene conditions
- [paper](https://www.sciencedirect.com/science/article/pii/S0168169926007994)
- [code](https://github.com/glimmerc33-ui/pig_track)
- A rotated-object detector (YOLOv11-OBB) for pig detection
- An appearance-based Re-identification (ReID) network for identity modeling
- A trajectory management strategy with temporal smoothing and identity re-linking
  - a relinking search is first performed in the identity bank for new detections
- Joint motion–appearance data association for robust multi-object tracking
- a non-maximum suppression-like redundancy suppression strategy is applied to the same-frame trajectory set T during trajectory management

### GTA: Global Tracklet Association for Multi-Object Tracking in Sports
- [paper](https://arxiv.org/abs/2411.08216)
- Machine Learning and Computing for Visual Semantic Analysis (MLCSA) workshop at ACCV2024
- splitting tracklets containing multiple identities and connecting tracklets seemingly from the same identity
  - tracklet splitter: cluster and split reid features of each object
  - tracklet connector: use spatial constraints to determine min dist pair, then update feature
- SoccerNet dataset, SportsMOT dataset

# Dataset
### A Large-Scale Longitudinal Dataset for Pig Tracking and Re-Identification
- Smart Agricultural Technology
- Bristol Robotics Laboratory, Scotland’s Rural College (SRUC)
- YOLOv8, SAM2
- 740,000+ labelled frames, over 5.31 1.97 weeks, featuring 7.62 1.87 pigs per frame

### OinkTrack
- over five hours of annotated video across sixteen sequences, ABBs
- sequence ranges from one minute to one hour
- ACM Multimedia 2025
- [paper](https://dl.acm.org/doi/10.1145/3746027.3758189)
- [project page](https://leohuang0511.github.io/oinktrack-page/)










