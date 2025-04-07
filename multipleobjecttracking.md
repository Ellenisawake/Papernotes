# Multiple Objects Tracking/Video Instance Segmentation
Methods: tracking-by-detection, end-to-end

Metrics: HOTA, MOTA, MOTP, IDF1

## MOT
### Benchmarks
#### AI City Challenge Multi-Camera People Tracking
- tracking people across multiple cameras using an expanded synthetic dataset
- 1300 cameras, 3400 people, HOTA metric in 3D
- [website](https://www.aicitychallenge.org/2024-challenge-tracks/)

#### Tracking any object dataset and benchmark (TAO)
- cat, dog, swan

### Algorithms
#### BoostTrack++
- SOTA on MOT20
- Improved similarity score calculation: shape+ Mahalanobis distance+soft BIoU, on top of BoostTrack+

#### Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown
- Latest on Arxiv

#### MCBLT: Multi-Camera Multi-Object 3D Tracking in Long Videos
- TUM
- aggregates multi-view images with necessary camera calibration parameters to obtain 3D object detections
- hierarchical graph neural networks (GNNs) to track these 3D detections in bird's eye view
- AICity'24 dataset, WildTrack dataset

## VIS
### Benchmarks
#### YouTube-VIS
- animal class (eagle, shark, horse, cow, person, ape, giant panda, parrot, lizard, dog, monkey, cat, rabbit, snake, duck, fox, bear, turtle, leopard, fish, deer, zebra, owl, giraffe, elephant, frog, tiger, mouse, seal)
  - 
- 2019 version: 2238 train + 302 val + 343 test, 40 catefories, 4883 instances, 131k annotations
- 2021 version: 2985 train + 421 val + 453 test, improved 40 catefories, 8171 instances, 232k annotations
- 2022 version: additional long videos for val/test
- evaluation metrics: AP, AR, 
- [website](https://youtube-vos.org/dataset/vis/)
- [tech report](https://arxiv.org/pdf/1905.04804)

### Algorithms
#### VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement
- [GitHub](https://github.com/KimHanjung/VISAGE?tab=readme-ov-file)
- ECCV24

