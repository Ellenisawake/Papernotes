# SAM3
## SAM3.1
## object multiplexing
- [blog](https://ai.meta.com/blog/segment-anything-model-3/)
- allows the model to track up to 16 objects in a single forward pass
- throughput at 32 frames per second on a single H100 GPU

## ViewSAM: Learning View-aware Cross-modal Semantics for Weakly Supervised Cross-view Referring Multi-Object Tracking
- Southeast University, Queen Mary University of London
- repurpose foundation models as pseudo-label generators
- design an Affinity-guided Cross-view Re-prompting trategy to refine and associate SAM3-generated tracklets across cameras
- ViewSAM, a CRMOT model built upon SAM2 that explicitly models view-aware cross-modal semantics
- [arXiv](https://arxiv.org/pdf/2605.02638)



# SAM2
## A Distractor-Aware Memory (DAM) for Visual Object Tracking with SAM2
- [github](https://github.com/jovanavidenovic/DAM4SAM)
- CVPR25, IJCV26, University of Ljubljana
- [MOT version github](https://github.com/alanlukezic/d4sm)
- SAM2.1
- distractor-distilled (DiDi) dataset
- [arXiv](https://arxiv.org/pdf/2411.17576)

## OAMVOS:2nd Report for 5th PVUW MOSE Track
- [arXiv](https://arxiv.org/pdf/2604.22837)
- occlusion- and reappearance-aware extension of DAM4SAM
- add a control layer that explicitly distinguishes stable tracking from uncertain tracking
  - stable mode, ambiguous mode, recovery mode to decide if going through several branches

## SAMIDARE: Tracking-by-Segmentation for Dense Scenarios
- [github](https://github.com/ZabuZabuZabu/SAMIDARE)
- enhances SAM2MOT for crowded scenes
- density-aware mask re-generation
- selective memory updates
- state-aware association and new track initialization
- SportsMOT dataset

## SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree
- [github](https://github.com/Mark12Ding/SAM2Long)
- ICCV 2025, CUHK, Shanghai AI Lab
- a training-free memory tree, maintaining diverse segmentation hypotheses, dynamically pruning less optimal paths

## SAM 2++: Tracking Anything at Any Granularity
- [arXiv](https://arxiv.org/pdf/2510.18822)
- [project page](https://tracking-any-granularity.github.io/)
- a unified model towards tracking at any granularity, including masks, boxes, and points

## SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation
- [arXiv](https://arxiv.org/pdf/2504.04519)
- [github](https://github.com/TripleJoy/SAM2MOT)
- integrates pre-trained detector, pre-trained segmentor with tracking logic into a zero-shot MOT system
- mmdetection
- AAAI 2026, Huawei
- Experiments on DanceTrack, UAVDT, and BDD100K

# MLLM
## X2SAM: Any Segmentation in Images and Videos
- Sun Yat-sen University, Peng Cheng Laboratory, Meituan Inc.
- segmentation MLLM that extends any-segmentation capabilities from images to videos
- couples an LLM with a Mask Memory module
- Video Visual Grounded (V-VGD) segmentation benchmark
- [project page](https://wanghao9610.github.io/X2SAM/)
