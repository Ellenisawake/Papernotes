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
- 
# MLLM
## X2SAM: Any Segmentation in Images and Videos
- Sun Yat-sen University, Peng Cheng Laboratory, Meituan Inc.
- segmentation MLLM that extends any-segmentation capabilities from images to videos
- couples an LLM with a Mask Memory module
- Video Visual Grounded (V-VGD) segmentation benchmark
- [project page](https://wanghao9610.github.io/X2SAM/)
