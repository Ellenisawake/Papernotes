## 1. [Beyond Explicit Language: Plug-and-Play Visual-to-Linguistic Modeling Toward General Object Tracking]
(https://openaccess.thecvf.com//content/CVPR2026/papers/Lan_Beyond_Explicit_Language_Plug-and-Play_Visual-to-Linguistic_Modeling_Toward_General_Object_Tracking_CVPR_2026_paper.pdf)
**Organization:** Zhejiang University of Technology

### Abstract Summary
Existing vision-language tracking (VLT) methods require explicit, pre-defined text descriptions that cannot adapt to a moving target, and fail entirely without text input. The authors propose a **plug-and-play Textual Inversion Module (TIM)** that implicitly generates linguistic representations directly from visual features — no explicit language input needed at inference. Visual features from both template and search regions are projected into the CLIP text embedding space ("textual inversion"), then injected back into the visual backbone via a **Multi-Layer Semantic Injection Mechanism** using cross-attention. The approach is training-efficient (only the new module and decoder are trained) and improves multiple state-of-the-art trackers on both visual and VL benchmarks.

### Introduction Key Points
- Problem: static text descriptions go stale as the target moves; models without text regress to visual-only.
- Solution: dynamic, patch-level implicit descriptions generated from visual context at every frame.
- No external text needed at inference; addresses the "language unavailability" failure mode.
- Demonstrates improved performance on MCITrack, DUTrack, SeqTrack with minimal overhead.

---

## 2. [Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking: Make it Strong Again]
(https://openaccess.thecvf.com//content/CVPR2026/papers/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.pdf)
**Organization:** Beijing University of Posts and Telecommunications

### Abstract Summary
Referring Multi-Object Tracking (RMOT) must track multiple objects specified by natural language in videos. One-stage methods have dominated recently, but the **two-stage Referring-by-Tracking (RBT)** paradigm offers lower training cost and incremental deployability. The authors identify two weaknesses in current two-stage RBT: (1) *overly heuristic feature construction* (redundant dual-encoding of full image and crop) and (2) *fragile correspondence modeling* (cosine similarity in CLIP space that breaks when the backbone changes). They propose **FlexHook**, with a **Conditioning Hook (C-Hook)** that samples target features directly from the visual backbone (restoring gradient flow) and injects language-conditioned auxiliary points. A **Pairwise Correspondence Decoder (PCD)** replaces static cosine similarity with learnable pairwise discrimination. FlexHook is the first two-stage RBT to outperform SOTA one-stage methods on Refer-KITTI/v2, Refer-Dance, and LaMOT.

### Introduction Key Points
- Three RMOT paradigms: Tracking-by-Referring (TBR), one-stage RBT, two-stage RBT.
- Two-stage RBT's incremental flexibility is valuable but badly underexploited (iKUN reaches only 10.32 HOTA on Refer-KITTI-v2).
- C-Hook avoids duplicating backbone computation; PCD eliminates CLIP dependence.

---

## 3. [Boosting Self-Supervised Tracking with Contextual Prompts and Noise Learning]
(https://openaccess.thecvf.com//content/CVPR2026/papers/Zheng_Boosting_Self-Supervised_Tracking_with_Contextual_Prompts_and_Noise_Learning_CVPR_2026_paper.pdf)
**Organizations:** Guangxi Normal University; University of Southampton

### Abstract Summary
Self-supervised trackers lack explicit context modeling, and directly transplanting the supervised-context-association paradigm (random non-semantic queries) to unlabeled video fails. **PNTrack** introduces a **dual-modal contextual association mechanism**: following an easy-to-hard curriculum, it first uses *instance patch tokens (prompts)* from the same video sequence, then progressively injects *background tokens (noise)* to harden the learned representations. Applied only during training (no overhead at inference), it achieves new SOTA on 8 benchmarks (GOT10K, LaSOT, TrackingNet, VOT2020, TNL2K, UAV123, OTB100, LaSOT_ext) with limited annotations.

### Introduction Key Points
- Cycle-consistency self-supervised trackers lack temporal context modeling.
- Non-semantic random queries are unsuitable for extracting reliable cues from unlabeled frames.
- Easy-to-hard: prompts first (reliable supervision signal), noise later (harder, more robust representations).
- Purely a training-time intervention — inference is identical to standard trackers.

---

## 4. [Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking]
(https://openaccess.thecvf.com//content/CVPR2026/papers/Li_Occlusion-Aware_SORT_Observing_Occlusion_for_Robust_Multi-Object_Tracking_CVPR_2026_paper.pdf)
**Organizations:** Sichuan University; Institute of Optics and Electronics, Chinese Academy of Sciences

### Abstract Summary
2D MOT suffers from *positional cost confusion* when objects occlude each other — the cost matrix can no longer reliably reflect detection-to-trajectory affinity, causing identity switches. **OA-SORT** is a plug-and-play, training-free framework with three components: (1) **Occlusion-Aware Module (OAM)** — estimates occlusion severity per object using a Gaussian Map to suppress background interference; (2) **Occlusion-Aware Offset (OAO)** — integrates occlusion coefficients into spatial consistency metrics; (3) **Bias-Aware Momentum (BAM)** — uses occlusion severity to dampen noisy predictions from the Kalman Filter. Achieves 63.1% HOTA and 64.2% IDF1 on DanceTrack; integrating into four other trackers yields average +2.08% HOTA and +3.05% IDF1.

### Introduction Key Points
- Existing cues (appearance, motion direction, detection confidence) all degrade under occlusion — the root issue remains unaddressed.
- OA-SORT directly estimates and exploits occlusion severity rather than indirectly compensating for it.
- Training-free and modular: can be bolted onto any SORT-family tracker without retraining.

---

## 5. [VidEoMT: Your ViT is Secretly Also a Video Segmentation Model]
(https://openaccess.thecvf.com//content/CVPR2026/papers/Norouzi_VidEoMT_Your_ViT_is_Secretly_Also_a_Video_Segmentation_Model_CVPR_2026_paper.pdf)
**Organizations:** Eindhoven University of Technology; RWTH Aachen University

### Abstract Summary
Existing online video segmentation models pair a per-frame segmenter with complex, hand-crafted tracking modules. 
**VidEoMT** (Video Encoder-only Mask Transformer) hypothesises that a large pre-trained ViT, scaled with DINOv2-style training, can absorb the tracking role too. It adds only: (1) **query propagation** — object-level queries are carried from frame *t-1* to frame *t*; and (2) **query fusion** — propagated queries are merged with learnable temporally-agnostic queries to handle newly appearing objects. No separate tracker, adapter, pixel decoder, or ReID layers needed. Result: 5×–10× faster than SOTA (up to 160 FPS with ViT-L), competitive accuracy on YouTube-VIS 2019, VIPSeg, and VSPW.

### Introduction Key Points
- Complexity inflation trend in video segmentation motivates a search for a simpler approach.
- DINOv2-style pre-training (cross-view consistency objective) naturally induces tracking-useful features.
- Inspired by EoMT (image segmentation without specialized components); extends the idea to video.
- Query propagation + fusion is the only temporal addition — everything else is inherited from the ViT.

---

## 6. [Learning to Track Instance from Single Natural Language Description]
(https://openaccess.thecvf.com//content/CVPR2026/papers/Zheng_Learning_to_Track_Instance_from_Single_Nature_Language_Description_CVPR_2026_paper.pdf)
**Organizations:** Guangxi Normal University; University of Southampton

### Abstract Summary
Fully supervised VL trackers need millions of bounding-box annotations, while natural language descriptions are cheap to obtain. 
**SVLTrack** tackles *self-supervised VL tracking*: given only a single natural language description (no bounding-box labels at training time), track the referred object across a video. It uses a large VLM to generate pseudo bounding boxes for unlabeled videos, then learns via a **Dynamic Token Aggregation Module** that treats visual tokens unequally: (i) select key target tokens from the template based on an anchor token; (ii) merge them by attention score and fuse into language tokens (removing visual noise, improving semantic alignment); (iii) use fused language tokens to extract and propagate target tokens across search frames. Surpasses SOTA self-supervised methods on VL tracking benchmarks.

### Introduction Key Points
- Fully supervised VL methods (e.g., JointNLT) need 3.5M+ bounding box annotations — prohibitively expensive.
- Equal-weight token fusion wastes computation and dilutes language-visual alignment.
- LVLM-generated pseudo-labels replace bounding-box annotations entirely.
- Self-supervised VL tracking enables natural-language-driven human-computer interaction without manual annotation.

---

## Comparative Analysis

| Dimension | Lan et al. (Beyond Language) | Du et al. (FlexHook) | Zheng et al. (PNTrack) | Li et al. (OA-SORT) | Norouzi et al. (VidEoMT) | Zheng et al. (SVLTrack) |
|---|---|---|---|---|---|---|
| **Task** | Single Object Tracking (SOT) | Referring MOT (RMOT) | Self-supervised SOT | Multi-Object Tracking (MOT) | Video Instance Segmentation | Self-supervised VL SOT |
| **Supervision** | Fully supervised | Fully supervised | Self-supervised | Training-free | Fully supervised | Self-supervised |
| **Modality** | Visual → implicit language | Visual + explicit language | Visual only | Visual (position) | Visual only | Visual + explicit language |
| **Core novelty** | Visual-to-text inversion without explicit text | Hook-based feature sampling + pairwise correspondence | Prompt+noise curriculum for unlabeled video | Occlusion severity estimation for cost de-confusion | Encoder-only ViT with query propagation | Dynamic token aggregation for self-supervised VL tracking |
| **Plug-and-play?** | Yes (module added to existing trackers) | No (new end-to-end framework) | No (training paradigm change) | Yes (training-free, modular) | No (new architecture) | No (new framework) |
| **Language use** | Implicit (generated internally from visual features) | Explicit (natural language expressions) | None | None | None | Explicit (single NL description, no box annotations) |
| **Key challenge** | Static/unavailable text in VLT | Weak two-stage RBT performance | Context modeling without labels | Occlusion-induced cost confusion | Complexity & speed of video segmentation | Annotation cost for VL tracking |

---

### Thematic Clusters

**Vision-Language Tracking (Papers 1, 2, 6):** 
All three integrate natural language into tracking, but from different angles. Paper 1 (Lan) removes the need for explicit language at inference by generating it internally from vision. Paper 2 (Du/FlexHook) improves the *correspondence* between explicit language and multi-object trajectories in a two-stage pipeline. Paper 6 (Zheng/SVLTrack) eliminates bounding-box supervision while keeping language as the user-facing interface. Together they span: eliminating language input → improving language-visual matching → reducing annotation cost.

**Self-Supervised Tracking (Papers 3, 6):** 
Both PNTrack and SVLTrack come from the same group (Guangxi Normal / Southampton) and tackle annotation reduction. PNTrack focuses on visual-only self-supervised SOT with a curriculum contextual mechanism; SVLTrack extends the self-supervised setting to language-guided tracking. They are complementary: one removes visual labels, the other adds language guidance without adding annotation cost.

**Plug-and-Play / Training-Free Design (Papers 1, 4):** 
Both Lan et al. and OA-SORT target the existing tracker ecosystem as drop-in improvements. OA-SORT is purely training-free with no new learnable parameters. Paper 1's module only trains the newly introduced components. Both generalize across multiple base trackers, making them practically attractive.

**Architecture Simplification (Paper 5 — VidEoMT):** 
VidEoMT stands apart in motivation: rather than adding capability, it *removes* complexity. Its core claim — that large pre-trained ViTs subsume the tracking role — contrasts sharply with Papers 1–4, which all add modules on top of existing pipelines. VidEoMT's 5×–10× speedup makes it the most practically impactful paper for real-time applications.

**Multi-Object vs. Single-Object Tracking:** 
Papers 2 and 4 address MOT/RMOT (multiple objects, data association required). Papers 1, 3, 6 address SOT (single target, no data association). Paper 5 addresses Video Instance Segmentation, bridging segmentation and tracking of multiple instances simultaneously.


Beyond Explicit Language (Lan et al.) — SOT. 
Plug-and-play Textual Inversion Module converts visual features into CLIP text tokens, injecting implicit language guidance without needing any text input at inference.

FlexHook (Du et al.) — RMOT. 
Revives the two-stage Referring-by-Tracking paradigm via a C-Hook (sampling from backbone rather than re-encoding) and a Pairwise Correspondence Decoder replacing CLIP cosine similarity. First two-stage method to beat one-stage SOTA.

PNTrack (Zheng et al.) — Self-supervised SOT. 
Easy-to-hard curriculum: first injects instance patch prompts, then background noise tokens, to teach robust representations from unlabeled video. SOTA on 8 benchmarks.

OA-SORT (Li et al.) — MOT. 
Training-free, plug-and-play. Directly estimates per-object occlusion severity (via Gaussian Map) and uses it to de-confuse the cost matrix and stabilize the Kalman Filter.

VidEoMT (Norouzi et al.) — Video Instance Segmentation. 
Proves that a large DINOv2 ViT can track objects with just query propagation + fusion — no specialized tracker/ReID/adapter modules. 5×–10× faster than SOTA.

SVLTrack (Zheng et al.) — Self-supervised VL SOT. 
Uses LVLMs for pseudo-label generation, then a Dynamic Token Aggregation Module that selectively fuses key visual tokens with language tokens to track from a single language description without bounding-box annotations.

Key comparative insight: 
Papers 1, 2, 6 all tackle vision-language tracking from different angles (no text needed / better matching / no box labels). 
Papers 1 & 4 are both plug-and-play improvements to existing trackers. VidEoMT uniquely goes against the complexity trend by removing specialized modules.
