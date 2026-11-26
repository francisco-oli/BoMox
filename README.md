# BoMox — Foundation Model for Human Body Motion Representation from Multi-Modal Data

MSc thesis in Electrical and Computer Engineering, Instituto Superior Técnico (Universidade de Lisboa), November 2026.

**Author:** Francisco Casaleiro Oliveira
**Supervisors:** Prof. Alexandre Bernardino, Prof. José Santos-Victor

📄 **Full dissertation:** [Francisco_Oliveira_Dissertation.pdf](Francisco_Oliveira_Dissertation.pdf)

---

## Abstract

Human motion representation is a fundamental challenge in artificial intelligence, with applications in
healthcare, rehabilitation and robotics. However, capturing the semantic context of human movement remains
difficult. This thesis presents **BoMox**, a multimodal foundation model that aligns **3D human motion**,
**video**, and **natural language** through contrastive learning. Its motion branch combines frozen
[MOMENT](https://github.com/moment-timeseries-foundation-model/moment) features with a motion-specific
adaptation of STAMP, while its video and text branches adapt Long-CLIP. Motion is represented using
**SMPL-X**, capturing body pose, hand movements, and facial articulation. A dataset of **25,250
motion–video–text samples** is curated from Motion-X, with subsets reserved as held-out source domains.

BoMox is evaluated through cross-modal retrieval, action and posture classification, ablation experiments,
and an exploratory human-participant study. Although no action labels are used in training, a linear probe
on its motion embeddings raises top-1 accuracy on 487 actions from **7.3%** with the frozen time-series model
to **15.9%**. In a controlled five-way matching study against six participants, BoMox matches their mean
accuracy in motion–text matching (**70.0%**) and approaches it in motion–video matching (**80.0%** versus
91.7%). Retaining hand, jaw and eye information improves motion-to-video Recall@1 from **8.77% to 13.86%**.
Reconstruction provides little improvement to the shared representation, and performance declines on
held-out domains. Overall, this work demonstrates the benefits of multimodal supervision and expressive
articulation while highlighting limitations in representation regularization and domain generalization of
the current approach.

**Keywords:** Human Motion Representation, Multimodal Learning, Foundation Models, Contrastive Learning

---

## Architecture

BoMox has three parallel branches, one per modality, built on pretrained encoders with targeted adaptations,
and maps each input to an embedding in a shared latent space.

- **Motion branch.** SMPL-X parameters (178 channels per frame: global translation, root orientation, body
  and hand poses, jaw and eye rotations, body shape) are treated as a multivariate time series. Each channel
  is encoded by a frozen **MOMENT** time-series foundation model, and a **Modality-Aware STAMP** adapter
  models the dependencies between channels. It adds a modality embedding that distinguishes translation,
  rotation and shape channels.
- **Video and text branches.** Pretrained **Long-CLIP** encoders adapted to the motion domain with **LoRA**
  (rank 32, α = 64). Per-frame video features are pooled with multi-head attention pooling.
- **Objectives.**
  - A symmetric **InfoNCE contrastive loss** aligns all three modality pairs (motion↔video, motion↔text,
    video↔text).
  - An auxiliary **masked reconstruction** objective uses a transformer decoder to reconstruct a masked,
    patch-aligned block of the motion sequence from MOMENT's patch-level features and the sequence-level
    motion embedding.

## Dataset

The dataset is curated from seven **Motion-X** subsets: HAA500, HuMMan, IDEA400, Perform, Music, Animation
and Kungfu. Faulty samples were found through an LLM-based semantic categorization of the text labels and a
modality-specific analysis of the videos and motion annotations, and were then removed. Each remaining subset
was split independently into train, validation and test sets. Animation and Kungfu are held out entirely to
test generalization to unseen source domains.

| Partition  | Samples    |
|------------|-----------:|
| Training   | 18,993     |
| Validation | 2,371      |
| Test       | 2,373      |
| Held-out   | 1,513      |
| **Total**  | **25,250** |

## Research questions and main findings

| Question | Finding |
|---|---|
| **O1:** Does the model learn a shared space in which motion is aligned with video and language? | Yes. Test-set R@1 is 4.00% for text→motion, 15.47% for video→motion and 14.37% for text→video, against 8.36% for zero-shot Long-CLIP. BoMox matches human accuracy in five-way motion–text matching. |
| **O2:** Do the representations transfer to new tasks and generalize out of distribution? | They transfer: linear probes, k-NN and zero-shot classifiers beat frozen MOMENT on action, activity and posture tasks, although no labels are used in training. Out-of-distribution generalization is limited. Weight-space interpolation (WiSE-FT) with mean pooling recovers part of it. |
| **RQ1:** Does supervision from video and language add information to the motion representation? | Yes. The motion embedding recovers static-pose information that MOMENT's per-channel normalization removes, about half of what is available from video. |
| **RQ2:** Do the expressive SMPL-X parameters (hands, jaw, eyes) help? | Yes. Removing them lowers motion–video R@1 by more than 30% (relative). The loss is largest on fine-grained, hand-centred actions. |
| **RQ3:** Does masked reconstruction preserve kinematic information beyond contrastive alignment? | Only marginally. The decoder learns a useful motion prior, but embedding-swap tests show that it largely bypasses the sequence-level embedding. |

### Model variants

| Name | Description |
|---|---|
| **BoMox** | Complete model, trained with the contrastive and masked reconstruction objectives. |
| **BoMox-CL** | Contrastive objective only (λ<sub>rec</sub> = 0). |
| **BoMox-Body** | Hand, jaw, eye and body-shape channels set to zero (`BOMOX_DROP_CHANNELS`). |
| **BoMox-Raw** | Earlier version whose decoder regresses raw motion values without observed-frame statistics. |

---

## Repository contents

This repository contains the architecture and training code of the final BoMox model (internally "Model 10"),
together with the dissertation. Evaluation and analysis scripts are not included.

```
.
├── train.py                  # training loop: contrastive + reconstruction losses, validation, checkpointing
├── Dataset.py                # multimodal dataset, SMPL-X preprocessing, patch-aligned masking
├── data_split.py             # name-matched, per-subset train/val/test split and held-out subsets
├── Video_Preprocess.py       # decodes videos into Long-CLIP-preprocessed frame tensors (.pt)
├── Text_Encoder.py           # Long-CLIP text encoder with LoRA
├── Video_Encoder.py          # Long-CLIP vision encoder with LoRA and multi-head attention pooling
├── patched_attention.py      # splits CLIP's fused attention into q/k/v/out layers so LoRA can adapt them
├── Motion_Encoder.py         # frozen MOMENT backbone and Modality-Aware STAMP adapter
├── Motion_Reconstruction.py  # transformer decoder for masked-block reconstruction
├── Motion_Branch.py          # motion branch: encoder + decoder, observed-frame statistics
└── Francisco_Oliveira_Dissertation.pdf
```

## Contact

Francisco Oliveira · francisco.casaleiro.oliveira@tecnico.ulisboa.pt
