# BoMox — Multimodal Motion Representation Learning

## Summary

Ongoing MSc thesis project at [Instituto Superior Técnico] building a
foundation model for 3D human motion: a model that learns a single,
general-purpose motion embedding by aligning motion, video, and text into a
shared representation space, while also learning to complete masked
(missing) motion sequences.

This repository contains a **curated set of components** from the full
project, selected to showcase system design and the debugging
and experiment-design work that went into making it actually function
correctly. It is not a runnable end-to-end pipeline; the full system
depends on a large, license-restricted dataset

---

## What the project is

**Goal.** Learn a motion representation useful for multiple downstream
tasks (retrieval, classification, motion reconstruction)

**Architecture, at a high level:**
- A **motion branch**: 3D motion sequences are encoded with a frozen
  [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
  time-series foundation model, then pooled into a sequence-level
  representation by a custom attention-pooling transformer (built on
  [STAMP](https://github.com/example/stamp)).
- Two **CLIP-based branches** (video and text), each a frozen pretrained
  Long-CLIP encoder adapted to the motion domain via
  [LoRA](https://arxiv.org/abs/2106.09685) — chosen specifically to adapt
  to a narrow, specialized domain without catastrophically overwriting the
  general-purpose semantic knowledge from large-scale pretraining.
- A **contrastive objective** (InfoNCE) aligning all three modality pairs
  — motion↔video, motion↔text, and video↔text — into one shared embedding
  space.
- An **auxiliary reconstruction objective**: a transformer decoder
  reconstructs masked motion frames, conditioned on the learned motion
  representation via cross-attention. This exists to (a) empirically
  measure how much real motion information the representation retains,
  beyond what's minimally needed for contrastive discrimination, and
  (b) give the model a direct motion-reconstruction capability.


  ---

## Dataset and Data Curation

Training data is drawn from Motion-X++, a large-scale, multimodal (motion + video + text) human motion dataset comprising several distinct subdatasets.

**Data Curation**
- The dataset used consists of approximately 26K samples, each comprising one video, one variable-length motion sequence, and a sequence-level text label, drawn from 7 different Motion-X++ subsets spanning a wide range of motion semantics.
- An LLM-based semantic characterization of the dataset was performed to profile its content and evaluate semantic diversity, identifying broad motion categories including Sports and Fitness, Music and Instrument Playing, Domestic Chores, and Communication/Expressive Action.
- This same characterization pipeline surfaced anomalies in the text labels (e.g., "Sorry, I can’t provide information about
the person in the video."), informing further data curation, alongside separate detection and removal of malformed motion/video samples.
- Subsets were split in a stratified manner (independently per subset, then recombined) to ensure proportional representation across train/validation/test despite substantial size imbalance between sources.
- Two subsets (Animation, Kungfu) — chosen for being the most stylistically distinct from the rest and comprising a small percentage of the data — were held out entirely, reserved as a dedicated out-of-distribution generalization test set.
---

## Problems I encountered (and how I solved them)

Each of these is a real problem hit during development, presented as
symptom → investigation → root cause → fix, since that's usually the more
informative part.

<details>
<summary><b>1. LoRA adapters showed zero gradient, despite being correctly configured</b></summary>

**Symptom:** LoRA adapters injected into a CLIP video encoder had
`requires_grad=True`, were correctly registered in the optimizer, and yet
their weights (`lora_B`) remained *exactly* at their zero-initialization
value after several training epochs.

**Investigation:** Checked `requires_grad` (True). Checked optimizer
parameter groups (present, correctly registered). Checked the actual
gradient tensor after a real backward pass — `None`. Structurally absent from the computation
graph entirely.

**Root cause:** `torch.nn.MultiheadAttention.forward()` doesn't call its
submodules (e.g. `out_proj(x)`) directly — it extracts their raw
`.weight`/`.bias` tensors and passes them into a fused CUDA kernel
(`F.multi_head_attention_forward`) for performance. Any module-level
wrapper placed on `out_proj` (as PEFT's LoRA implementation does) is
therefore invisible to the actual forward computation — the parameter
exists in the model, but the computation graph never routes through it.

**Fix:** Implemented `PatchedMultiheadAttention` — a drop-in replacement
exposing `q_proj`/`k_proj`/`v_proj`/`out_proj` as real, individually-called
`nn.Linear` submodules, so PEFT's module-replacement approach actually
works. 

</details>

<details>
<summary><b>2. Training intermittently collapsed into NaN losses under mixed precision</b></summary>

**Symptom:** Training would proceed normally for a variable number of
steps, then abruptly produce NaN loss for one or more batches — sometimes
recovering, sometimes not.

**Investigation, hypotheses ruled out in order, each with direct evidence
rather than assumption:**
- Degenerate all-masked batches → checked directly, zero such batches present.
- Gradient explosion → added gradient clipping, problem persisted.
- Temperature/logit-scale instability → added explicit scale clamping,
  problem persisted.
- A `BatchNorm` train/eval-mode statistics mismatch → tested directly by
  running *identical weights on identical data* in both modes — losses
  were nearly the same, cleanly ruling this out instead of leaving it
  ambiguous.

**Root cause:** A normalization layer computed `(x - mean) / (stdev + eps)`
per input instance. Motion channels with near-zero real variance in their
visible frames (common in short or largely static sequences) drove `stdev`
close to zero, and the library's default `eps` was too small to prevent
the division from producing extreme values — which then overflowed under
`float16`.

**Fix:** Raised `eps` and added a defensive output clamp, with the fix
placed inside the model's own constructor rather than as an external
training-script patch — after an earlier version of the same fix was
silently lost during an unrelated code refactor. That was its own lesson:
safety-critical fixes need to live where they can't be refactored away.

</details>

<details>
<summary><b>3. The reconstruction decoder produced (nearly) the same output regardless of input</b></summary>

**Symptom:** The reconstruction decoder converged to outputting
essentially the same "average" motion pose for every input video,
regardless of what the video actually showed — measured directly via a
diversity metric (mean pairwise cosine similarity of outputs across a
batch), not just a visual impression.

**Investigation:** The learned motion representation itself was confirmed
*not* collapsed (its own diversity metric was healthy) — so the problem
was specifically in how the decoder used that representation, not in the
representation itself. Traced to an architectural bottleneck: the
representation was a single pooled vector, broadcast identically to every
output timestep, giving the decoder no per-timestep information to
differentiate its predictions — positional encoding alone was doing most
of the work.

**Fix, iterated in stages:**
1. Routed the frozen encoder's *unpooled*, per-timestep features directly
   into the decoder (bypassing the pooling bottleneck for this specific
   path), restoring real per-timestep signal.
2. Replaced a flat, uniform fusion of the pooled representation with the
   unpooled features (originally FiLM-style scale/shift conditioning) with
   genuine cross-attention, letting the decoder attend into the
   representation with content- and position-specific weights instead of
   one fixed modulation.
3. Validated the fix with a **causal attribution test**: reconstruct the
   same input twice, once with the correct representation and once with a
   mismatched one from a different sample (same patch-level features held
   fixed) — the gap between the two directly measures how much the
   representation, specifically, contributes. This confirmed the
   representation went from barely influencing reconstruction quality to
   being a substantial, measurable contributor.

</details>

<details> <summary><b>5. Rising validation contrastive loss looked like overfitting</b></summary>

Symptom: Validation contrastive loss rose ~18% over the course of training (while training loss fell over the same span) 
Investigation: First ruled out a BatchNorm train/eval mode mismatch as an alternative explanation, by directly comparing identical weights on identical data in both modes (losses were nearly equal — ruled out, see #2). This left genuine overfitting as the apparent remaining explanation. However, InfoNCE cross-entropy is scaled by the model's learned temperature parameter (logit_scale), which increases over training as a normal part of contrastive learning.

Resolution: Computed retrieval accuracy (Top-1/Top-5) on the same validation checkpoints — a ranking-based metric that is mathematically invariant to any monotonic rescaling of similarity scores, including temperature. Retrieval accuracy was flat-to-improving across the exact epochs where loss appeared to worsen, directly showing the rising loss was a scale artifact, not model degradation. Retrieval-based metrics were adopted as the primary validation signal going forward, with loss retained only as a secondary diagnostic.

</details>


</details>

---

## Tech / concepts applied

`PyTorch` · `torch.compile` · Mixed-precision training (`autocast`/`GradScaler`)
· Contrastive learning (InfoNCE) · Parameter-efficient fine-tuning (LoRA / PEFT)
· Transformer architectures (custom attention, cross-attention fusion)
· Multimodal representation learning · CLIP-style vision-language models
· `wandb` experiment tracking · Systematic ML debugging & ablation design

---

## Status

Active thesis project, expected completion **November/2026**. Full results and final architecture will be added on
completion; this repository will be updated accordingly.

## Contact

Francisco Oliveira · francisco.casaleiro.oliveira@tecnico.ulisboa.pt
