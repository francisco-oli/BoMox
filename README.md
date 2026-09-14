# BoMox — Multimodal Motion Representation Learning

## Summary

An ongoing MSc thesis project ([Instituto Superior Técnico]) building a
foundation model for 3D human motion: a model that learns a single,
general-purpose motion embedding by aligning motion, video, and text into a
shared representation space, while also learning to complete masked
(missing) motion sequences.

This repository contains a **curated set of components** from the full
project, selected to showcase system design and — mainly — the debugging
and experiment-design work that went into making it actually function
correctly. It is not a runnable end-to-end pipeline; the full system
depends on a large, license-restricted dataset and a university compute
cluster. What's here is the technical substance, not the plumbing.

---

## What the project is

**Goal.** Learn a motion representation useful for multiple downstream
tasks (retrieval, classification, motion completion/forecasting) — not
just one, which is what distinguishes a *foundation* model from a
single-task one.

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
  (b) give the model a direct motion-completion/inpainting capability.

**Evaluation methodology:**
- Cross-modal retrieval (Recall@1/5/10), following standard practice from
  the CLIP / vision-language retrieval literature.
- A dedicated held-out data split (two full data subsets, never seen
  during training or validation) specifically for testing generalization
  to novel motion domains.
- A causal-attribution test (embedding-swap ablation) isolating how much
  of the model's reconstruction quality is actually attributable to the
  learned representation, as opposed to other available signal.

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
value after hundreds of training epochs — not slowly moving, not noisy,
completely static.

**Investigation:** Checked `requires_grad` (True). Checked optimizer
parameter groups (present, correctly registered). Checked the actual
gradient tensor after a real backward pass — `None`. Not small, not
starved by other loss terms: structurally absent from the computation
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
works. Verified numerically identical to the original layer (max
difference `0.0` across matched inputs, both with and without a causal
attention mask) before any adapter training — confirming the fix changed
*trainability*, not *behavior*. → [`Patched_Attention.py`](./Patched_Attention.py)

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
   being a substantial, measurable contributor. →
   [`noise_swap_check.py`](./noise_swap_check.py)

</details>

<details>
<summary><b>4. A silent train/test split mismatch between two scripts</b></summary>

**Symptom:** A held-out validation/test split was correctly designed
(stratified across multiple data sources, with dedicated
never-trained-on subsets for generalization testing) — but a *second*
script, used for qualitative inspection of model outputs, independently
re-implemented the same splitting logic instead of reusing it.

**Root cause:** The second implementation differed in three ways that
weren't obvious individually — it included data sources meant to be fully
held out, it aligned samples via non-deterministic set ordering rather
than a fixed sort, and its Python dictionary/set iteration order wasn't
guaranteed stable across runs. The two scripts' "test sets" had silently
diverged.

**Fix:** Factored the entire splitting procedure into a single shared
module (`data_split.py`), imported identically by every script that needs
it, with a fixed seed and deterministic sorting before any random
shuffling. This eliminates the possibility of two scripts disagreeing
about what "test data" means, by construction rather than by convention.
→ [`data_split.py`](./data_split.py)

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

Active thesis project, expected completion **[Month/Year]**, supervised by
**[Advisor Name]**. Full results and final architecture will be added on
completion; this repository will be updated accordingly.

## Contact

[Your name] · [email] · [LinkedIn] · [other links]
