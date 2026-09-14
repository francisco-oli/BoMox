# BoMox — Multimodal Motion Representation Learning

A research project (ongoing MSc thesis, [Instituto Superior Técnico]) building a
foundation model for 3D human motion, aligning motion, video, and text into a
shared embedding space via contrastive learning, with an auxiliary
reconstruction objective for masked-motion completion.

This repository contains a **curated set of components** from the full
project — selected specifically to showcase debugging depth, experiment
design, and applied ML engineering. It is not a runnable end-to-end
pipeline (the full training pipeline depends on a large, license-restricted
dataset and a university compute cluster) — it's a technical portfolio of
the harder problems I solved while building it.

---

## What this project actually involved

- Built a multimodal contrastive architecture aligning a motion encoder
  (built on a frozen [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
  time-series foundation model) with LoRA-adapted CLIP video and text
  towers, using masked motion reconstruction as an auxiliary training
  objective.
- **Diagnosed and fixed a silent gradient-flow bug** in LoRA-adapted
  attention layers, caused by `nn.MultiheadAttention`'s fused CUDA kernel
  bypassing wrapped submodules — confirmed via direct gradient inspection,
  fixed with a custom attention module, and validated to be numerically
  identical to the original layer before any adapter training. → see
  [`Patched_Attention.py`](./Patched_Attention.py)
- **Root-caused a recurring NaN-loss failure** in mixed-precision training
  through systematic elimination — instrumenting internal library state,
  isolating train/eval-mode behavior with a controlled same-weights
  comparison, and ruling out three plausible explanations (masking edge
  cases, gradient explosion, temperature scaling) before finding the actual
  cause: a normalization epsilon underflow specific to low-variance input
  channels under `float16`.
- **Designed a causal attribution experiment** (embedding-swap ablation) to
  empirically isolate how much a learned representation actually
  contributes to downstream reconstruction quality, separating genuine
  signal from a confounding, always-available auxiliary input — rather
  than trusting an aggregate loss number at face value. → see
  [`noise_swap_check.py`](./noise_swap_check.py)
- Built a reproducible, **stratified data-splitting pipeline** across 7
  heterogeneous data sources (~26K samples), with dedicated held-out
  subsets reserved specifically for out-of-distribution generalization
  testing, and fixed a silent train/test leakage bug caused by duplicated
  split logic across scripts. → see [`data_split.py`](./data_split.py)
- Applied parameter-efficient fine-tuning (LoRA) to adapt frozen
  pretrained CLIP encoders to a specialized domain, informed by
  catastrophic-forgetting literature, with an explicit plan to measure
  general-domain degradation empirically rather than assume it away.
- Implemented and evaluated cross-modal retrieval (Recall@1/5/10) following
  standard evaluation methodology from the CLIP/vision-language literature.

---

<details>
<summary><b>Deep dive: the LoRA gradient bug (click to expand)</b></summary>

**Symptom:** LoRA adapters injected into a CLIP video encoder had
`requires_grad=True`, were correctly registered in the optimizer, and yet
their weights (`lora_B`) remained *exactly* at their zero-initialization
value after hundreds of training epochs.

**Investigation:** Checked `requires_grad` (True). Checked optimizer
parameter groups (present). Checked the actual gradient after a real
backward pass — `None`. Not small, not noisy: structurally absent.

**Root cause:** `torch.nn.MultiheadAttention.forward()` doesn't call its
submodules (`out_proj(x)`) directly — it extracts their raw
`.weight`/`.bias` tensors and passes them into a fused CUDA kernel
(`F.multi_head_attention_forward`) for performance. Any module-level
wrapper placed on `out_proj` (as PEFT's LoRA implementation does) is
therefore invisible to the actual forward computation — the parameter
exists, but the computation never runs through it.

**Fix:** Implemented `PatchedMultiheadAttention` — a drop-in replacement
exposing `q_proj`/`k_proj`/`v_proj`/`out_proj` as real, individually-called
`nn.Linear` submodules, so PEFT's module-replacement approach actually
works. Verified numerically identical to the original layer (max
difference `0.0` across matched inputs) before training, both with and
without a causal attention mask, so the fix introduced zero behavioral
change beyond making the adapters trainable.

</details>

<details>
<summary><b>Deep dive: the mixed-precision NaN chase (click to expand)</b></summary>

Training intermittently collapsed into full-batch NaN losses partway
through otherwise-healthy runs. Ruled out, in order, with direct evidence
at each step rather than assumption:
- Masked-batch edge cases (checked directly — zero degenerate batches)
- Gradient-clipping/exploding-gradient theory (added clipping — problem
  persisted)
- Learning-rate/temperature-scaling instability (added explicit scale
  clamping — problem persisted)
- A BatchNorm train/eval-mode statistics mismatch (tested directly by
  running the *same weights* on the *same data* in both modes — losses
  were nearly identical, ruling this out cleanly)

Root cause: `RevIN`'s per-instance normalization divided by
`(std + eps)`, and the default `eps` was too small for low-variance
channels common in short or static motion sequences — producing
values that overflowed under `float16`. Fixed by raising `eps` and adding
a defensive value clamp, with the fix moved into the model's own
constructor (not an external training-script patch) after an earlier
version of the fix was silently lost during a refactor — a lesson in
where safety-critical fixes should actually live.

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
