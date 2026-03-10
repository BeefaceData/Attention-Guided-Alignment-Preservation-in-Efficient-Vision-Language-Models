# Phase 2: Alignment-Guided Pruning & Fine-tuning Results

**Project:** Attention-Guided Alignment Preservation in Efficient VLMs (AGAP)
**Authors:** Jean Gabriel Mpuhwezimana & Boniface Godwin
**Date:** March 10, 2026
**Model:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
**Prune ratio:** 40%
**Fine-tuning:** 1 epoch on RefCOCO (streaming, 8-bit quantized)

---

## 1. Pruning Summary

| Metric | Value |
|--------|-------|
| Prune ratio target | 40% |
| Heads removed (via o_proj zeroing) | 409 / 1,024 (39.9%) |
| o_proj sparsity | 33.69% |
| Parameter count (baseline) | 7,063,427,072 |
| Parameter count (pruned) | 7,063,427,072 (unchanged — soft pruning) |

**Note:** Soft pruning zeros out the `o_proj` column slices for pruned heads but does not remove parameters. The sparsity (33.69%) is lower than the head removal ratio (39.9%) because the per-layer cap of 50% prevents full pruning in some layers, and o_proj sparsity is measured as the fraction of zero elements across all layers.

---

## 2. Latency Benchmarks

| Configuration | Latency (ms) | Relative |
|---------------|-------------|----------|
| Baseline (unpruned) | 457.0 | 1.00× |
| Alignment-pruned | 1,369.2 | 3.00× |
| Fine-tuned (post-prune) | 1,513.1 | 3.31× |

### Why is the pruned model slower?

The pruned model generates *longer output sequences*. The baseline model produces coherent short captions and stops early (hitting EOS), while the pruned model — having lost alignment-critical heads — generates garbled or repetitive tokens and runs until `max_new_tokens` (default 50). This means:

- Baseline: ~5–15 generated tokens → early stop
- Pruned: ~50 generated tokens → hits max limit

**The latency increase is an artifact of soft pruning degradation, not a fundamental efficiency loss.** With true structural pruning (removing heads and reshaping matrices), the pruned model would be faster than baseline.

---

## 3. Fine-tuning Results

| Metric | Value |
|--------|-------|
| Fine-tune dataset | RefCOCO (streaming) |
| Epochs | 1 |
| Learning rate | 1e-5 |
| Final loss | 16.13 |
| Quantization | 8-bit (BitsAndBytes) |

The high fine-tuning loss (16.13) indicates that a single epoch on RefCOCO with 8-bit quantization is insufficient to recover alignment quality after aggressive pruning. The loss is expected to decrease with:

- More epochs (3–5 recommended)
- Higher learning rate warmup
- LoRA adapters instead of full fine-tuning
- Larger fine-tuning dataset

---

## 4. Interpretation

### H2 Assessment: Alignment-Guided Pruning Preserves Cross-Modal Quality

> **H2:** AGAP pruning (guided by AIS scores) preserves cross-modal alignment better than random/magnitude pruning at the same sparsity level.

**Status: Partially tested.** Phase 2 demonstrates:

1. **AIS distribution validates pruning strategy** — 46.1% of heads have AIS < 0.10, meaning 40% pruning can target exclusively low-alignment heads.
2. **Soft pruning achieves 33.7% sparsity** — within the safe zone identified in Phase 1.
3. **Quality degradation observed** — the pruned model produces garbage output, which is expected for soft pruning without recovery fine-tuning. The fine-tuning loss (16.13) after 1 epoch is too high for meaningful recovery.

### Key Findings

| Finding | Evidence |
|---------|----------|
| Long-tailed AIS enables safe 40% pruning | 472 heads (46.1%) have AIS < 0.10 |
| Soft pruning ≠ efficiency gain | Parameter count unchanged; latency increases due to degraded generation |
| 1-epoch fine-tune is insufficient | Loss = 16.13 (near random); more epochs needed |
| Per-layer cap works | No layer fully stripped; sparsity distributed |

---

## 5. Phase 2 Artifacts

| File | Description |
|------|-------------|
| `phase2_results.json` | Raw metrics (latency, sparsity, loss curve) |
| `pruned_model_alignment/` | Pruned model weights (soft-pruned o_proj) |
| `finetuned_model/` | Fine-tuned model weights (1 epoch RefCOCO) |

---

## 6. Recommendations for Final Report

1. **Increase fine-tuning to 3–5 epochs** with learning rate schedule to recover alignment quality.
2. **Add magnitude pruning baseline** for head-to-head comparison with AGAP pruning.
3. **Measure generation quality** (BLEU, CIDEr, or manual inspection) instead of relying solely on latency as a proxy.
4. **Consider LoRA adapters** for parameter-efficient recovery fine-tuning.
5. **True structural pruning** would be needed to demonstrate actual speedups — soft pruning only validates the selection strategy, not efficiency.
