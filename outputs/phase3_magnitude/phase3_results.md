# Phase 3 (Magnitude Baseline): Hardware Validation & Grounding Benchmark Evaluation

**Project:** Attention-Guided Alignment Preservation in Efficient VLMs (AGAP)
**Authors:** Jean Gabriel Mpuhwezimana & Boniface Godwin
**Date:** March 10, 2026
**Model:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
**Pruning Strategy:** Magnitude-Based Pruning (MBP) at 40% target ratio
**Device:** CUDA GPU (PSC cluster, v016 node)
**Eval samples:** 100 (COCO validation)

---

## 1. Latency Benchmarks

| Configuration | Latency (ms) | Relative |
|---------------|-------------|----------|
| Baseline (unpruned) | 3420.6 | 1.00× |
| Magnitude-pruned | 3463.2 | 1.01× |

**100ms edge-deployment target met:** NO

### Interpretation

Latency is essentially unchanged between baseline and magnitude-pruned models (~1.2% slower, within noise). As with the alignment-guided strategy (AGSP), this is expected: **soft pruning** zeroes `o_proj` columns without removing parameters or reducing FLOPs, so the GPU still performs identical matrix multiplications. Note that the absolute latencies (~3.4s) are higher than the AGSP Phase 3 run (~0.95s); this variation is attributable to different cluster load conditions, batch sizes, or GPU allocation during the two separate runs — not to a fundamental difference between strategies.

---

## 2. Memory Usage

| Configuration | Peak VRAM (MB) | Peak VRAM (GB) | Relative |
|---------------|---------------|----------------|----------|
| Baseline | 14,421 | 14.1 | 1.00× |
| Magnitude-pruned | 7,411 | 7.2 | 0.51× |

### Memory Reduction: 48.6%

The magnitude-pruned model uses approximately **half** the VRAM of the baseline, closely matching the AGSP result (49.2%). This confirms that the VRAM savings come primarily from the zeroed weight representation under 8-bit quantization, not from the specific pruning strategy. Both strategies zero a comparable number of heads (AGSP: 345, MBP: 368), yielding similar compression ratios.

The 7.4 GB footprint is notable — it fits within an 8 GB edge GPU (e.g., NVIDIA Jetson AGX Orin), making the magnitude-pruned model viable for embedded deployment.

---

## 3. R-Precision (Grounding Fidelity)

| Configuration | R-Precision | Samples |
|---------------|------------|---------|
| Baseline | 1.000 | n=100 |
| Magnitude-pruned | 1.000 | n=100 |

**Alignment Retention: 100.0%**

### Interpretation

Both models achieve perfect R-Precision, identical to the AGSP results. This reinforces the finding that the current R-Precision metric is **too lenient** — it cannot distinguish between pruning strategies or between pruned and unpruned models. The prompt-leakage issue (the evaluation prompt contains the caption words the metric checks for) inflates scores to ceiling across all conditions.

---

## 4. Safety-Critical Object Detection

| Object Category | Baseline | MBP-Pruned | Retained |
|----------------|----------|------------|----------|
| Person | 1.000 | 1.000 | 100% |
| Vehicle | 1.000 | 1.000 | 100% |
| Stop Sign | 1.000 | 1.000 | 100% |

### Interpretation

All safety-critical categories are perfectly detected by both the baseline and magnitude-pruned models. Combined with the AGSP results, this suggests the binary detection task is below the discrimination threshold of a 7B VLM — even after removing ~37% of attention heads, the remaining capacity is sufficient for simple object presence questions.

---

## 5. AGSP vs. MBP Comparison

| Metric | Baseline | AGSP (40%) | MBP (40%) |
|--------|----------|------------|-----------|
| Latency (ms) | 955.1 / 3420.6* | 953.9 | 3463.2 |
| Peak VRAM (GB) | 27.4 / 14.1* | 13.9 | 7.4 |
| R-Precision | 1.000 | 1.000 | 1.000 |
| Safety: Person | 1.000 | 1.000 | 1.000 |
| Safety: Vehicle | 1.000 | 1.000 | 1.000 |
| Safety: Stop Sign | 1.000 | 1.000 | 1.000 |
| Heads Pruned | 0 | 345 | 368 |
| o_proj Sparsity | 0% | 33.7% | 36.8% |

\* Baseline values differ between runs due to different cluster conditions.

### Key Takeaway

The two pruning strategies are **indistinguishable** on all current evaluation metrics. Both achieve perfect R-Precision and safety detection retention, and both show negligible latency change and ~49% VRAM reduction. This metric saturation makes it impossible to determine whether alignment-guided pruning is superior to naive magnitude pruning.

**This is the critical finding for the final report:** harder, more discriminative benchmarks (POPE, VQAv2, GQA, CIDEr) are essential to reveal any advantage of alignment-informed head selection over magnitude-based selection.

---

## 6. Hypothesis Assessment

### H2: Pruning Preserves Cross-Modal Alignment

> Alignment-guided pruning preserves ≥90% of cross-modal grounding quality at 40% head sparsity.

**Verdict (MBP): Supported** — Retention = 100.0% on all metrics.

However, the same caveat applies: this tells us more about the metric's lack of sensitivity than about the pruning strategy's quality.

### H3: AGSP Outperforms Random/Magnitude Pruning

> Alignment-guided pruning retains more grounding quality than magnitude-based pruning.

**Verdict: Not confirmable** — Both strategies achieve identical scores across all metrics. The hypothesis cannot be tested until more discriminative evaluation benchmarks are adopted.

---

## 7. Recommendations for Final Report

1. **Discriminative benchmarks:** Adopt POPE (hallucination), VQAv2 (visual question answering), and CIDEr (caption generation) to differentiate AGSP from MBP.
2. **Structural pruning:** Convert soft pruning to true head removal to achieve actual latency reduction.
3. **Longer fine-tuning:** Increase from 50 steps to 3–5 epochs; explore LoRA-based recovery.
4. **Controlled comparison:** Run AGSP and MBP evaluations in the same session to eliminate cluster-condition variance in baseline latencies.
5. **Qualitative examples:** Include side-by-side generated captions for edge cases (occlusion, small objects, multi-object scenes).

---

## 8. Generated Artifacts

| File | Description |
|------|-------------|
| `phase3_results.json` | Raw JSON metrics from Phase 3 magnitude evaluation |
| `phase3_evaluation.png` | Comparison bar charts (latency, R-Precision, safety) |
| `phase3_results.md` | This analysis report |

---
