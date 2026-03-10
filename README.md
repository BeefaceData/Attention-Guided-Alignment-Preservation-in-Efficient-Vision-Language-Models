# Attention-Guided Alignment Preservation in Efficient VLMs (AGAP)
## CMU 18-662 PRINCIPLES of Deep Learning – Mid-Term Project

**Authors:** Boniface Godwin (`bgodwin@andrew.cmu.edu`), Jean Gabriel Mpuhwezimana (`jmpuhwez@andrew.cmu.edu`), Mohamed Awud (`mawud@andrew.cmu.edu`} 

---

## Overview

AGAP proposes a gradient-based **Alignment Importance Score (AIS)** to identify which attention heads in LLaVA-1.5-7B are critical for cross-modal alignment, then selectively prunes low-importance heads while preserving grounding fidelity. The project has three phases:

1. **Phase 1** — Compute per-head AIS via gradient norms on COCO calibration data
2. **Phase 2** — Apply alignment-guided structured pruning + RefCOCO fine-tuning
3. **Phase 3** — Evaluate latency, VRAM, R-Precision, and safety-critical detection

### Key Results (Mid-Term)

| Metric | Baseline | AGSP (40%) | MBP (40%) |
|--------|----------|------------|----------|
| Peak VRAM (GB) | 27.4 | 13.9 (−49.2%) | 7.4 (−48.6%) |
| Latency (ms) | 955.1 | 953.9 | 3463.2* |
| R-Precision | 1.000 | 1.000 | 1.000 |
| Safety Detection | 1.000 | 1.000 | 1.000 |
| Heads Pruned | 0 | 345 | 368 |
| o_proj Sparsity | 0% | 33.7% | 36.8% |

\* MBP baseline/latency numbers come from a separate cluster run under different load conditions.

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

Key packages: `transformers`, `torch`, `bitsandbytes`, `accelerate`, `datasets`, `requests`, `Pillow`, `scikit-learn`, `evaluate`

### Platform
All experiments were run on the **Pittsburgh Supercomputing Center (PSC)** with NVIDIA GPUs. The `--load_in_8bit` flag is recommended to keep VRAM under 16 GB.

---

## Usage

### Phase 1 — Baseline & Alignment Scoring
```bash
python phase1_baseline.py \
    --model_name llava-hf/llava-1.5-7b-hf \
    --num_samples 20 \
    --output_dir ./outputs/phase1 \
    --load_in_8bit
```
**Outputs:**
- `outputs/phase1/head_alignment_scores.json` — AIS for all 1,024 heads (32 layers × 32 heads)
- `outputs/phase1/alignment_score_summary.png` — score distribution plot
- `outputs/phase1/attn_sample_*.png` — cross-attention heatmaps

### Phase 2 — Pruning & Fine-Tuning

**Alignment-guided pruning (AGSP):**
```bash
python phase2_pruning.py \
    --scores_path ./outputs/phase1/head_alignment_scores.json \
    --prune_ratio 0.40 \
    --finetune_steps 200 \
    --output_dir ./outputs/phase2 \
    --load_in_8bit
```

**Magnitude-based pruning baseline (MBP):**
```bash
python phase2_pruning.py \
    --scores_path ./outputs/phase1/head_alignment_scores.json \
    --prune_ratio 0.40 \
    --finetune_steps 0 \
    --output_dir ./outputs/phase2 \
    --load_in_8bit \
    --baseline_comparison
```

**Outputs:**
- `outputs/phase2/pruned_model_alignment/` — AGSP pruned model weights
- `outputs/phase2/pruned_model_magnitude/` — MBP pruned model weights
- `outputs/phase2/phase2_results.json` — latency, sparsity, loss curve

### Phase 3 — Evaluation

**Evaluate AGSP-pruned model:**
```bash
python phase3_eval.py \
    --model_path ./outputs/phase2/pruned_model_alignment \
    --baseline_path llava-hf/llava-1.5-7b-hf \
    --num_eval_samples 100 \
    --output_dir ./outputs/phase3 \
    --load_in_8bit
```

**Evaluate MBP-pruned model:**
```bash
python phase3_eval.py \
    --model_path ./outputs/phase2/pruned_model_magnitude \
    --baseline_path llava-hf/llava-1.5-7b-hf \
    --num_eval_samples 100 \
    --output_dir ./outputs/phase3_magnitude \
    --load_in_8bit
```

**Outputs:**
- `outputs/phase3/phase3_results.json` — AGSP evaluation metrics
- `outputs/phase3/phase3_evaluation.png` — AGSP summary figure
- `outputs/phase3_magnitude/phase3_results.json` — MBP evaluation metrics
- `outputs/phase3_magnitude/phase3_evaluation.png` — MBP summary figure

---

## Project Structure
```
├── phase1_baseline.py          # AIS computation + attention heatmaps
├── phase2_pruning.py           # Alignment-guided & magnitude pruning + fine-tuning
├── phase3_eval.py              # Latency/memory/R-Precision/safety eval
├── requirements.txt
├── README.md
├── midterm_report_final_v3.docx # Mid-term report with AGSP + MBP results
└── outputs/
    ├── phase1/                 # AIS scores, heatmaps, summary plot
    ├── phase2/                 # Pruned models (alignment + magnitude), results JSON
    ├── phase3/                 # AGSP evaluation results, figure
    └── phase3_magnitude/       # MBP evaluation results, figure
```

## Compute Requirements
| Phase | Min GPU VRAM | Est. Time | Platform |
|-------|-------------|-----------|----------|
| Phase 1 (20 samples) | 16 GB (8-bit) | ~20 min | PSC GPU cluster |
| Phase 2 (200 steps) | 16 GB (8-bit) | ~1 hour | PSC GPU cluster |
| Phase 3 (100 samples) | 16 GB (8-bit) | ~30 min | PSC GPU cluster |

---

## Running the Full Experiment (End-to-End)

The commands below reproduce the complete AGAP experiment pipeline on a CUDA-capable machine.
All commands assume `--load_in_8bit` to stay within 16 GB VRAM.

```bash
# --- Phase 1: Compute Alignment Importance Scores ---
python phase1_baseline.py \
    --model_name llava-hf/llava-1.5-7b-hf \
    --num_samples 20 \
    --output_dir ./outputs/phase1 \
    --load_in_8bit

# --- Phase 2a: Alignment-Guided Structured Pruning (AGSP) ---
python phase2_pruning.py \
    --scores_path ./outputs/phase1/head_alignment_scores.json \
    --prune_ratio 0.40 \
    --finetune_steps 200 \
    --output_dir ./outputs/phase2 \
    --load_in_8bit

# --- Phase 2b: Magnitude-Based Pruning Baseline (MBP) ---
python phase2_pruning.py \
    --scores_path ./outputs/phase1/head_alignment_scores.json \
    --prune_ratio 0.40 \
    --finetune_steps 0 \
    --output_dir ./outputs/phase2 \
    --load_in_8bit \
    --baseline_comparison

# --- Phase 3a: Evaluate AGSP-Pruned Model ---
python phase3_eval.py \
    --model_path ./outputs/phase2/pruned_model_alignment \
    --baseline_path llava-hf/llava-1.5-7b-hf \
    --num_eval_samples 100 \
    --output_dir ./outputs/phase3 \
    --load_in_8bit

# --- Phase 3b: Evaluate MBP-Pruned Model ---
python phase3_eval.py \
    --model_path ./outputs/phase2/pruned_model_magnitude \
    --baseline_path llava-hf/llava-1.5-7b-hf \
    --num_eval_samples 100 \
    --output_dir ./outputs/phase3_magnitude \
    --load_in_8bit
```

---

## Ablation Grid (Optional)
Run multiple pruning ratios for the report ablation table:

```bash
for ratio in 0.20 0.30 0.40 0.50 0.60; do
    python phase2_pruning.py \
        --scores_path ./outputs/phase1/head_alignment_scores.json \
        --prune_ratio $ratio \
        --finetune_steps 200 \
        --output_dir ./outputs/ablation_$ratio \
        --load_in_8bit
done
```



