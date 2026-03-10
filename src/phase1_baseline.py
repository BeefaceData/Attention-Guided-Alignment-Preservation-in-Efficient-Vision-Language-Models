"""
Phase 1: Baseline Analysis & Cross-Attention Map Extraction
===========================================================
Project: Attention-Guided Alignment Preservation in Efficient VLMs
Authors: Jean Gabriel Mpuhwezimana, Boniface Godwin
CMU PRINCIPLES Project – Mid-Term

SETUP (run once):
    pip install transformers torch torchvision pillow datasets matplotlib seaborn --break-system-packages

USAGE:
    python phase1_baseline.py --num_samples 50 --output_dir ./outputs/phase1
"""

import os
# Redirect HuggingFace cache to /ocean (home /jet is quota-limited)
os.environ.setdefault("HF_HOME", "/ocean/projects/cis250019p/bgodwin/.cache/huggingface")

import argparse
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import requests
import io

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Phase 1: VLM Baseline & Attention Analysis")
parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
                    help="HuggingFace model ID (use a smaller model for testing)")
parser.add_argument("--num_samples", type=int, default=20,
                    help="Number of COCO samples to analyse")
parser.add_argument("--output_dir", type=str, default="./outputs/phase1")
parser.add_argument("--device", type=str, default="auto",
                    help="cuda | cpu | auto")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Device setup ─────────────────────────────────────────────────────────────
if args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = args.device
print(f"[INFO] Using device: {DEVICE}")

# ─── Load model ───────────────────────────────────────────────────────────────
print(f"[INFO] Loading model: {args.model_name}")
from transformers import AutoProcessor, LlavaForConditionalGeneration

processor = AutoProcessor.from_pretrained(args.model_name)
model = LlavaForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    output_attentions=True,          # <-- enables attention output
    attn_implementation="eager",     # required for attention map access
).to(DEVICE)
model.eval()

# ─── Load COCO validation subset ──────────────────────────────────────────────
print("[INFO] Loading COCO dataset ...")
from datasets import load_dataset
coco = load_dataset("phiyodr/coco2017", split="validation", streaming=True)
samples = list(coco.take(args.num_samples))
print(f"[INFO] Loaded {len(samples)} samples.")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_inference_for_ais(image: Image.Image, caption: str):
    """
    Run a forward pass WITH gradient tracking to compute the
    gradient-based Alignment Importance Score (AIS).

    AIS(l, h) = || dL_proxy / dW_Q(l,h) ||_F

    We use causal LM cross-entropy on the ground-truth caption as a
    proxy for alignment loss.  Heads whose Q projections are most
    sensitive to this loss are deemed alignment-critical.
    """
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": caption},
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

    # Enable grad on W_Q parameters so we can compute AIS
    for name, param in model.named_parameters():
        param.requires_grad_(True if "q_proj" in name else False)

    model.zero_grad()

    # Forward pass (no_grad=False to allow backward)
    outputs = model(**inputs, output_attentions=True)

    # Build contrastive loss using the last-token hidden state as embedding proxy.
    # We use the logits to compute a cross-entropy against the target (first token
    # of the correct caption). This approximates dL_alignment / dW_Q without
    # requiring a separate CLIP tower.
    logits = outputs.logits  # [1, seq_len, vocab_size]
    target_ids = inputs["input_ids"][:, 1:]  # shift right
    pred_logits = logits[:, :-1, :]           # [1, seq_len-1, vocab]

    # Cross-entropy as proxy for alignment loss
    loss = torch.nn.functional.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=processor.tokenizer.pad_token_id or 0
    )
    loss.backward()

    return outputs, inputs

def compute_ais_per_head(model):
    """
    After backward(), read accumulated gradients on all q_proj weights.
    AIS(l, h) = || grad of W_Q(l,h) ||_F

    LLaVA-1.5 uses LlamaAttention with a single q_proj [hidden, heads*head_dim].
    We reshape to [num_heads, head_dim, hidden] to get per-head slices.
    """
    scores = {}
    llm_config = model.config.text_config
    num_heads = llm_config.num_attention_heads   # 32 for LLaMA-7B
    head_dim  = llm_config.hidden_size // num_heads  # 128

    for layer_idx, layer in enumerate(model.model.language_model.layers):
        attn = layer.self_attn
        q_proj = attn.q_proj  # nn.Linear: weight shape [num_heads*head_dim, hidden]

        if q_proj.weight.grad is None:
            scores[layer_idx] = {h: 0.0 for h in range(num_heads)}
            continue

        grad = q_proj.weight.grad.detach()  # [num_heads*head_dim, hidden]
        # Reshape to [num_heads, head_dim, hidden]
        grad_per_head = grad.view(num_heads, head_dim, -1)
        scores[layer_idx] = {}
        for h in range(num_heads):
            scores[layer_idx][h] = float(torch.norm(grad_per_head[h], p="fro").cpu())

    model.zero_grad()
    return scores

def extract_cross_attention_map(outputs, inputs):
    """
    Visualisation helper: compute mean cross-modal attention map
    (text tokens → visual tokens) averaged over last 4 layers.
    Returns a 2-D array [num_text_tokens, num_visual_tokens].
    """
    all_attn = outputs.attentions
    layers_to_use = all_attn[-4:]
    stacked = torch.stack([a.squeeze(0).detach() for a in layers_to_use])  # [4, H, S, S]
    mean_attn = stacked.mean(dim=(0, 1)).cpu().float().numpy()              # [S, S]

    num_visual = getattr(model.config, "image_seq_length", 576)

    # Locate where visual tokens sit in the expanded sequence.
    # The processor inserts one <image> placeholder; the model expands it
    # to num_visual patch embeddings at that position.
    image_token_id = getattr(model.config, "image_token_index", 32000)
    input_ids = inputs["input_ids"][0].cpu().tolist()
    if image_token_id in input_ids:
        visual_start = input_ids.index(image_token_id)
    else:
        visual_start = 1  # conservative fallback
    visual_end = visual_start + num_visual
    text_start = visual_end

    cross_attn = mean_attn[text_start:, visual_start:visual_end]
    return cross_attn, num_visual

def visualize_attention(cross_attn, sample_id, caption):
    """Save a heat-map of the cross-modal attention map."""
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(cross_attn[:20, :50], ax=ax, cmap="viridis",
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Sample {sample_id} | Cross-Attention (text→visual)\n{caption[:80]}")
    ax.set_xlabel("Visual Token Index")
    ax.set_ylabel("Text Token Index")
    fig.tight_layout()
    save_path = OUTPUT_DIR / f"attn_sample_{sample_id}.png"
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return str(save_path)

# ─── Main loop ────────────────────────────────────────────────────────────────
all_head_scores = {}      # accumulated AIS scores across samples
latency_records = []      # inference times

for idx, sample in enumerate(samples):
    print(f"[{idx+1}/{len(samples)}] Processing sample ...")

    # phiyodr/coco2017 stores URLs, not raw images
    img_url = sample.get("coco_url", "")
    try:
        resp = requests.get(img_url, timeout=15)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [WARN] Could not download image for sample {idx}: {e}")
        continue
    captions = sample.get("captions", ["N/A"])
    caption = captions[0] if isinstance(captions, list) and captions else "N/A"

    t0 = time.perf_counter()
    try:
        outputs, inputs = run_inference_for_ais(image, caption)
    except Exception as e:
        print(f"  [WARN] Sample {idx} failed: {e}")
        continue
    latency = (time.perf_counter() - t0) * 1000  # ms
    latency_records.append(latency)

    # Free intermediate tensors between samples
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Cross-attention map for visualisation
    cross_attn, num_visual = extract_cross_attention_map(outputs, inputs)

    if idx < 5:
        visualize_attention(cross_attn, idx, caption)

    # Gradient-based AIS per head
    head_scores = compute_ais_per_head(model)
    for layer_idx, heads in head_scores.items():
        if layer_idx not in all_head_scores:
            all_head_scores[layer_idx] = {}
        for head_idx, score in heads.items():
            all_head_scores[layer_idx].setdefault(head_idx, []).append(score)

# ─── Aggregate & report ───────────────────────────────────────────────────────
# Average AIS per head across calibration samples
avg_scores = {
    layer: {head: float(np.mean(vals)) for head, vals in heads.items()}
    for layer, heads in all_head_scores.items()
}

# Flatten to list for ranking
flat_scores = [
    {"layer": l, "head": h, "alignment_score": s}
    for l, heads in avg_scores.items()
    for h, s in heads.items()
]
flat_scores.sort(key=lambda x: x["alignment_score"], reverse=True)

# Save scores
scores_path = OUTPUT_DIR / "head_alignment_scores.json"
with open(scores_path, "w") as f:
    json.dump(flat_scores, f, indent=2)
print(f"[INFO] Saved alignment scores → {scores_path}")

# ─── Summary statistics ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 SUMMARY")
print("="*60)
print(f"  Samples processed : {len(latency_records)}")
print(f"  Avg. latency      : {np.mean(latency_records):.1f} ms")
print(f"  Std. latency      : {np.std(latency_records):.1f} ms")
print(f"  Top-5 alignment heads:")
for entry in flat_scores[:5]:
    print(f"    Layer {entry['layer']:2d} Head {entry['head']:2d}  →  score={entry['alignment_score']:.4f}")
print(f"\n  Results written to: {OUTPUT_DIR}")

# ─── Plot: Distribution of alignment scores ───────────────────────────────────
all_score_vals = [e["alignment_score"] for e in flat_scores]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_score_vals, bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Alignment Scores (All Heads)")
axes[0].set_xlabel("Alignment Score")
axes[0].set_ylabel("Count")

# Plot top-20 heads
top20 = flat_scores[:20]
labels = [f"L{e['layer']}H{e['head']}" for e in top20]
vals   = [e["alignment_score"] for e in top20]
axes[1].barh(labels[::-1], vals[::-1], color="darkorange")
axes[1].set_title("Top-20 Alignment-Critical Heads")
axes[1].set_xlabel("Alignment Score")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "alignment_score_summary.png", dpi=120)
plt.close(fig)
print("[INFO] Saved summary plot.")
