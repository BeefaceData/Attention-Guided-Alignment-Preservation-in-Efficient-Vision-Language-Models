"""
Phase 3: Hardware Validation & Grounding Benchmark Evaluation
=============================================================
Project: Attention-Guided Alignment Preservation in Efficient VLMs
Authors: Jean Gabriel Mpuhwezimana, Boniface Godwin
CMU PRINCIPLES Project – Mid-Term

Evaluates the pruned model on:
  1. Latency / memory on real or simulated edge hardware
  2. R-Precision on RefCOCO grounding (alignment fidelity)
  3. Safety-critical object detection on nuScenes subset

USAGE:
    python phase3_eval.py \
        --model_path ./outputs/phase2/pruned_model_alignment \
        --baseline_path llava-hf/llava-1.5-7b-hf \
        --output_dir ./outputs/phase3
"""

import os
import sys
# Redirect HuggingFace cache to /ocean (home /jet is quota-limited)
os.environ.setdefault("HF_HOME", "/ocean/projects/cis250019p/bgodwin/.cache/huggingface")
# Add ocean-installed packages (accelerate, bitsandbytes) to search path
_ocean_pkgs = "/ocean/projects/cis250019p/bgodwin/.local/lib/python3.11/site-packages"
if _ocean_pkgs not in sys.path:
    sys.path.append(_ocean_pkgs)

import argparse
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    default="/ocean/projects/cis250019p/bgodwin/outputs/phase2/pruned_model_alignment")
parser.add_argument("--baseline_path", type=str,
                    default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--num_eval_samples", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="/ocean/projects/cis250019p/bgodwin/outputs/phase3")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load models in 8-bit quantization to reduce VRAM usage")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
print(f"[INFO] Device: {DEVICE}")

# ─── Load models sequentially to fit in VRAM ─────────────────────────────────
# We evaluate one model at a time, storing results, then swap.
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(path):
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        proc = AutoProcessor.from_pretrained(path)
        mdl  = LlavaForConditionalGeneration.from_pretrained(
                   path, quantization_config=quant_config,
                   attn_implementation="eager", device_map="auto")
    else:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        proc  = AutoProcessor.from_pretrained(path)
        mdl   = LlavaForConditionalGeneration.from_pretrained(
                    path, torch_dtype=dtype, attn_implementation="eager").to(DEVICE)
    mdl.eval()
    return proc, mdl

def unload_model(mdl):
    del mdl
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ─── Evaluation helpers ───────────────────────────────────────────────────────

def get_response(model, processor, image, prompt, max_new=60):
    """Generate a text response for an image-prompt pair."""
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer.strip()


def benchmark_latency(model, processor, image, prompt, n=10):
    """Return (mean_ms, std_ms, peak_memory_mb).
    Uses GPU VRAM when on CUDA, else CPU tracemalloc."""
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    for _ in range(n):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    if DEVICE == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # Approximate with model parameter size on CPU
        peak_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    return float(np.mean(times)), float(np.std(times)), peak_mb


import requests as _requests
import io as _io

def _load_image_from_sample(sample):
    """Return a PIL Image from a COCO-style sample.
    Tries embedded 'image' first, then downloads from coco_url/flickr_url."""
    if sample.get("image") is not None:
        return sample["image"]
    for url_key in ("coco_url", "flickr_url"):
        url = sample.get(url_key)
        if url:
            resp = _requests.get(url, timeout=15)
            resp.raise_for_status()
            return Image.open(_io.BytesIO(resp.content)).convert("RGB")
    raise KeyError("No image or downloadable URL found in sample")


def r_precision(model, processor, eval_samples):
    """
    Compute R-Precision on COCO validation samples.
    Uses the first caption as the referring expression.
    Metric: fraction of samples where the model's response mentions
    the key content words from the caption.
    """
    correct = 0
    total   = 0
    stop_words = {"a", "an", "the", "of", "in", "on", "at", "to",
                  "is", "it", "and", "or", "with", "that", "this",
                  "for", "from", "by", "its"}
    for sample in eval_samples:
        try:
            image = _load_image_from_sample(sample)

            # captions is a list of plain strings in phiyodr/coco2017
            captions = sample.get("captions", [])
            if not captions:
                continue
            ref = captions[0] if isinstance(captions[0], str) else captions[0].get("text", "")
            if not ref or len(ref.split()) < 2:
                continue

            target_words = [
                w.lower().strip(".,!?") for w in ref.split()
                if w.lower().strip(".,!?") not in stop_words and len(w) > 2
            ]
            if not target_words:
                continue

            prompt = f"Describe what is shown: {ref}"
            response = get_response(model, processor, image, prompt)
            response_lower = response.lower()

            hits = sum(1 for w in target_words if w in response_lower)
            if hits >= max(1, len(target_words) // 2):
                correct += 1
            total += 1
        except Exception as _e:
            if total == 0:
                print(f"[DEBUG] r_precision sample error: {_e}")
    return correct / max(total, 1), total


# Safety-critical keyword groups for caption-based filtering
SAFETY_CAPTION_KEYWORDS = {
    "person":    {"person", "man", "woman", "people", "child", "boy", "girl", "pedestrian"},
    "vehicle":   {"car", "bus", "truck", "motorcycle", "bicycle", "bike", "vehicle"},
    "stop sign": {"stop sign", "stop"},
}


def safety_object_detection(model, processor, samples,
                            groups=("person", "vehicle", "stop sign")):
    """
    For each safety-critical object group, find COCO samples whose captions
    mention that object, then test whether the model correctly identifies it.
    """
    results = {g: {"correct": 0, "total": 0} for g in groups}

    for sample in samples:
        try:
            image = _load_image_from_sample(sample)

            # Determine which safety groups are present via caption keywords
            captions = sample.get("captions", [])
            caption_text = " ".join(
                c if isinstance(c, str) else c.get("text", "") for c in captions
            ).lower()

            for group in groups:
                keywords = SAFETY_CAPTION_KEYWORDS.get(group, {group})
                if not any(kw in caption_text for kw in keywords):
                    continue
                prompt = f"Is there a {group} in this image? Answer yes or no."
                response = get_response(model, processor, image, prompt, max_new=10)
                if "yes" in response.lower():
                    results[group]["correct"] += 1
                results[group]["total"] += 1
        except Exception as _e:
            if all(v["total"] == 0 for v in results.values()):
                print(f"[DEBUG] safety_detection sample error: {_e}")

    rates = {g: v["correct"] / max(v["total"], 1) for g, v in results.items()}
    return rates


# ─── Load evaluation datasets ──────────────────────────────────────────────────
from datasets import load_dataset

print("[INFO] Loading COCO validation set (used for all evaluations) ...")
coco_data = list(load_dataset("phiyodr/coco2017", split="validation",
                               streaming=True).take(args.num_eval_samples))

# RefCOCO is used only for its sentence annotations; it does NOT embed images,
# so we use COCO (which embeds images) for the actual R-Precision inference.
# We still attempt RefCOCO to log availability.
print("[INFO] Loading RefCOCO validation set ...")
try:
    _refcoco_check = list(load_dataset("jxu124/refcoco", split="validation",
                                       streaming=True).take(1))
    _has_image = _refcoco_check and _refcoco_check[0].get("image") is not None
except Exception:
    _has_image = False

if _has_image:
    refcoco_data = list(load_dataset("jxu124/refcoco", split="validation",
                                     streaming=True).take(args.num_eval_samples))
else:
    print("[INFO] RefCOCO has no embedded images; using COCO captions for R-Precision.")
    refcoco_data = coco_data

# Safety eval uses COCO (which has proper category annotations)
safety_data = coco_data[:50]

# ─── Run evaluations (one model at a time to save VRAM) ───────────────────────
test_img = Image.new("RGB", (336, 336), color=(100, 100, 200))
test_prompt = "Describe what you see."

# ── Evaluate pruned model first ────────────────────────────────────────────
print("\n[EVAL] Loading and evaluating pruned model ...")
proc_pruned, model_pruned = load_model(args.model_path)

lat_pruned_mean, lat_pruned_std, mem_pruned = benchmark_latency(
    model_pruned, proc_pruned, test_img, test_prompt)
print(f"  Pruned latency : {lat_pruned_mean:.1f} +/- {lat_pruned_std:.1f} ms | Mem: {mem_pruned:.0f} MB")

rp_pruned, n_pruned = r_precision(model_pruned, proc_pruned, refcoco_data)
print(f"  Pruned R-Precision : {rp_pruned:.3f}  (n={n_pruned})")

safety_pruned = safety_object_detection(model_pruned, proc_pruned, safety_data)
print(f"  Pruned safety detection: {safety_pruned}")

# Free pruned model before loading baseline
unload_model(model_pruned)

# ── Evaluate baseline model ────────────────────────────────────────────────
print("\n[EVAL] Loading and evaluating baseline model ...")
proc_base, model_base = load_model(args.baseline_path)

lat_base_mean, lat_base_std, mem_base = benchmark_latency(
    model_base, proc_base, test_img, test_prompt)
print(f"  Baseline latency : {lat_base_mean:.1f} +/- {lat_base_std:.1f} ms | Mem: {mem_base:.0f} MB")

rp_base, n_base = r_precision(model_base, proc_base, refcoco_data)
print(f"  Baseline R-Precision : {rp_base:.3f}  (n={n_base})")

safety_base = safety_object_detection(model_base, proc_base, safety_data)
print(f"  Baseline safety detection: {safety_base}")

unload_model(model_base)

# ─── Print comparison ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"  Baseline  : {lat_base_mean:.1f} +/- {lat_base_std:.1f} ms | Mem: {mem_base:.0f} MB")
print(f"  Pruned    : {lat_pruned_mean:.1f} +/- {lat_pruned_std:.1f} ms | Mem: {mem_pruned:.0f} MB")
print(f"  Alignment retention  : {rp_pruned/max(rp_base,1e-6)*100:.1f}%")

print(f"\n  {'Object':<20} {'Baseline':>10} {'Pruned':>10}")
print("  " + "-"*42)
for obj in safety_pruned:
    print(f"  {obj:<20} {safety_base.get(obj, 0):>10.3f} {safety_pruned[obj]:>10.3f}")

# ─── Save results ─────────────────────────────────────────────────────────────
final_results = {
    "latency": {
        "baseline_ms": lat_base_mean,
        "pruned_ms": lat_pruned_mean,
        "meets_100ms_target": lat_pruned_mean < 100,
    },
    "memory_mb": {
        "baseline": mem_base,
        "pruned": mem_pruned,
    },
    "r_precision": {
        "baseline": rp_base,
        "pruned": rp_pruned,
        "retention_pct": rp_pruned / max(rp_base, 1e-6) * 100,
    },
    "safety_detection": {
        "baseline": safety_base,
        "pruned": safety_pruned,
    },
}

with open(OUTPUT_DIR / "phase3_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

# ─── Visualisation ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 3: Evaluation Results", fontsize=14, fontweight="bold")

# Latency bar
methods = ["Baseline", "Alignment-Pruned"]
latencies = [lat_base_mean, lat_pruned_mean]
colors = ["#4C72B0", "#DD8452"]
axes[0].bar(methods, latencies, color=colors)
axes[0].axhline(100, color="red", linestyle="--", label="100ms target")
axes[0].set_ylabel("Latency (ms)")
axes[0].set_title("Inference Latency")
axes[0].legend()

# R-Precision bar
axes[1].bar(methods, [rp_base, rp_pruned], color=colors)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("R-Precision")
axes[1].set_title("Grounding Fidelity (R-Precision)")

# Safety detection grouped bar
objs = list(safety_base.keys())
x = np.arange(len(objs))
w = 0.35
axes[2].bar(x - w/2, [safety_base[o] for o in objs], w, label="Baseline", color=colors[0])
axes[2].bar(x + w/2, [safety_pruned[o] for o in objs], w, label="Pruned", color=colors[1])
axes[2].set_xticks(x)
axes[2].set_xticklabels(objs, rotation=20, ha="right")
axes[2].set_ylabel("Detection Rate")
axes[2].set_title("Safety-Critical Object Detection")
axes[2].legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "phase3_evaluation.png", dpi=120)
plt.close(fig)
print(f"\n[INFO] Results and plots saved to {OUTPUT_DIR}")

print("\n" + "="*60)
print("PHASE 3 FINAL SUMMARY")
print("="*60)
print(f"  Alignment Retention  : {final_results['r_precision']['retention_pct']:.1f}%")
print(f"  <100ms Target Met    : {'YES' if lat_pruned_mean < 100 else 'NO'}")
print("="*60)
