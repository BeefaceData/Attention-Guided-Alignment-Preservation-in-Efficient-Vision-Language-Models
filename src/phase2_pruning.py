"""
Phase 2: Alignment-Guided Structured Pruning Algorithm
=======================================================
Project: Attention-Guided Alignment Preservation in Efficient VLMs
Authors: Jean Gabriel Mpuhwezimana, Boniface Godwin
CMU PRINCIPLES Project – Mid-Term

Reads head alignment scores from Phase 1, applies structured pruning at
configurable sparsity ratios, and fine-tunes on RefCOCO.

SETUP:
    pip install transformers torch torchvision pillow datasets evaluate \
                scikit-learn --break-system-packages

USAGE:
    # Load scores from Phase 1 and prune at 40%
    python phase2_pruning.py \
        --scores_path ./outputs/phase1/head_alignment_scores.json \
        --prune_ratio 0.40 \
        --finetune_steps 500 \
        --output_dir ./outputs/phase2
"""

import os
import sys
# Redirect HuggingFace cache to /ocean (home /jet is quota-limited)
os.environ.setdefault("HF_HOME", "/ocean/projects/cis250019p/bgodwin/.cache/huggingface")
# Add ocean-installed packages (e.g. bitsandbytes) to search path.
# Append (not prepend) so the conda env's packages take precedence,
# preventing the ocean numpy from shadowing the conda env's numpy.
_ocean_pkgs = "/ocean/projects/cis250019p/bgodwin/.local/lib/python3.11/site-packages"
if _ocean_pkgs not in sys.path:
    sys.path.append(_ocean_pkgs)

import argparse
import json
import time
import io
import torch
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--scores_path", type=str,
                    default="./outputs/phase1/head_alignment_scores.json")
parser.add_argument("--prune_ratio", type=float, default=0.40,
                    help="Fraction of heads to prune (0.40 = 40%%)")
parser.add_argument("--baseline_comparison", action="store_true",
                    help="Also run magnitude-based pruning for comparison")
parser.add_argument("--finetune_steps", type=int, default=200,
                    help="Fine-tune steps on RefCOCO after pruning")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--output_dir", type=str, default="/ocean/projects/cis250019p/bgodwin/outputs/phase2")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit quantization to reduce RAM from ~28 GB to ~8 GB")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = args.device
print(f"[INFO] Device: {DEVICE}")

# ─── Load model ───────────────────────────────────────────────────────────────
print("[INFO] Loading model ...")
processor = AutoProcessor.from_pretrained(args.model_name)

def _load_model(model_name, device, load_in_8bit=False):
    """Load LLaVA model, optionally in 8-bit quantization."""
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print("[INFO] Using 8-bit quantization (bitsandbytes)")
        m = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            attn_implementation="eager",
            device_map="auto",
        )
    else:
        m = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="eager",
        ).to(device)
    return m

model = _load_model(args.model_name, DEVICE, args.load_in_8bit)

# ─── Load Phase 1 alignment scores ────────────────────────────────────────────
print(f"[INFO] Loading alignment scores from {args.scores_path}")
with open(args.scores_path) as f:
    scores = json.load(f)  # list of {layer, head, alignment_score}

scores.sort(key=lambda x: x["alignment_score"])  # ascending → bottom = lowest score

total_heads = len(scores)
n_prune = int(total_heads * args.prune_ratio)
heads_to_prune_alignment = {(s["layer"], s["head"]) for s in scores[:n_prune]}
heads_to_keep_alignment  = {(s["layer"], s["head"]) for s in scores[n_prune:]}

print(f"[INFO] Total heads: {total_heads}, pruning {n_prune} ({args.prune_ratio*100:.0f}%)")

# ─── Parameter counting helper ────────────────────────────────────────────────
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def count_active_params(m, pruned_heads: set):
    """Count non-zeroed parameters (approximation)."""
    total = 0
    for name, param in m.named_parameters():
        total += param.numel()
    return total   # actual zero-masking computed below

# ─── Pruning strategies ───────────────────────────────────────────────────────

class HeadPruner:
    """
    Applies structured head pruning by zeroing out the output projection
    weights of selected attention heads ("soft pruning").  This is an
    in-place operation.  Because the weight matrices are not physically
    removed, the FLOPs/latency stay roughly constant; the benefit is
    measured through quality degradation analysis and sparsity metrics.
    Full latency gains require weight-matrix restructuring (future work).
    """

    def __init__(self, model, num_heads_per_layer: int, head_dim: int,
                 max_prune_per_layer: float = 0.5):
        self.model = model
        self.num_heads = num_heads_per_layer
        self.head_dim = head_dim
        self.max_prune_per_layer = max_prune_per_layer
        self.pruned_head_mask = {}   # {layer_idx: set of head indices zeroed}

    def _get_o_proj(self, layer_idx):
        """Return the output projection weight tensor for a given LLM layer."""
        # LLaVA's language model is a LlamaForCausalLM → .model.layers[i].self_attn.o_proj
        return self.model.model.language_model.layers[layer_idx].self_attn.o_proj.weight

    def prune_heads(self, heads_to_prune: set, method: str = "alignment"):
        """
        Zero out head output projections.
        heads_to_prune: set of (layer_idx, head_idx) tuples.
        Respects max_prune_per_layer to avoid removing all heads from one layer.
        """
        # Enforce per-layer cap
        layer_counts = {}
        for (l, h) in heads_to_prune:
            layer_counts.setdefault(l, []).append(h)
        max_per_layer = int(self.num_heads * self.max_prune_per_layer)

        pruned_count = 0
        for (layer_idx, head_idx) in heads_to_prune:
            already = len(self.pruned_head_mask.get(layer_idx, set()))
            if already >= max_per_layer:
                continue  # skip – would remove too many heads from this layer
            try:
                o_proj = self._get_o_proj(layer_idx)
                # o_proj weight shape: [hidden_size, num_heads*head_dim]
                # Each head contributes head_dim *columns* (in-features)
                start = head_idx * self.head_dim
                end   = start + self.head_dim
                with torch.no_grad():
                    # Cast to float32 for the assignment — handles Int8 (8-bit quant) weights
                    if o_proj.dtype == torch.int8:
                        o_proj[:, start:end] = torch.zeros(
                            o_proj.shape[0], end - start, dtype=torch.int8,
                            device=o_proj.device)
                    else:
                        o_proj[:, start:end] = 0.0
                self.pruned_head_mask.setdefault(layer_idx, set()).add(head_idx)
                pruned_count += 1
            except (IndexError, AttributeError):
                pass   # layer out of range – skip silently
        print(f"[PRUNE] Zeroed {pruned_count} heads ({method} method).")
        return pruned_count

    def magnitude_prune(self, prune_ratio: float):
        """
        Baseline: prune heads with smallest L2-norm output projections.
        """
        norms = []
        try:
            num_layers = len(self.model.model.language_model.layers)
        except AttributeError:
            print("[WARN] Cannot access LLM layers for magnitude pruning.")
            return
        for layer_idx in range(num_layers):
            try:
                o_proj = self._get_o_proj(layer_idx)
                for head_idx in range(self.num_heads):
                    start = head_idx * self.head_dim
                    end   = start + self.head_dim
                    # Cast to float32 first — 8-bit quantized weights are Int8
                    norm = o_proj[:, start:end].to(torch.float32).norm().item()
                    norms.append((layer_idx, head_idx, norm))
            except (IndexError, AttributeError):
                pass

        norms.sort(key=lambda x: x[2])  # ascending
        n_prune = int(len(norms) * prune_ratio)
        heads_to_prune = {(l, h) for l, h, _ in norms[:n_prune]}
        self.prune_heads(heads_to_prune, method="magnitude")
        return heads_to_prune


# ─── Measure baseline performance ─────────────────────────────────────────────

def measure_latency(model, processor, image, prompt, n_trials=5):
    """Return mean inference latency in ms (includes 1 warmup run)."""
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

    # Warmup run (JIT / CUDA kernel compilation)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    times = []
    for _ in range(n_trials):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ─── Fine-tuning on RefCOCO ───────────────────────────────────────────────────

def finetune_refcoco(model, processor, steps: int, lr: float,
                     pruned_head_mask: dict = None, head_dim: int = 128):
    """
    Light-weight causal-LM fine-tuning of the pruned model on RefCOCO.
    Gradient masking prevents pruned heads from recovering nonzero weights.
    """
    print(f"[FINETUNE] Loading RefCOCO dataset ...")
    # Use the referring expression split from HuggingFace
    try:
        refcoco = load_dataset("jxu124/refcoco", split="train", streaming=True)
        dataset = list(refcoco.take(steps * 4))  # generous buffer
    except Exception as e:
        print(f"[WARN] Could not load RefCOCO: {e}. Using COCO as fallback.")
        dataset = list(load_dataset("phiyodr/coco2017", split="train",
                                    streaming=True).take(steps * 4))

    # Freeze all parameters; only fine-tune the top N LLM layers + lm_head.
    # A 7B model has ~7B params; training all of them needs ~56 GB of fp32
    # optimizer state — far more than a 32 GB GPU can hold.  Training only
    # the top 4 layers (~150 M params) keeps optimizer state under 1.2 GB.
    #
    # LLaVA path: LlavaForConditionalGeneration
    #               .model (LlavaModel)
    #                 .language_model (LlamaForCausalLM)
    #                   .model (LlamaModel)
    #                     .layers (ModuleList of LlamaDecoderLayer)
    #                   .lm_head (Linear)
    FINETUNE_LAYERS = 4
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        # LLaVA: LlavaForConditionalGeneration
        #          .model (LlavaModel)
        #            .language_model (LlamaModel — layers are directly here)
        llm = model.model.language_model
        llm_layers = llm.layers
        num_llm_layers = len(llm_layers)
        for layer in llm_layers[num_llm_layers - FINETUNE_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        # lm_head lives on the top-level model, not on the LlamaModel
        for attr in ("lm_head",):
            head = getattr(model, attr, None) or getattr(llm, attr, None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(True)
                break
    except AttributeError as e:
        print(f"[WARN] Could not find LLM layers ({e}); fine-tuning all params.")
        for p in model.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[FINETUNE] Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # Enable gradient checkpointing to reduce activation memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[FINETUNE] Gradient checkpointing enabled.")

    # Free any cached allocator memory before creating optimizer states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 8-bit AdamW stores optimizer moments in 8-bit → ~8x less memory than fp32 Adam.
    # Falls back to standard AdamW (fine here since trainable params are now small).
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr)
        print("[FINETUNE] Using 8-bit AdamW (bitsandbytes).")
    except Exception:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        print("[FINETUNE] Using standard AdamW.")

    model.train()

    loss_history = []
    sample_idx   = 0

    for step in range(steps):
        if sample_idx >= len(dataset):
            sample_idx = 0
        sample = dataset[sample_idx]
        sample_idx += 1

        # Extract image — RefCOCO doesn't embed images; download from COCO URL
        image = sample.get("image")
        if image is None:
            try:
                import requests, io
                raw_info = sample.get("raw_image_info", "")
                if isinstance(raw_info, str):
                    import json as _json
                    raw_info = _json.loads(raw_info)
                img_url = raw_info.get("flickr_url") or raw_info.get("coco_url", "")
                if not img_url:
                    continue
                resp = requests.get(img_url, timeout=15)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                continue

        # Build a referring expression prompt
        captions = sample.get("captions", [])
        if isinstance(captions, list) and len(captions) > 0:
            ref_text = captions[0]
        else:
            sentences = sample.get("sentences", [{"raw": "Describe this."}])
            if isinstance(sentences, list) and len(sentences) > 0:
                ref_text = sentences[0].get("raw", "Describe this image.")
            else:
                ref_text = "Describe this image."

        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"Locate: {ref_text}"}
        ]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)

        try:
            inputs = processor(images=image, text=text,
                               return_tensors="pt").to(DEVICE)
        except Exception:
            continue

        # Forward pass; labels = input_ids for causal-LM recovery fine-tuning
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()

        # Gradient masking: zero out o_proj gradients for pruned head columns
        # so their weights remain zero throughout fine-tuning.
        if pruned_head_mask:
            for name, param in model.named_parameters():
                if param.grad is not None and "o_proj" in name:
                    # Extract layer index from parameter name
                    try:
                        layer_idx = int(name.split("layers.")[1].split(".")[0])
                    except (IndexError, ValueError):
                        continue
                    if layer_idx in pruned_head_mask:
                        for h in pruned_head_mask[layer_idx]:
                            start = h * head_dim
                            end   = start + head_dim
                            param.grad[:, start:end] = 0.0

        optimizer.step()
        loss_history.append(loss.item())

        if (step + 1) % 50 == 0:
            avg = np.mean(loss_history[-50:])
            print(f"  Step {step+1}/{steps} | Loss: {avg:.4f}")

    model.eval()
    print(f"[FINETUNE] Done. Final avg loss: {np.mean(loss_history[-20:]):.4f}")
    return loss_history


# ─── Sparsity measurement ─────────────────────────────────────────────────────

def compute_sparsity(model):
    """Return fraction of zero parameters in attention o_proj layers."""
    total, zeros = 0, 0
    for name, param in model.named_parameters():
        if "o_proj" in name:
            total += param.numel()
            zeros += (param.data == 0).sum().item()
    return zeros / max(total, 1)


# ─── Main experiment ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from PIL import Image

    # Use a simple test image for latency benchmarking
    test_img = Image.new("RGB", (336, 336), color=(128, 64, 32))
    prompt   = "What objects are in this image?"

    # ── Baseline (before pruning) ──────────────────────────────────────────
    print("\n[EXPERIMENT] Measuring baseline latency ...")
    params_before = count_params(model)
    lat_mean, lat_std = measure_latency(model, processor, test_img, prompt)
    print(f"  Params (baseline)   : {params_before:,}")
    print(f"  Latency (baseline)  : {lat_mean:.1f} ± {lat_std:.1f} ms")

    results = {
        "prune_ratio": args.prune_ratio,
        "baseline": {"params": params_before, "latency_ms": lat_mean}
    }

    # ── Detect model head config ───────────────────────────────────────────
    try:
        llm_config = model.config.text_config
        num_heads  = llm_config.num_attention_heads
        head_dim   = llm_config.hidden_size // num_heads
    except AttributeError:
        num_heads, head_dim = 32, 128   # LLaMA-7B defaults

    # ── Alignment-guided pruning ────────────────────────────────────────────
    # NOTE: We prune in-place to avoid copy.deepcopy OOM on the 7B model.
    # The baseline latency has already been recorded above.
    print(f"\n[EXPERIMENT] Applying alignment-guided pruning ({args.prune_ratio*100:.0f}%) ...")
    pruner_align = HeadPruner(model, num_heads, head_dim)
    pruner_align.prune_heads(heads_to_prune_alignment, method="alignment")

    lat_a_mean, lat_a_std = measure_latency(model, processor, test_img, prompt)
    sparsity = compute_sparsity(model)

    print(f"  Latency (alignment-pruned) : {lat_a_mean:.1f} ± {lat_a_std:.1f} ms")
    print(f"  o_proj sparsity            : {sparsity*100:.1f}%")
    print(f"  NOTE: Soft pruning (zero-masking) preserves matrix shapes;")
    print(f"        latency reduction requires weight restructuring (future work).")

    results["alignment_pruned"] = {
        "params": params_before,
        "latency_ms": lat_a_mean,
        "o_proj_sparsity_pct": sparsity * 100,
        "heads_removed": n_prune,
    }

    # ── Magnitude-based pruning (baseline comparison) ───────────────────────
    if args.baseline_comparison:
        print(f"\n[EXPERIMENT] Applying magnitude-based pruning ({args.prune_ratio*100:.0f}%) ...")
        # Reload a fresh model for fair comparison
        print("  Reloading fresh model for magnitude comparison ...")
        model_magnitude = _load_model(args.model_name, DEVICE, args.load_in_8bit)
        pruner_mag = HeadPruner(model_magnitude, num_heads, head_dim)
        pruner_mag.magnitude_prune(args.prune_ratio)

        lat_m_mean, lat_m_std = measure_latency(model_magnitude, processor, test_img, prompt)
        sparsity_mag = compute_sparsity(model_magnitude)
        print(f"  Latency (magnitude-pruned) : {lat_m_mean:.1f} ± {lat_m_std:.1f} ms")
        print(f"  o_proj sparsity            : {sparsity_mag*100:.1f}%")
        results["magnitude_pruned"] = {
            "latency_ms": lat_m_mean,
            "o_proj_sparsity_pct": sparsity_mag * 100,
        }

         # Save magnitude-pruned model for Phase 3 comparison
        mag_save_dir = OUTPUT_DIR / "pruned_model_magnitude"
        model_magnitude.save_pretrained(mag_save_dir)
        processor.save_pretrained(mag_save_dir)
        print(f"  Magnitude-pruned model saved to {mag_save_dir}")
        # Free the magnitude model
        del model_magnitude
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    if args.finetune_steps > 0:
        print(f"\n[EXPERIMENT] Fine-tuning pruned model for {args.finetune_steps} steps ...")
        loss_hist = finetune_refcoco(
            model, processor, args.finetune_steps, args.learning_rate,
            pruned_head_mask=pruner_align.pruned_head_mask, head_dim=head_dim,
        )
        results["finetune_loss_curve"] = loss_hist

        # Re-measure latency after fine-tuning
        lat_ft_mean, _ = measure_latency(model, processor, test_img, prompt)
        results["finetuned"] = {"latency_ms": lat_ft_mean}
        print(f"  Latency (fine-tuned) : {lat_ft_mean:.1f} ms")

    # ── Save model and results ──────────────────────────────────────────────
    results_path = OUTPUT_DIR / "phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved → {results_path}")

    model_path = OUTPUT_DIR / "pruned_model_alignment"
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    print(f"[INFO] Pruned model saved → {model_path}")

    # ── Print final table ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PHASE 2 SUMMARY  (soft pruning – zero-masking)")
    print("="*65)
    print(f"{'Method':<30} {'Latency (ms)':<16} {'Sparsity'}")
    print("-"*65)
    print(f"{'Baseline (uncompressed)':<30} {lat_mean:<16.1f} 0.0%")
    print(f"{'Alignment-Guided Pruning':<30} {lat_a_mean:<16.1f} {sparsity*100:.1f}%")
    if args.baseline_comparison:
        print(f"{'Magnitude-Based Pruning':<30} {lat_m_mean:<16.1f} {sparsity_mag*100:.1f}%")
    print("\nNOTE: Latency is similar because soft pruning preserves matrix")
    print("      dimensions.  Actual speedup requires weight restructuring.")
    print("="*65)
