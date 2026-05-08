"""
Quantized KV Cache for LLM Inference
=====================================
Designed for PARAM Rudra (IIT Patna) — NVIDIA A100 80GB HBM2e
Falls back to CPU simulation automatically if no GPU is available.

What this code demonstrates:
  1. BF16 (baseline) KV cache — full precision, memory-heavy
  2. INT8 KV cache — 2x memory reduction, dynamic per-token scaling
  3. FP8 KV cache — 4x memory reduction (simulated, A100 supports via torch)

Each step is profiled: memory used, bandwidth achieved, latency.

Run on PARAM Rudra:
  sbatch run_kv_quant.sh

Run locally (CPU simulation):
  python kv_cache_quantized.py
"""

import torch
import time
import math
import json
import sys
import os
import argparse

# ──────────────────────────────────────────────
# 1. DEVICE DETECTION
# ──────────────────────────────────────────────
def setup_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        info = {
            "device": "cuda",
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "sm_count": props.multi_processor_count,
            "cuda_version": torch.version.cuda,
        }
    else:
        dev = torch.device("cpu")
        info = {
            "device": "cpu",
            "name": "CPU (simulation mode — no GPU detected)",
            "total_memory_gb": "N/A",
            "sm_count": "N/A",
            "cuda_version": "N/A",
        }
    return dev, info


# ──────────────────────────────────────────────
# 2. MODEL CONFIG  (GPT-2 small scale for demo)
# ──────────────────────────────────────────────
class ModelConfig:
    """
    Matches a small transformer decoder.
    Scale up n_layers / n_heads / head_dim for larger models.
    A100 80GB can handle Llama-3-8B at full precision.
    """
    def __init__(self, preset="small"):
        presets = {
            "small": dict(n_layers=12, n_heads=12, head_dim=64,  vocab=50257),
            "medium": dict(n_layers=24, n_heads=16, head_dim=64,  vocab=50257),
            "large":  dict(n_layers=32, n_heads=32, head_dim=128, vocab=32000),  # ~Llama-7B shape
        }
        p = presets[preset]
        self.n_layers   = p["n_layers"]
        self.n_heads    = p["n_heads"]
        self.head_dim   = p["head_dim"]
        self.d_model    = self.n_heads * self.head_dim
        self.vocab_size = p["vocab"]

    def kv_bytes(self, seq_len: int, dtype_bytes: int, batch: int = 1) -> int:
        """Bytes for full KV cache (K + V, all layers, all heads)."""
        # K: [batch, n_layers, n_heads, seq_len, head_dim]
        # V: same shape
        per_tensor = batch * self.n_layers * self.n_heads * seq_len * self.head_dim
        return 2 * per_tensor * dtype_bytes   # K + V


# ──────────────────────────────────────────────
# 3. KV CACHE IMPLEMENTATIONS
# ──────────────────────────────────────────────

class KVCacheBF16:
    """
    Baseline: full BF16 precision.
    Shape: [n_layers, 2, batch, n_heads, seq_len, head_dim]
              layer  K/V  batch  heads   seq      dim
    This is exactly what PyTorch / HuggingFace use by default.
    """
    def __init__(self, config: ModelConfig, max_seq: int, batch: int, device):
        self.config  = config
        self.device  = device
        self.dtype   = torch.bfloat16
        # Allocate K and V tensors for all layers at once
        self.K = torch.zeros(
            config.n_layers, batch, config.n_heads, max_seq, config.head_dim,
            dtype=self.dtype, device=device)
        self.V = torch.zeros_like(self.K)

    def write(self, layer: int, pos: int, k: torch.Tensor, v: torch.Tensor):
        """Write one token's K/V into position `pos` in layer `layer`."""
        self.K[layer, :, :, pos, :] = k.to(self.dtype)
        self.V[layer, :, :, pos, :] = v.to(self.dtype)

    def read(self, layer: int, pos: int) -> tuple:
        """Read all K/V up to position `pos`."""
        return self.K[layer, :, :, :pos, :], self.V[layer, :, :, :pos, :]

    def memory_bytes(self) -> int:
        return (self.K.nelement() + self.V.nelement()) * 2  # BF16 = 2 bytes


class KVCacheINT8:
    """
    INT8 quantized KV cache with dynamic per-token scaling.

    Algorithm:
      WRITE:
        1. For each new K/V vector (one token, one layer):
           scale_k = max(|k|) / 127          ← per-token absmax scaling
           k_int8  = round(k / scale_k)       ← quantize to [-127, 127]
           store k_int8 and scale_k

      READ (dequantize):
        2. k_fp = k_int8.float() * scale_k   ← reconstruct BF16 for attention

    Why per-token (not per-tensor)?
      KV activations have outliers concentrated in specific tokens.
      Per-tensor scaling clips these outliers → big accuracy loss.
      Per-token scaling captures each token's dynamic range separately.

    Memory saving: BF16 is 2 bytes, INT8 is 1 byte → 2x reduction.
    The scale adds (seq_len × n_heads × n_layers × 4 bytes) overhead,
    which is tiny compared to the KV tensor itself.
    """
    def __init__(self, config: ModelConfig, max_seq: int, batch: int, device):
        self.config = config
        self.device = device
        # INT8 storage for K and V
        self.K_q = torch.zeros(
            config.n_layers, batch, config.n_heads, max_seq, config.head_dim,
            dtype=torch.int8, device=device)
        self.V_q = torch.zeros_like(self.K_q)
        # Per-token scale factors (float32, one per token per head per layer)
        self.K_scale = torch.zeros(
            config.n_layers, batch, config.n_heads, max_seq,
            dtype=torch.float32, device=device)
        self.V_scale = torch.zeros_like(self.K_scale)

    def _quantize(self, x: torch.Tensor):
        """
        Per-token dynamic INT8 quantization.
        x shape: [batch, n_heads, head_dim]
        Returns: (int8_tensor, scale_per_head)
        """
        # Compute scale per (batch, head): max absolute value / 127
        # keepdim so we can broadcast over head_dim
        scale = x.abs().max(dim=-1, keepdim=True).values / 127.0
        scale = scale.clamp(min=1e-8)          # avoid division by zero
        x_q   = (x / scale).round().clamp(-127, 127).to(torch.int8)
        return x_q, scale.squeeze(-1)           # scale shape: [batch, n_heads]

    def write(self, layer: int, pos: int, k: torch.Tensor, v: torch.Tensor):
        k_f = k.float()
        v_f = v.float()
        k_q, k_s = self._quantize(k_f)
        v_q, v_s = self._quantize(v_f)
        self.K_q[layer, :, :, pos, :] = k_q
        self.V_q[layer, :, :, pos, :] = v_q
        self.K_scale[layer, :, :, pos] = k_s
        self.V_scale[layer, :, :, pos] = v_s

    def read(self, layer: int, pos: int):
        """Dequantize on read — reconstruct BF16 tensors for attention."""
        k_q = self.K_q[layer, :, :, :pos, :].float()          # [B, H, S, D]
        v_q = self.V_q[layer, :, :, :pos, :].float()
        k_s = self.K_scale[layer, :, :, :pos].unsqueeze(-1)   # [B, H, S, 1]
        v_s = self.V_scale[layer, :, :, :pos].unsqueeze(-1)
        k   = (k_q * k_s).to(torch.bfloat16)
        v   = (v_q * v_s).to(torch.bfloat16)
        return k, v

    def memory_bytes(self) -> int:
        kv_bytes    = (self.K_q.nelement() + self.V_q.nelement()) * 1   # INT8 = 1 byte
        scale_bytes = (self.K_scale.nelement() + self.V_scale.nelement()) * 4  # F32 = 4 bytes
        return kv_bytes + scale_bytes


class KVCacheFP8:
    """
    FP8 (e4m3) simulated KV cache.
    A100 doesn't have native FP8 in hardware (that's H100), but we simulate
    the quantization/dequantization to show the accuracy and memory tradeoff.
    H100 (next-gen) can do this natively with torch.float8_e4m3fn.

    FP8 e4m3: 4 exponent bits, 3 mantissa bits → range [-448, 448]
    Effective precision ≈ 3 significant decimal digits.
    Memory: 1 byte (same as INT8) → 2x over BF16.
    Advantage over INT8: no per-token scale needed for most layers
    (FP8 handles dynamic range in the floating-point format itself).
    """
    FP8_MAX = 448.0   # max value representable in e4m3

    def __init__(self, config: ModelConfig, max_seq: int, batch: int, device):
        self.config = config
        self.device = device
        # Store as int8 (bit-equivalent storage; we simulate FP8 math)
        self.K_q = torch.zeros(
            config.n_layers, batch, config.n_heads, max_seq, config.head_dim,
            dtype=torch.int8, device=device)
        self.V_q = torch.zeros_like(self.K_q)
        # FP8 still needs a per-tensor (or per-layer) scale to handle activations
        # that exceed FP8_MAX. We use per-layer scales (much cheaper than per-token).
        self.K_scale = torch.ones(config.n_layers, dtype=torch.float32, device=device)
        self.V_scale = torch.ones(config.n_layers, dtype=torch.float32, device=device)
        self._layer_max_k = torch.zeros(config.n_layers, device=device)
        self._layer_max_v = torch.zeros(config.n_layers, device=device)

    def _fp8_quantize(self, x: torch.Tensor, scale: float):
        """Simulate FP8 e4m3 quantization."""
        x_scaled  = x / scale
        x_clipped = x_scaled.clamp(-self.FP8_MAX, self.FP8_MAX)
        # Simulate reduced mantissa precision (3 bits → ~0.125 step)
        step      = 1.0 / 8.0
        x_q       = (x_clipped / step).round() * step
        # Pack to int8 for storage (we just store the scaled values * 8)
        x_stored  = (x_q * 8).round().clamp(-127, 127).to(torch.int8)
        return x_stored

    def _fp8_dequantize(self, x_stored: torch.Tensor, scale: float):
        return (x_stored.float() / 8.0) * scale

    def write(self, layer: int, pos: int, k: torch.Tensor, v: torch.Tensor):
        k_f = k.float()
        v_f = v.float()
        # Update running max for per-layer scale (amax tracking)
        cur_max_k = k_f.abs().max().item()
        cur_max_v = v_f.abs().max().item()
        if cur_max_k > self._layer_max_k[layer]:
            self._layer_max_k[layer] = cur_max_k
            self.K_scale[layer] = max(cur_max_k / self.FP8_MAX, 1e-8)
        if cur_max_v > self._layer_max_v[layer]:
            self._layer_max_v[layer] = cur_max_v
            self.V_scale[layer] = max(cur_max_v / self.FP8_MAX, 1e-8)

        k_scale = self.K_scale[layer].item()
        v_scale = self.V_scale[layer].item()
        self.K_q[layer, :, :, pos, :] = self._fp8_quantize(k_f, k_scale)
        self.V_q[layer, :, :, pos, :] = self._fp8_quantize(v_f, v_scale)

    def read(self, layer: int, pos: int):
        k_q = self.K_q[layer, :, :, :pos, :]
        v_q = self.V_q[layer, :, :, :pos, :]
        k   = self._fp8_dequantize(k_q, self.K_scale[layer].item()).to(torch.bfloat16)
        v   = self._fp8_dequantize(v_q, self.V_scale[layer].item()).to(torch.bfloat16)
        return k, v

    def memory_bytes(self) -> int:
        kv_bytes    = (self.K_q.nelement() + self.V_q.nelement()) * 1
        scale_bytes = (self.K_scale.nelement() + self.V_scale.nelement()) * 4
        return kv_bytes + scale_bytes


# ──────────────────────────────────────────────
# 4. ATTENTION (simplified scaled dot-product)
# ──────────────────────────────────────────────

def scaled_dot_product_attention(q, k, v):
    """
    Standard attention: softmax(Q K^T / sqrt(d)) V
    q: [batch, n_heads, 1, head_dim]       ← current query (one new token)
    k: [batch, n_heads, seq_len, head_dim] ← all past keys
    v: [batch, n_heads, seq_len, head_dim] ← all past values
    """
    d    = q.size(-1)
    # Attention scores: [batch, n_heads, 1, seq_len]
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    # Output: [batch, n_heads, 1, head_dim]
    out  = torch.matmul(attn, v)
    return out, attn


# ──────────────────────────────────────────────
# 5. BENCHMARK RUNNER
# ──────────────────────────────────────────────

def run_benchmark(cache_cls, config: ModelConfig, seq_len: int,
                  batch: int, device, label: str, n_warmup: int = 3) -> dict:
    """
    Simulates LLM decode: token by token, each token does:
      - project Q, K, V from a random hidden state
      - write K/V to cache
      - read full cache and compute attention
    Returns timing, memory, and quantization error stats.
    """
    print(f"\n{'─'*55}")
    print(f"  Running: {label}")
    print(f"  Seq len={seq_len}, Batch={batch}, Layers={config.n_layers}")
    print(f"{'─'*55}")

    cache = cache_cls(config, seq_len + 10, batch, device)

    # Random projection matrices (simulating trained weights)
    W_q = torch.randn(config.d_model, config.n_heads * config.head_dim,
                      dtype=torch.bfloat16, device=device) * 0.02
    W_k = torch.randn_like(W_q)
    W_v = torch.randn_like(W_q)

    # Ground-truth BF16 cache for error measurement
    is_baseline = (cache_cls == KVCacheBF16)
    gt_cache    = None if is_baseline else KVCacheBF16(config, seq_len + 10, batch, device)

    quant_errors_k = []
    quant_errors_v = []

    # ── Warmup ────────────────────────────────
    for _ in range(n_warmup):
        dummy = torch.randn(batch, config.d_model, dtype=torch.bfloat16, device=device)
        for layer in range(min(2, config.n_layers)):
            k = torch.randn(batch, config.n_heads, config.head_dim,
                            dtype=torch.bfloat16, device=device)
            v = torch.randn_like(k)
            cache.write(layer, 0, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # ── Decode loop ───────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for pos in range(seq_len):
        # Simulated hidden state for this token
        hidden = torch.randn(batch, config.d_model, dtype=torch.bfloat16, device=device)

        for layer in range(config.n_layers):
            # Project to Q, K, V — shape [batch, n_heads, head_dim]
            k = (hidden @ W_k).view(batch, config.n_heads, config.head_dim)
            v = (hidden @ W_v).view(batch, config.n_heads, config.head_dim)
            q = (hidden @ W_q).view(batch, config.n_heads, 1, config.head_dim)

            # Write to cache
            cache.write(layer, pos, k, v)

            # Also write to ground-truth cache for error computation
            if gt_cache is not None:
                gt_cache.write(layer, pos, k, v)

            # Only do attention read+compute after at least 1 token is written
            if pos > 0:
                k_cache, v_cache = cache.read(layer, pos)
                # Compute attention
                out, _ = scaled_dot_product_attention(q, k_cache, v_cache)

                # Measure quantization error (last token, first layer only — for speed)
                if layer == 0 and gt_cache is not None and pos % 20 == 0:
                    k_gt, v_gt = gt_cache.read(layer, pos)
                    err_k = (k_cache.float() - k_gt.float()).abs().mean().item()
                    err_v = (v_cache.float() - v_gt.float()).abs().mean().item()
                    quant_errors_k.append(err_k)
                    quant_errors_v.append(err_v)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    mem_bytes   = cache.memory_bytes()
    mem_mb      = mem_bytes / 1e6
    tok_per_sec = seq_len / elapsed
    ms_per_tok  = elapsed * 1000 / seq_len

    # Effective bandwidth: we read the full KV cache each step
    # Total data read ≈ sum over positions of cache_size_at_that_position
    total_bytes_read = sum(
        cache.memory_bytes() * (pos / max(seq_len, 1))
        for pos in range(1, seq_len)
    )
    eff_bw_gbps = total_bytes_read / elapsed / 1e9

    result = {
        "label":           label,
        "seq_len":         seq_len,
        "batch":           batch,
        "elapsed_s":       round(elapsed, 3),
        "ms_per_token":    round(ms_per_tok, 3),
        "tokens_per_sec":  round(tok_per_sec, 1),
        "cache_memory_mb": round(mem_mb, 2),
        "eff_bandwidth_gbps": round(eff_bw_gbps, 2),
        "mean_quant_err_k":  round(float(sum(quant_errors_k) / max(len(quant_errors_k), 1)), 6),
        "mean_quant_err_v":  round(float(sum(quant_errors_v) / max(len(quant_errors_v), 1)), 6),
    }

    print(f"  Cache memory   : {mem_mb:.1f} MB")
    print(f"  Total time     : {elapsed:.3f}s")
    print(f"  ms / token     : {ms_per_tok:.3f}")
    print(f"  Tokens / sec   : {tok_per_sec:.1f}")
    if quant_errors_k:
        print(f"  Mean quant err K: {result['mean_quant_err_k']:.6f}")
        print(f"  Mean quant err V: {result['mean_quant_err_v']:.6f}")

    return result


# ──────────────────────────────────────────────
# 6. MEMORY PROJECTION TABLE
# ──────────────────────────────────────────────

def print_memory_table(config: ModelConfig):
    print("\n═══ KV Cache Memory Projection (batch=1) ═══")
    print(f"{'Seq Len':>10} │ {'BF16 (MB)':>12} │ {'INT8 (MB)':>12} │ {'FP8 (MB)':>12} │ {'INT8 saving':>12}")
    print("─" * 70)
    for seq in [512, 1024, 2048, 4096, 8192, 16384]:
        bf16 = config.kv_bytes(seq, 2) / 1e6
        int8 = config.kv_bytes(seq, 1) / 1e6
        fp8  = config.kv_bytes(seq, 1) / 1e6
        saving = (1 - int8 / bf16) * 100
        print(f"{seq:>10,} │ {bf16:>12.1f} │ {int8:>12.1f} │ {fp8:>12.1f} │ {saving:>11.0f}%")


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset",  default="small",
                        choices=["small", "medium", "large"],
                        help="Model size preset")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Number of decode steps (tokens)")
    parser.add_argument("--batch",   type=int, default=1)
    parser.add_argument("--out",     default="results.json",
                        help="Path to save JSON results for visualization")
    args = parser.parse_args()

    device, dev_info = setup_device()
    config = ModelConfig(preset=args.preset)

    print("\n" + "═"*55)
    print("  Quantized KV Cache Benchmark")
    print("  PARAM Rudra (IIT Patna) / A100-80GB")
    print("═"*55)
    print(f"  Device    : {dev_info['name']}")
    print(f"  GPU VRAM  : {dev_info['total_memory_gb']} GB")
    print(f"  CUDA      : {dev_info['cuda_version']}")
    print(f"  Model     : {args.preset} ({config.n_layers}L × {config.n_heads}H × {config.head_dim}D)")
    print(f"  d_model   : {config.d_model}")
    print(f"  Seq len   : {args.seq_len}")
    print(f"  Batch     : {args.batch}")

    print_memory_table(config)

    results = []

    # BF16 baseline
    r = run_benchmark(KVCacheBF16, config, args.seq_len, args.batch,
                      device, "BF16 (baseline)")
    results.append(r)

    # INT8
    r = run_benchmark(KVCacheINT8, config, args.seq_len, args.batch,
                      device, "INT8 (per-token dynamic)")
    results.append(r)

    # FP8
    r = run_benchmark(KVCacheFP8, config, args.seq_len, args.batch,
                      device, "FP8 e4m3 (simulated)")
    results.append(r)

    # Save JSON for visualizer
    output = {
        "device_info": dev_info,
        "model": {
            "preset": args.preset,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "head_dim": config.head_dim,
            "d_model": config.d_model,
        },
        "results": results,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n Results saved to: {out_path}")
    print(" Run visualizer : python visualize_results.py")
    print(" Or open        : kv_cache_dashboard.html in browser\n")

    # ── Summary table ─────────────────────────
    print("═"*65)
    print(f"  {'Method':<26} {'Mem MB':>8} {'ms/tok':>8} {'tok/s':>8} {'Err K':>10}")
    print("─"*65)
    bf16_mem = results[0]["cache_memory_mb"]
    for r in results:
        saving = (1 - r["cache_memory_mb"] / bf16_mem) * 100 if bf16_mem > 0 else 0
        err = f"{r['mean_quant_err_k']:.5f}" if r["mean_quant_err_k"] > 0 else "  —  "
        print(f"  {r['label']:<26} {r['cache_memory_mb']:>6.1f}MB "
              f"{r['ms_per_token']:>8.2f} {r['tokens_per_sec']:>8.1f} {err:>10}")
    print("═"*65)

    return output


if __name__ == "__main__":
    main()
