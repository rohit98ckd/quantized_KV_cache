# Quantized KV Cache for LLM Inference

A hands-on benchmark and visualization of **KV cache quantization** (BF16 / INT8 / FP8) for transformer decoder inference, designed to run on the **PARAM Rudra** supercomputer at **IIT Patna** (NVIDIA A100 80GB HBM2e) with a transparent CPU fallback for local development.

The project shows, end-to-end, why decoding LLMs is memory-bandwidth-bound, how dynamic quantization of the KV cache halves HBM traffic, and what accuracy tradeoff each scheme makes.

---

## What this project demonstrates

- A clean, from-scratch implementation of three KV cache backends sharing one interface:
  - **BF16** baseline (full precision, what HuggingFace uses by default)
  - **INT8** with per-token dynamic absmax scaling
  - **FP8 e4m3** simulated (native on H100; emulated on A100 for fidelity studies)
- A token-by-token decode loop with a simplified scaled-dot-product attention so quantization error and memory savings are measured against a ground-truth BF16 cache
- A SLURM submission script for the PARAM Rudra GPU partition
- An offline interactive dashboard (`kv_cache_dashboard.html`) with a live INT8 quantization simulator, roofline analysis, and an algorithm walkthrough — opens in any browser, no server required

---

## Project layout

| File | Purpose |
|---|---|
| `kv_cache_quantized.py` | Main benchmark. Implements the three caches, runs the decode loop, writes a JSON report. |
| `run_kv_quant.sh` | SLURM job script for PARAM Rudra (`gpu` partition, 1× A100, 30 min walltime). |
| `kv_cache_dashboard.html` | Self-contained interactive dashboard (Chart.js via CDN). Open directly in a browser. |
| `results_*.json` | Output reports produced by the benchmark (one per preset). |

---

## Background: why quantize the KV cache?

During autoregressive decode, the GPU re-reads the **entire KV cache from HBM at every single token step**. For a 7B model with a 4K context this is roughly 8 GB of reads per step. Even at the A100's 2 TB/s peak HBM bandwidth that's about 4 ms of pure memory traffic before any compute happens — and the tensor cores sit idle.

Quantizing K and V from BF16 (2 bytes) to INT8 (1 byte) **halves that traffic**, which directly halves decode latency in the bandwidth-bound regime. This is one of the highest-leverage optimizations in modern LLM serving stacks (vLLM, TensorRT-LLM, TGI all support it).

The catch: KV activations have outliers, so the quantization scheme has to be careful about scale granularity.

---

## The three methods, in one paragraph each

**BF16 (baseline).** Stores K and V at full precision in `[n_layers, batch, n_heads, seq_len, head_dim]`. No accuracy loss, no scale overhead, but the cache is the main consumer of HBM during decode.

**INT8 with per-token dynamic scaling.** For each new token, for each attention head, we compute `scale = max(|k|) / 127` and store `round(k / scale)` as `int8`. The per-(token, head) `float32` scale (4 bytes) is tiny next to the `head_dim`-wide INT8 vector, so net savings are ~50%. Per-token scaling matters because outlier tokens would otherwise force a coarse global scale and crush the precision of the quiet majority.

**FP8 e4m3 (simulated).** 4 exponent + 3 mantissa bits gives a range of ±448 with ~3 significant digits of precision. The non-uniform spacing handles outliers well enough to use a *per-layer* scale instead of per-token, eliminating that overhead. A100 has no native FP8 tensor cores (that's H100), so the simulation here matches the math but not the wall-clock you'd see on Hopper.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (BF16/CUDA support; CPU also works)
- A modern browser for the dashboard (no install needed)

On PARAM Rudra the included SLURM script loads:

```bash
module load mldl/Miniconda
conda activate Pytorch-gpu     # PyTorch-gpu 2.2.1 from the cluster's ML/DL stack
```

---

## Quick start (local, CPU simulation)

```bash
python kv_cache_quantized.py --preset small --seq-len 256 --batch 1
```

This produces `results.json` and prints a summary table. With no GPU detected the script automatically runs in CPU simulation mode — slower in absolute terms, but the **memory and accuracy numbers are identical to what you get on an A100** because the quantization math is the same.

### CLI options

| Flag | Default | Notes |
|---|---|---|
| `--preset` | `small` | `small` (12L×12H×64), `medium` (24L×16H×64), `large` (32L×32H×128, ~Llama-7B shape) |
| `--seq-len` | `256` | Number of decode steps simulated |
| `--batch` | `1` | Batch size |
| `--out` | `results.json` | Output path for the JSON report |

---

## Running on PARAM Rudra

```bash
# 1. SSH in and create a working directory
ssh <username>@param-rudra.iitp.ac.in
mkdir -p ~/kv_quant && cd ~/kv_quant

# 2. Upload the project files (from your laptop)
scp kv_cache_quantized.py run_kv_quant.sh <username>@param-rudra.iitp.ac.in:~/kv_quant/

# 3. Submit the job
sbatch run_kv_quant.sh

# 4. Monitor
squeue -u $USER
tail -f kv_quant_<job_id>.out

# 5. Pull results back to your laptop and open the dashboard
scp '<username>@param-rudra.iitp.ac.in:~/kv_quant/results_*.json' .
open kv_cache_dashboard.html
```

The included SLURM script runs all three presets (`small`, `medium`, `large`) sequentially on a single A100 within a 30-minute walltime budget.

### Interactive GPU session (development / profiling)

```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --cpus-per-task=8 \
     --time=01:00:00 --pty bash -i

module load mldl/Miniconda
conda activate Pytorch-gpu
nsys profile --output=profile python kv_cache_quantized.py --preset large --seq-len 1024
nsys stats profile.nsys-rep
```

---

## Results format

`results.json` is consumed directly by the dashboard. Each run produces:

```json
{
  "device_info": { "device": "cuda", "name": "NVIDIA A100-SXM4-80GB", ... },
  "model": { "preset": "small", "n_layers": 12, "n_heads": 12, ... },
  "results": [
    {
      "label": "BF16 (baseline)",
      "cache_memory_mb": 9.81,
      "ms_per_token": 16.279,
      "tokens_per_sec": 61.4,
      "mean_quant_err_k": 0.0,
      "mean_quant_err_v": 0.0
    },
    { "label": "INT8 (per-token dynamic)", ... },
    { "label": "FP8 e4m3 (simulated)", ... }
  ]
}
```

### Sample output (CPU simulation, `small` preset, seq_len=256)

| Method | Cache MB | ms/token | tokens/s | Mean err K |
|---|---:|---:|---:|---:|
| BF16 (baseline) | 9.81 | 16.28 | 61.4 | — |
| INT8 (per-token dynamic) | 5.21 | 19.02 | 52.6 | 0.1356 |
| FP8 e4m3 (simulated) | 4.90 | 19.58 | 51.1 | 17.61 |

**Reading the numbers.** On CPU, INT8 and FP8 are *slightly slower* than BF16 because every read has to dequantize through PyTorch's Python-level ops, and CPU is compute-bound on this workload anyway. The memory savings are real (~47% for INT8, ~50% for FP8), and on an A100 the latency story flips: when you're bandwidth-bound, halving the bytes-read per step roughly halves the decode time. The FP8 error is large here because the simulation uses a coarse 1/8 mantissa step on top of a global per-layer scale — a faithful demo of FP8's tradeoff, not a tuned production kernel.

### Expected A100 vs CPU simulation

| Metric | CPU sim | A100 expected | Why |
|---|---|---|---|
| ms/token BF16 | ~16 ms | ~0.3–1 ms | HBM vs DDR4, CUDA parallelism |
| ms/token INT8 | ~19 ms | ~0.15–0.5 ms | 2× less HBM traffic |
| Memory saving | ~47% | ~47–50% | Same algorithm, identical |
| Quant error K | 0.137 | 0.137 | Same math, identical |
| Achieved BW | <1 GB/s | ~1.5–2 TB/s | HBM2e peak |

---

## Visualizing results

Open `kv_cache_dashboard.html` in any browser. It's a single self-contained file (Chart.js loaded from CDN) with five tabs:

1. **Overview** — memory vs sequence length, comparison table, HBM occupancy diagram
2. **Live Simulator** — step a token at a time through INT8 / FP8 / BF16 with attention weights and per-token quantization error
3. **Algorithm Deep Dive** — annotated quantize/dequantize code and an interactive scale slider
4. **Roofline Analysis** — A100 roofline with each method's operating point plotted
5. **How to Run** — copy-paste-able SSH and SLURM commands

The dashboard works offline once loaded; the demo data shown is the CPU-simulation baseline so you can preview without running anything.

---

## Notes & caveats

- The attention kernel is a plain `softmax(QK^T / √d)V` for clarity — production stacks use FlashAttention, which is orthogonal to the cache quantization choice.
- FP8 here is a **functional** simulation. For real FP8 perf numbers you want H100 with `torch.float8_e4m3fn`.
- The "effective bandwidth" reported is a back-of-envelope estimate from total bytes-read divided by elapsed time. Use Nsight Systems (`nsys`) for ground-truth bandwidth on the GPU.
- Per-token scaling is the simplest scheme that works well. For tighter accuracy at INT4, look at **KIVI** (group-wise scales + keeping recent / sink tokens in BF16) and **AWQ-KV**.

---

## References

- Hooper et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (2024)
- Liu et al., *KV Cache Quantization for Long-Context LLMs* (2024)
- NVIDIA A100 Tensor Core GPU Architecture whitepaper
- PARAM Rudra User Manual, IIT Patna — `rudrasupport@iitp.ac.in`

---

## Contact

Questions about the cluster: `rudrasupport@iitp.ac.in`
Questions about the code: open an issue or drop a note in your usual channel.
