#!/bin/bash -x
# ═══════════════════════════════════════════════════════════════════
#  SLURM Job Script — Quantized KV Cache Benchmark
#  Cluster : PARAM Rudra, IIT Patna
#  Node    : GPU partition (NVIDIA A100 80GB HBM2e)
#  Contact : rudrasupport@iitp.ac.in
# ═══════════════════════════════════════════════════════════════════

#SBATCH --job-name=kv_quant_bench
#SBATCH --partition=gpu              # GPU partition on PARAM Rudra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # 8 CPU cores per GPU task
#SBATCH --gres=gpu:1                 # Request 1 A100 GPU
#SBATCH --mem=32G                    # 32 GB RAM
#SBATCH --time=00:30:00              # 30 min walltime
#SBATCH --output=kv_quant_%j.out    # stdout → file with job ID
#SBATCH --error=kv_quant_%j.err     # stderr
#SBATCH --mail-type=END,FAIL         # email on completion or failure
# #SBATCH --mail-user=your@iitp.ac.in   # uncomment + set your email

# ── Move to working directory ──────────────────────────────────────
cd $SLURM_SUBMIT_DIR

echo "═══════════════════════════════════════════════"
echo " Job ID       : $SLURM_JOB_ID"
echo " Node         : $SLURMD_NODENAME"
echo " Submit dir   : $SLURM_SUBMIT_DIR"
echo " Start time   : $(date)"
echo "═══════════════════════════════════════════════"

# ── Load modules (PARAM Rudra module names from User Manual §ML/DL) ─
module purge
module load mldl/Miniconda

# Activate the PyTorch GPU environment
# Available envs (from manual): Pytorch-gpu 2.2.1
conda activate Pytorch-gpu

# Verify GPU
echo ""
echo "── GPU Info ───────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
           --format=csv,noheader
echo "── CUDA / PyTorch ─────────────────────────────"
python -c "import torch; print('PyTorch:', torch.__version__); \
           print('CUDA:', torch.version.cuda); \
           print('GPU:', torch.cuda.get_device_name(0)); \
           print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

# ── Run benchmarks (three presets) ────────────────────────────────
echo ""
echo "── Running: small model ───────────────────────"
python kv_cache_quantized.py --preset small  --seq-len 512  --batch 4 \
       --out results_small.json

echo ""
echo "── Running: medium model ──────────────────────"
python kv_cache_quantized.py --preset medium --seq-len 1024 --batch 2 \
       --out results_medium.json

echo ""
echo "── Running: large model (Llama-7B shape) ──────"
python kv_cache_quantized.py --preset large  --seq-len 2048 --batch 1 \
       --out results_large.json

# ── Profile with nsys (optional — comment out if nsys not available) ─
# module load nsys   # load NVIDIA Nsight Systems if available
# nsys profile --output=kv_quant_profile \
#      python kv_cache_quantized.py --preset medium --seq-len 256

echo ""
echo "── Done ───────────────────────────────────────"
echo " End time: $(date)"
echo " Results : results_small.json, results_medium.json, results_large.json"
echo ""
echo " Transfer results to your laptop with:"
echo "   scp username@param-rudra.iitp.ac.in:$SLURM_SUBMIT_DIR/results_*.json ."
echo " Then open kv_cache_dashboard.html in your browser."
