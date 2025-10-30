#!/bin/bash
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be



cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote

mkdir -p sh_scripts/kmm_aftertune/sh_logs/tgat

# PBS sets CUDA_VISIBLE_DEVICES to the GPUs you own (e.g., "2,5").
echo "PBS-assigned GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"

# Optional: help PyTorch memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run experiments for TGAT with kan_mammote_dual_kmote time encoder
# Fourth batch: UNtrade, Enron

# UNtrade (you set 64/64/2/0.1 here)
CUDA_VISIBLE_DEVICES=0 python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models TGAT \
  --datasets UNtrade --disable_progress_bar --num_runs 1 \
  --expert_dim 128 --mamba_d_state 64 --mamba_expand 2 --mamba_headdim 32 --encoder_dropout 0.1 \
  > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_untrade_kmote02.log 2>&1 &

# Enron (you set 64/64/4/0.1 here)
CUDA_VISIBLE_DEVICES=1 python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models TGAT \
  --datasets Enron --disable_progress_bar --num_runs 1 \
  --expert_dim 128 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
  > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_enron_kmote02.log 2>&1 &

wait
