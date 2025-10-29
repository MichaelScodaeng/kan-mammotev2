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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PBS exposes the two allocated GPUs via CUDA_VISIBLE_DEVICES like "3,5".
echo "PBS-assigned GPUs: $CUDA_VISIBLE_DEVICES"

# Map each process to one of the assigned logical IDs (0 and 1 in this namespace)
CUDA_VISIBLE_DEVICES=0 python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models TGAT \
  --datasets SocialEvo --disable_progress_bar --num_runs 1 \
  --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
  > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_socialevo_kmote02.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models TGAT \
  --datasets Flights --disable_progress_bar --num_runs 1 \
  --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
  > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_flights_kmote02.log 2>&1 &

wait
