#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_aftertune/sh_logs/dygformer

# Run experiments for DyGFormer with kan_mammote_dual_kmote time encoder
# Second batch (GPU-S): mooc, lastfm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python experiments/train_link_prediction.py --model_name DyGFormer --dataset_name lastfm \
    --time_encoder_type kan_mammote_dual_kmote --num_runs 1 \
    --use_amp --expert_dim 64 --mamba_d_state 64 --mamba_expand 4  \
    --encoder_dropout 0.1 --mamba_headdim 32 --use_gradient_checkpointing --disable_progress_bar \
    > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_lastfm_kmote.log 2>&1