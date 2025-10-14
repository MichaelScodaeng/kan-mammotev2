#!/bin/bash
#PBS -j oe
#PBS -q GPU-LA
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
conda activate kan_mammote
nvidia-smi -a > nvidia-smi.log

# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py \
  --models "TGAT" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets mooc \
  --disable_progress_bar --num_runs 1 > kanmammote_TGAT_01_mooc.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified.py \
  --models "TGAT" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets USLegis \
  --disable_progress_bar --num_runs 1 > kanmammote_TGAT_02_uslegis.log 2>&1 &

wait