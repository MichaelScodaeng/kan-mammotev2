#!/bin/bash
#PBS -j oe
#PBS -q GPU-LA
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote

# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py \
  --models "TGAT" \
  --single_encoder "time2vec" \
  --datasets Contacts \
  --disable_progress_bar --num_runs 1 > current_thin/sh_logs/TGAT_time2vec_01_contacts.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified.py \
  --models "TGAT" \
  --single_encoder "time2vec" \
  --datasets Flights uci \
  --disable_progress_bar --num_runs 1 > current_thin/sh_logs/TGAT_time2vec_02_flights_uci.log 2>&1 &

wait