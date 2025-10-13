#!/bin/bash
#PBS -N TGAT_LA_7GPU_MPI
#PBS -q GPU-LA
# 7 GPUs total: 3 nodes×2 GPU + 1 node×1 GPU
#PBS -l select=3:ncpus=52:ngpus=2:mem=512gb+1:ncpus=52:ngpus=1:mem=512gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m be
#PBS -M s2516027@jaist.ac.jp

cd "$PBS_O_WORKDIR"
source ~/.bashrc
conda activate kan_mammote

hostname
nvidia-smi -L

# ---- exactly 7 datasets (one per GPU) ----
cat > datasets.txt <<'EOF'
CanParl
Contacts
Flights
SocialEvo
lastfm
mooc
reddit
EOF
# uci not used yet

# ---- Per-rank launcher ----
cat > run_one.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
RANK=${OMPI_COMM_WORLD_RANK:-0}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}   # 0..1 on 2-GPU nodes
DATASET=$(sed -n "$((RANK+1))p" datasets.txt)

echo "[`hostname`] global_rank=$RANK local_rank=$LOCAL_RANK → GPU=$LOCAL_RANK dataset=$DATASET"

export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
python experiment_unified.py \
  --models "TGAT" \
  --datasets "$DATASET" \
  --single_encoder "kan_mammote_dual_kmote" \
  --disable_progress_bar \
  --num_runs 1 \
  > "kanmammote_${DATASET}_tgatA100.log" 2>&1
EOF
chmod +x run_one.sh

# ---- MPI launcher ----
# Use 7 total ranks; 2 ranks per 2-GPU node, 1 rank for the 1-GPU node.
# Let OpenMPI automatically map: 2,2,2,1 ranks → 7 GPUs total.
mpirun -np 7 --hostfile "$PBS_NODEFILE" --map-by node --bind-to none ./run_one.sh

echo "✅ All 7 A100 jobs completed successfully."