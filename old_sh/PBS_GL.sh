#!/bin/bash
#PBS -N GL_5GPU_MPI
#PBS -q GPU-L
# 5 GPUs total: 2 nodes×2 GPU + 1 node×1 GPU
#PBS -l select=2:ncpus=52:ngpus=2:mem=512gb+1:ncpus=26:ngpus=1:mem=256gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m be
#PBS -M s2516027@jaist.ac.jp

cd "$PBS_O_WORKDIR"
source ~/.bashrc
conda activate kan_mammote
hostname
nvidia-smi -L

# If your site needs it:
# module load openmpi

# Exactly 5 (model,dataset) pairs — 3 TGAT + 2 TGN
cat > tasks.csv <<'EOF'
TGAT,UNtrade
TGAT,UNvote
TGAT,USLegis
TGN,SocialEvo
TGN,UNtrade
EOF

cat > run_one.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
RANK=${OMPI_COMM_WORLD_RANK:-0}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}   # 0..(gpus_on_node-1)
TASK=$(sed -n "$((RANK+1))p" tasks.csv)
MODEL=${TASK%,*}; DATASET=${TASK#*,}

echo "[`hostname`] global_rank=$RANK local_rank=$LOCAL_RANK -> GPU=$LOCAL_RANK  ${MODEL}/${DATASET}"
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

python experiment_unified.py \
  --models "$MODEL" \
  --datasets "$DATASET" \
  --single_encoder "kan_mammote_dual_kmote" \
  --disable_progress_bar --num_runs 1 \
  > "kanmammote_${DATASET}_${MODEL}_A40.log" 2>&1
EOF
chmod +x run_one.sh

# Let OpenMPI place ranks 1-per-node round-robin → distribution becomes 2,2,1
# Use PBS's hostfile so mpirun knows your allocated nodes
mpirun -np 5 --hostfile "$PBS_NODEFILE" --map-by node --bind-to none ./run_one.sh

echo "✅ All 5 A40 jobs completed."



# Not available yet "UNvote" "USLegis" "enron"