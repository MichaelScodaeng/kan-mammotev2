#PBS -j oe
#PBS -q GPU-S
#PBS -l select=1:ngpus=2:ncpus=52
#PBS -l walltime=120:00:00
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p sh_scripts/sh_logs/lete
# Task assignments
# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py \
  --models "DyGMamba" \
  --single_encoder "lete" \
  --datasets wikipedia enron \
  --disable_progress_bar --num_runs 1 > sh_scripts/sh_logs/lete/lete_DyGMamba_04.log 2>&1 &
