#PBS -j oe
#PBS -q GPU-S
#PBS -l select=1:ngpus=2:ncpus=52
#PBS -l walltime=120:00:00
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
conda activate kan_mammote

# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py \
  --models "DyGMamba" \
  --single_encoder "lete" \
  --datasets uci CanParl \
  --disable_progress_bar --num_runs 1 > lete_DyGMamba_01.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified.py \
  --models "DyGMamba" \
  --single_encoder "lete" \
  --datasets Contacts Flights \
  --disable_progress_bar --num_runs 1 > lete_DyGMamba_02.log 2>&1 &

wait