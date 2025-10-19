#PBS -j oe
#PBS -q GPU-LA
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p sh_scripts/sh_logs/kmm
# Task assignments
# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py \
  --models "DyGMamba" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets SocialEvo enron \
  --disable_progress_bar --num_runs 1 > sh_scripts/sh_logs/kmm/kmm_DyGMamba_05.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified.py \
  --models "DyGMamba" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets UNvote Flights \
  --disable_progress_bar --num_runs 1 > sh_scripts/sh_logs/kmm/kmm_DyGMamba_06.log 2>&1 &

wait
