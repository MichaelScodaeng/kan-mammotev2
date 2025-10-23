#PBS -j oe
#PBS -q GPU-L
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p optuna_scripts/sh_scripts/sh_logs
# Task assignments
# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python -u tune_kan_mammote_optuna.py \
    --multi_dataset \
    --datasets lastfm UNtrade wikipedia CanParl \
    --models DyGFormer \
    --trials_per_combo 30 \
    > optuna_scripts/sh_scripts/sh_logs/dgf_tune01.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python -u tune_kan_mammote_optuna.py \
    --multi_dataset \
    --datasets UNvote mooc enron uci USLegis \
    --models DyGFormer \
    --trials_per_combo 30 \
    > optuna_scripts/sh_scripts/sh_logs/dgf_tune02.log 2>&1 &

wait