#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
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
python -u tune_kan_mammote_optuna.py \
    --multi_dataset \
    --datasets UNvote \
    --models TGN \
    --trials_per_combo 50 \
    --num_epochs 30 \
    > optuna_scripts/sh_scripts/sh_logs/tgn_unv.log 2>&1