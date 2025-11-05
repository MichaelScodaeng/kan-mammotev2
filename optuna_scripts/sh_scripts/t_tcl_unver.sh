#PBS -j oe
#PBS -q GPU-S
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
    --datasets UNvote \
    --models TCL \
    --trials_per_combo 25 \
    > optuna_scripts/sh_scripts/sh_logs/tcl_unvote.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 \
python -u tune_kan_mammote_optuna.py \
    --multi_dataset \
    --datasets enron \
    --models TCL \
    --trials_per_combo 25 \
    > optuna_scripts/sh_scripts/sh_logs/tcl_enron.log 2>&1 &
wait