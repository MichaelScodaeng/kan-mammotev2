#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p validation/logs
# Task assignments
# Task 1 on GPU 0
python -u run_evaluation_all.py > validation/logs/01.log 2>&1 