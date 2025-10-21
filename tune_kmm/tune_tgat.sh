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
mkdir -p tune_kmm/tgat
# Task assignments
# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python -u tune_kan_mammote_direct.py --models TGAT --datasets 'wikipedia' 'reddit' 'mooc' 'lastfm' 'enron' 'SocialEvo' 'uci' \
                                 > tune_kmm/tgat/tune_tgat01.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python -u tune_kan_mammote_direct.py --models TGAT --datasets 'CanParl' 'Contacts' 'Flights' 'UNtrade' 'UNvote' 'USLegis' > tune_kmm/tgat/tune_tgat02.log 2>&1 &

wait
