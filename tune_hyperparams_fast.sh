#!/bin/bash
#PBS -N hp_tune_fast
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -j oe

# Fast Hyperparameter Tuning Script for HPC
# Tunes learning rate and weight decay across models, datasets, and time encoders
# Uses 10% temporal prefix data, 10 epochs, patience 3

cd $PBS_O_WORKDIR

# Activate virtual environment
source .venv/bin/activate

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run with subset for testing (remove --subset to run all)
# python tune_hyperparams_fast.py \
#     --models TGAT TGN \
#     --datasets wikipedia reddit \
#     --time_encoders lete mercer \
#     --gpu 0 \
#     --subset 10

# Full run - uncomment to run all experiments
python tune_hyperparams_fast.py \
    --models JODIE TGAT TGN TCL DyGFormer DyGMamba \
    --datasets wikipedia reddit mooc lastfm enron SocialEvo uci CanParl Contacts Flights UNtrade UNvote USLegis \
    --time_encoders lete kan_mammote_dual_kmote mercer time2vec \
    --gpu 0

echo "Hyperparameter tuning completed at $(date)"
