#!/bin/bash
#PBS -N resume_JODIE_SocialEvo_time2vec
#PBS -l select=1:ncpus=8:mem=60gb:ngpus=1:gpu_model=a100
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -o /home/s2516027/kan-mammotev2/resume_jobs/resume_jodie_socialevo_time2vec.o
#PBS -e /home/s2516027/kan-mammotev2/resume_jobs/resume_jodie_socialevo_time2vec.e

# Load required modules
module load python/3.11.0-ffypltn cuda/12.1.1-y3rfgp6

# Navigate to project directory  
cd /home/s2516027/kan-mammotev2

# Activate environment
source mambaforge/envs/py11_cuda121/bin/activate

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# Resume training from checkpoint
echo "===== RESUMING TRAINING ====="
echo "Model: JODIE"
echo "Dataset: SocialEvo"
echo "Time Encoder: time2vec"
echo "Checkpoint: /home/s2516027/kan-mammotev2/saved_models/JODIE/SocialEvo/JODIE_time2vec_seed0/checkpoint_epoch_100.pth"
echo "Resuming from epoch: 101"
echo "Target epochs: 200"
echo "=============================="

python -u experiments/train_link_prediction.py \
    --model_name JODIE \
    --dataset_name SocialEvo \
    --time_encoder time2vec \
    --num_epochs 200 \
    --num_runs 1 \
    --seed 0 \
    --resume_from_checkpoint /home/s2516027/kan-mammotev2/saved_models/JODIE/SocialEvo/JODIE_time2vec_seed0/checkpoint_epoch_100.pth \
    --validate_checkpoints \
    --save_checkpoints \
    --checkpoint_interval 10 \
    --max_checkpoints_to_keep 5 \
    --disable_progress_bar \
    --gpu 0

echo "===== TRAINING COMPLETED ====="
