#!/bin/bash
#PBS -N hptune_wikipedia_TGAT_c000
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o hptune_test_output/logs/hptune_wikipedia_TGAT_c000.log

# Load environment
cd $PBS_O_WORKDIR
source /home/s2516027/kan-mammotev2/.venv/bin/activate

# Print job info
echo "========================================="
echo "Job: hptune_wikipedia_TGAT_c000"
echo "Dataset: wikipedia"
echo "Model: TGAT"
echo "Config Index: 0"
echo "Config: {
  "expert_dim": 128,
  "mamba_d_state": 256,
  "mamba_expand": 2,
  "dropout": 0.1,
  "mamba_headdim": 64,
  "mamba_d_conv": 4
}"
echo "Start Time: $(date)"
echo "========================================="

# Run experiment
python experiments/train_link_prediction.py --dataset_name wikipedia --model_name TGAT --time_encoder_type kan_mammote_dual_kmote --expert_dim 128 --mamba_d_state 256 --mamba_expand 2 --dropout 0.1 --mamba_headdim 64 --mamba_d_conv 4 --data_ratio 0.1 --num_epochs 10 --patience 3 --num_runs 1 --seed 0 --test_interval_epochs 2 --checkpoint_strategy minimal --disable_progress_bar --save_model_name_suffix hptune_c000 --ablation_dir ./hptune_results/wikipedia/TGAT

# Print completion
echo "========================================="
echo "End Time: $(date)"
echo "========================================="
