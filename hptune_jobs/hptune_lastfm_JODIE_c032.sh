#!/bin/bash
#PBS -N hptune_lastfm_JODIE_c032
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o hptune_jobs/logs/hptune_lastfm_JODIE_c032.log

# Load environment
cd $PBS_O_WORKDIR
source /home/s2516027/kan-mammotev2/.venv/bin/activate

# Print job info
echo "========================================="
echo "Job: hptune_lastfm_JODIE_c032"
echo "Dataset: lastfm"
echo "Model: JODIE"
echo "Config Index: 32"
echo "Config: {
  "expert_dim": 256,
  "mamba_d_state": 512,
  "mamba_expand": 4,
  "dropout": 0,
  "mamba_headdim": 64,
  "mamba_d_conv": 4
}"
echo ""
echo "Mamba2 Validation:"
echo "  expert_dim × mamba_expand = 256 × 4 = 1024"
echo "  inner_dim / mamba_headdim = 1024 / 64 = 16 (ngroups)"
echo "  ngroups % 8 = 0 ✓ (valid)" 
echo ""
echo "Start Time: $(date)"
echo "========================================="

# Run experiment
python experiments/train_link_prediction.py --dataset_name lastfm --model_name JODIE --time_encoder_type kan_mammote_dual_kmote --expert_dim 256 --mamba_d_state 512 --mamba_expand 4 --dropout 0 --mamba_headdim 64 --mamba_d_conv 4 --data_ratio 1.0 --train_only_ratio 1.0 --num_epochs 10 --patience 3 --num_runs 1 --seed 0 --test_interval_epochs 1 --checkpoint_strategy minimal --disable_progress_bar --save_model_name_suffix hptune_c032_ed256_ds512_ex4 --ablation_dir ./hptune_results/lastfm/JODIE

# Print completion
echo "========================================="
echo "End Time: $(date)"
echo "========================================="
