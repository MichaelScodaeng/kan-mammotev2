#!/bin/bash

# KAN-MAMMOTE Time Encoder Comparison Experiment
# This script runs experiments comparing Original, LeTE, and KAN-MAMMOTE encoders

DATASET="wikipedia"
MODEL="DyGMamba"
GPU=0
EPOCHS=2
BATCH_SIZE=200

echo "=== KAN-MAMMOTE Time Encoder Comparison Experiment ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "GPU: $GPU"
echo

# Create results directory
mkdir -p results/time_encoder_comparison

echo "1. Running with Original Time Encoder..."
python experiments/train_link_prediction.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --time_encoder_type original \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gpu $GPU \
    --save_result_name original_encoder \
    --save_model_dir results/time_encoder_comparison/original \
    2>&1 | tee results/time_encoder_comparison/original.log

echo
echo "2. Running with LeTE..."
python experiments/train_link_prediction.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --time_encoder_type lete \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gpu $GPU \
    --save_result_name lete_encoder \
    --save_model_dir results/time_encoder_comparison/lete \
    2>&1 | tee results/time_encoder_comparison/lete.log

echo
echo "3. Running with KAN-MAMMOTE..."
python experiments/train_link_prediction.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --time_encoder_type kan_mammote \
    --expert_dim 64 \
    --num_mixtures 8 \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gpu $GPU \
    --save_result_name kan_mammote_encoder \
    --save_model_dir results/time_encoder_comparison/kan_mammote \
    2>&1 | tee results/time_encoder_comparison/kan_mammote.log

echo
echo "=== All experiments completed! ==="
echo "Results saved in: results/time_encoder_comparison/"
echo
echo "To compare results:"
echo "python analyze_results.py --result_dir results/time_encoder_comparison/"
