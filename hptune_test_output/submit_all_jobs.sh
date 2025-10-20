#!/bin/bash

# Auto-generated script to submit all hyperparameter tuning jobs
# Generated: 2025-10-21T04:48:50.400058
# Total jobs: 2

qsub hptune_test_output/hptune_wikipedia_TGAT_c000.sh
sleep 1  # Avoid overwhelming scheduler
qsub hptune_test_output/hptune_wikipedia_TGAT_c001.sh
sleep 1  # Avoid overwhelming scheduler
