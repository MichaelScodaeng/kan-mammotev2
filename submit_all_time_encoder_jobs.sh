#!/bin/bash

# Submit all time encoder experiments to HPC
# This script submits separate jobs for each time encoder for parallel execution

TIME_ENCODERS=("original" "lete" "kan_mammote" "mercer" "bochner" "time2vec")

echo "Submitting Time Encoder Experiments to HPC"
echo "==========================================="

for encoder in "${TIME_ENCODERS[@]}"; do
    echo "Submitting job for time encoder: $encoder"
    
    # Submit job with time encoder as environment variable
    job_id=$(qsub -v TIME_ENCODER="$encoder" run_time_encoder_experiment.sh)
    
    if [ $? -eq 0 ]; then
        echo "✅ Job submitted: $job_id (encoder: $encoder)"
    else
        echo "❌ Failed to submit job for encoder: $encoder"
    fi
    
    # Small delay between submissions
    sleep 2
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: qstat -u $USER"
echo ""
echo "Check status of specific encoder experiments:"
for encoder in "${TIME_ENCODERS[@]}"; do
    echo "  python experiment_kanmammote.py --single_encoder $encoder --generate_report"
done

echo ""
echo "Log files will be created as:"
echo "  - experiment_{encoder}_{job_id}.log"
echo "  - run_time_encoder_experiment.sh.o{job_id}"
echo "  - nvidia-smi_{encoder}_{job_id}.log"
