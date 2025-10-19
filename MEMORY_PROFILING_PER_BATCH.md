# Memory Profiling Per-Batch Output

## Changes Made

Modified `train_link_prediction_estimate.py` to print **detailed memory snapshots at each step** within every training batch, instead of just a summary at the end.

## What You'll See Now

For each batch during timing (up to 10 batches), you'll see:

```
======================================================================
📦 BATCH 1/10
======================================================================

📸 Batch 1 - Start
--------------------------------------------------
GPU Memory:
  ├─ Allocated: 234.56 MB
  ├─ Reserved:  512.00 MB
  ├─ Cached:    512.00 MB
  └─ Peak:      234.56 MB
CPU Memory:
  ├─ Used:      1234.56 MB
  └─ Percent:   12.3%
--------------------------------------------------

   Batch size: 200

📸 Batch 1 - After data loading
--------------------------------------------------
GPU Memory:
  ├─ Allocated: 245.78 MB
  ├─ Reserved:  512.00 MB
  ├─ Cached:    512.00 MB
  └─ Peak:      245.78 MB
CPU Memory:
  ├─ Used:      1245.67 MB
  └─ Percent:   12.4%
--------------------------------------------------

📸 Batch 1 - After positive embeddings
--------------------------------------------------
GPU Memory:
  ├─ Allocated: 567.89 MB  ← Watch for big jumps here!
  ├─ Reserved:  1024.00 MB
  ├─ Cached:    1024.00 MB
  └─ Peak:      567.89 MB
CPU Memory:
  ├─ Used:      1345.67 MB
  └─ Percent:   13.4%
--------------------------------------------------

📸 Batch 1 - After negative embeddings
--------------------------------------------------
GPU Memory:
  ├─ Allocated: 890.12 MB  ← Or here!
  ├─ Reserved:  1024.00 MB
  ├─ Cached:    1024.00 MB
  └─ Peak:      890.12 MB
...
--------------------------------------------------

📸 Batch 1 - After prediction
--------------------------------------------------
...

📸 Batch 1 - After loss computation
--------------------------------------------------
   Loss: 0.6931

📸 Batch 1 - After backward pass
--------------------------------------------------
...

📸 Batch 1 - End (after cleanup)
--------------------------------------------------
   ⏱️  Batch time: 2.345s
======================================================================
```

## Key Memory Checkpoints Per Batch

1. **Start** - Baseline memory before batch processing
2. **After data loading** - Memory after loading batch data from disk
3. **After positive embeddings** - Memory after computing embeddings for positive edges (CRITICAL - often where spikes occur)
4. **After negative embeddings** - Memory after computing embeddings for negative edges (CRITICAL)
5. **After prediction** - Memory after link prediction forward pass
6. **After loss computation** - Memory after calculating loss (includes NaN/Inf checks)
7. **After backward pass** - Memory after gradient computation and optimizer step
8. **End (after cleanup)** - Final memory state after memory bank detachment

## NaN/Inf Detection

Added automatic detection for NaN/Inf values in predictions:
```
❌ NaN/Inf detected in predictions!
   Positive probs - min: 0.0000, max: nan
   Negative probs - min: 0.0000, max: 1.0000
```

## Benefits for Debugging

### 1. **Identify Memory Spikes**
You can see exactly which operation causes memory to spike:
- Is it during positive embeddings? → Check encoder input sizes
- Is it during negative embeddings? → Check negative sampling logic
- Is it during backward pass? → Check gradient accumulation

### 2. **Track Memory Growth Across Batches**
Compare "Start" memory across batches:
- Batch 1 Start: 234 MB
- Batch 2 Start: 256 MB ← Memory not being freed properly!
- Batch 3 Start: 278 MB ← Gradual leak!

### 3. **Detect NaN/Inf Early**
Catches NaN/Inf right after they occur, showing which embeddings produced them.

### 4. **Compare Different Configurations**
Run with different encoders and compare memory patterns:
- LeTE: Peak at 890 MB after positive embeddings
- Time2Vec: Peak at 456 MB after positive embeddings
- KAN-MAMMOTE: Peak at 1234 MB after positive embeddings ← Needs optimization!

## Files Modified

- `experiments/train_link_prediction_estimate.py`:
  - Added memory snapshots at 8 checkpoints per batch
  - Added NaN/Inf detection after predictions
  - Added batch time logging
  - Removed end-of-run summary (since per-batch is more useful)

## CSV Export Still Available

The detailed per-operation memory profiling data is still exported to CSV:
```
./time_estimates/{model}_{encoder}_{dataset}_dr{ratio}_memory.csv
```

This contains:
- Component name (e.g., "TGAT_positive_embeddings")
- Call number
- GPU allocated/reserved/cached MB
- CPU used MB and percent
- Peak GPU MB

## Usage

Run the benchmark script as before:
```bash
python experiments/train_link_prediction_estimate.py \
    --model_name TGAT \
    --time_encoder_type lete \
    --dataset_name wikipedia \
    --data_ratio 0.01 \
    --batch_size 200
```

Now you'll see detailed memory tracking for each batch, making it much easier to:
1. Identify which operation causes OOM (Out Of Memory)
2. Detect which batch triggers BCELoss assertion errors
3. Track memory leaks across batches
4. Compare memory usage between different encoders

## Troubleshooting

### If you see memory spike at "After positive embeddings":
- Check time encoder output dimensions
- Verify neighbor sampling isn't loading too many neighbors
- Check if time encoder has memory-efficient implementation

### If you see gradual memory growth across batches:
- Memory leak in model (not freeing intermediate tensors)
- Check `detach_memory_bank()` is being called for memory-based models
- Verify no global variables accumulating tensors

### If you see NaN/Inf in predictions:
- Check encoder initialization (LeTE Spline weights issue)
- Verify time encoding values are in reasonable range
- Check for division by zero in custom modules
