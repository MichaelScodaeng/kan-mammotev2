# KMM

Code for the paper  
**"KMM: Learnable Dual-Stream Time Encoding for Continuous-Time Dynamic Graphs"**  
Submitted to the Main Track of **PAKDD 2026**.

This repository contains the implementation of KMM, a learnable dual-stream time encoder for continuous-time dynamic graphs (CTDGs). The code supports two main experiment settings:

1. **Dynamic link prediction** on several CTDG benchmark datasets  
2. **Next-event prediction** on the Stack Overflow badge sequence dataset

---

## Link Prediction in CTDGs

To run the unified link prediction experiment, use the following command:

```
python experiment_unified.py \
    --single_encoder KMM \
    --models TGN \
    --datasets uci \
    --num_epochs 200
```

## CTDG datasets Download

The CTDG datasets used in our experiments can be downloaded [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o).

After downloading and unzipping, place the dataset folder inside: 
```
processed_data/
```

## Next-Event Prediction on Stack Overflow

To run the Stack Overflow badge prediction experiment:
```
python experiments/stackoverflow_badge_prediction.py \
    --encoders KMM \
    --epochs 50
```

##
