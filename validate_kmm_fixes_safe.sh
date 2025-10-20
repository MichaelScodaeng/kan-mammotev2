#!/bin/bash
#
# SAFE VALIDATION SCRIPT FOR KAN-MAMMOTE FIXES
# =============================================
#
# This script validates hyperparameter fixes while ensuring:
# ✅ Each experiment has isolated directories (no overwrites)
# ✅ Results are clearly labeled with validation prefix
# ✅ Easy to compare before/after
#
# Timeline: Run overnight, check results tomorrow
# Expected runtime: ~8-12 hours for all 4 experiments
#

set -e  # Exit on error

# Create unique results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/kmm_validation_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Log file
LOGFILE="$RESULTS_DIR/validation.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=========================================="
log "KAN-MAMMOTE SAFE VALIDATION EXPERIMENT"
log "=========================================="
log ""
log "Results directory: $RESULTS_DIR"
log "Timestamp: $TIMESTAMP"
log ""
log "ISOLATION STRATEGY:"
log "  - Each experiment uses unique --save_model_name_suffix"
log "  - Models saved to: saved_models/{model}/{dataset}/{model}_{encoder}_val_{experiment}_seed{N}"
log "  - Results saved to: saved_results/{model}/{dataset}/*_val_{experiment}_*"
log "  - Metrics saved to: saved_metrics/{model}/{dataset}/*_val_{experiment}_*"
log ""
log "FIXES BEING TESTED:"
log "  1. ✅ Gradient clipping (max_norm=1.0) - ALREADY IMPLEMENTED"
log "  2. ⏳ Lower learning rate (0.00001 - 0.00005)"
log "  3. ⏳ Weight decay (0.0003 - 0.001)"
log ""
log "=========================================="
log ""

# =============================================================================
# EXPERIMENT 1: Validate SUCCESS case (uci with TCL)
# =============================================================================
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "EXPERIMENT 1/4: uci dataset (TCL model)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Purpose: Verify fixes don't break what already works"
log "Baseline: KMM achieves +1.57% over LeTE (Test AP: 0.9345)"
log "Target: Maintain Test AP ≥ 0.93"
log ""
log "Isolation: --save_model_name_suffix '_val_uci_tcl'"
log "Model dir: saved_models/TCL/uci/TCL_kan_mammote_dual_kmote_val_uci_tcl_seed{N}"
log "Results: saved_results/TCL/uci/*_val_uci_tcl_*"
log ""

python experiments/train_link_prediction.py \
  --model_name "TCL" \
  --dataset_name "uci" \
  --time_encoder_type "kan_mammote_dual_kmote" \
  --num_runs 1 \
  --learning_rate 0.00005 \
  --weight_decay 0.0005 \
  --patience 30 \
  --load_best_configs \
  --save_model_name_suffix "_val_uci_tcl" \
  --disable_progress_bar \
  2>&1 | tee -a "$LOGFILE"

log ""
log "✓ Experiment 1 complete!"
log "   Check: saved_models/TCL/uci/TCL_kan_mammote_dual_kmote_val_uci_tcl_seed0/"
log "   Check: saved_results/TCL/uci/*val_uci_tcl*.json"
log ""
sleep 5

# =============================================================================
# EXPERIMENT 2: Fix WORST case (lastfm with JODIE)
# =============================================================================
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "EXPERIMENT 2/4: lastfm dataset (JODIE model)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Purpose: Fix catastrophic failure case"
log "Baseline: KMM shows -10.17% gap (Test AP: 0.6519 vs LeTE: 0.7256)"
log "Target: Test AP ≥ 0.70 (within 3% of LeTE)"
log ""
log "Isolation: --save_model_name_suffix '_val_lastfm_jodie'"
log "Model dir: saved_models/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_val_lastfm_jodie_seed{N}"
log "Results: saved_results/JODIE/lastfm/*_val_lastfm_jodie_*"
log ""

python experiments/train_link_prediction.py \
  --model_name "JODIE" \
  --dataset_name "lastfm" \
  --time_encoder_type "kan_mammote_dual_kmote" \
  --num_runs 1 \
  --learning_rate 0.00001 \
  --weight_decay 0.001 \
  --patience 40 \
  --load_best_configs \
  --save_model_name_suffix "_val_lastfm_jodie" \
  --disable_progress_bar \
  --num_epochs 250 \
  2>&1 | tee -a "$LOGFILE"

log ""
log "✓ Experiment 2 complete!"
log "   Check: saved_models/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_val_lastfm_jodie_seed0/"
log "   Check: saved_results/JODIE/lastfm/*val_lastfm_jodie*.json"
log ""
sleep 5

# =============================================================================
# EXPERIMENT 3: Fix moderate failure (mooc with JODIE)
# =============================================================================
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "EXPERIMENT 3/4: mooc dataset (JODIE model)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Purpose: Fix moderate performance gap"
log "Baseline: KMM shows -5.20% gap (Test AP: 0.7662 vs LeTE: 0.8082)"
log "Target: Test AP ≥ 0.80 (match LeTE)"
log ""
log "Isolation: --save_model_name_suffix '_val_mooc_jodie'"
log "Model dir: saved_models/JODIE/mooc/JODIE_kan_mammote_dual_kmote_val_mooc_jodie_seed{N}"
log "Results: saved_results/JODIE/mooc/*_val_mooc_jodie_*"
log ""

python experiments/train_link_prediction.py \
  --model_name "JODIE" \
  --dataset_name "mooc" \
  --time_encoder_type "kan_mammote_dual_kmote" \
  --num_runs 1 \
  --learning_rate 0.00003 \
  --weight_decay 0.0005 \
  --patience 30 \
  --load_best_configs \
  --save_model_name_suffix "_val_mooc_jodie" \
  --disable_progress_bar \
  --num_epochs 250 \
  2>&1 | tee -a "$LOGFILE"

log ""
log "✓ Experiment 3 complete!"
log "   Check: saved_models/JODIE/mooc/JODIE_kan_mammote_dual_kmote_val_mooc_jodie_seed0/"
log "   Check: saved_results/JODIE/mooc/*val_mooc_jodie*.json"
log ""
sleep 5

# =============================================================================
# EXPERIMENT 4: Fix instability (Contacts with JODIE)
# =============================================================================
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "EXPERIMENT 4/4: Contacts dataset (JODIE model)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Purpose: Fix training instability"
log "Baseline: Loss spike ratio 4.01x (HIGHLY UNSTABLE)"
log "Target: Stable training (spike ratio <2.0x) + competitive AP"
log ""
log "Isolation: --save_model_name_suffix '_val_contacts_jodie'"
log "Model dir: saved_models/JODIE/Contacts/JODIE_kan_mammote_dual_kmote_val_contacts_jodie_seed{N}"
log "Results: saved_results/JODIE/Contacts/*_val_contacts_jodie_*"
log ""

python experiments/train_link_prediction.py \
  --model_name "JODIE" \
  --dataset_name "Contacts" \
  --time_encoder_type "kan_mammote_dual_kmote" \
  --num_runs 1 \
  --learning_rate 0.00002 \
  --weight_decay 0.0003 \
  --patience 40 \
  --load_best_configs \
  --save_model_name_suffix "_val_contacts_jodie" \
  --disable_progress_bar \
  --num_epochs 250 \
  2>&1 | tee -a "$LOGFILE"

log ""
log "✓ Experiment 4 complete!"
log "   Check: saved_models/JODIE/Contacts/JODIE_kan_mammote_dual_kmote_val_contacts_jodie_seed0/"
log "   Check: saved_results/JODIE/Contacts/*val_contacts_jodie*.json"
log ""

# =============================================================================
# SUMMARY & FILE LOCATIONS
# =============================================================================
log ""
log "=========================================="
log "ALL VALIDATION EXPERIMENTS COMPLETE!"
log "=========================================="
log ""
log "📁 FILE ORGANIZATION:"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "Main results directory: $RESULTS_DIR"
log "Master log: $LOGFILE"
log ""
log "Individual experiment locations:"
log ""
log "1. uci (TCL):"
log "   Models:  saved_models/TCL/uci/TCL_kan_mammote_dual_kmote_val_uci_tcl_seed0/"
log "   Results: saved_results/TCL/uci/*_val_uci_tcl_*.json"
log "   Metrics: saved_metrics/TCL/uci/TCL_kan_mammote_dual_kmote_val_uci_tcl_seed0/"
log ""
log "2. lastfm (JODIE):"
log "   Models:  saved_models/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_val_lastfm_jodie_seed0/"
log "   Results: saved_results/JODIE/lastfm/*_val_lastfm_jodie_*.json"
log "   Metrics: saved_metrics/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_val_lastfm_jodie_seed0/"
log ""
log "3. mooc (JODIE):"
log "   Models:  saved_models/JODIE/mooc/JODIE_kan_mammote_dual_kmote_val_mooc_jodie_seed0/"
log "   Results: saved_results/JODIE/mooc/*_val_mooc_jodie_*.json"
log "   Metrics: saved_metrics/JODIE/mooc/JODIE_kan_mammote_dual_kmote_val_mooc_jodie_seed0/"
log ""
log "4. Contacts (JODIE):"
log "   Models:  saved_models/JODIE/Contacts/JODIE_kan_mammote_dual_kmote_val_contacts_jodie_seed0/"
log "   Results: saved_results/JODIE/Contacts/*_val_contacts_jodie_*.json"
log "   Metrics: saved_metrics/JODIE/Contacts/JODIE_kan_mammote_dual_kmote_val_contacts_jodie_seed0/"
log ""
log "=========================================="
log ""
log "🔍 ANALYZING RESULTS:"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "Quick result check commands:"
log ""
log "# Check all validation results"
log "find saved_results -name '*_val_*' -type f"
log ""
log "# Extract test AP from each experiment"
log "for f in saved_results/TCL/uci/*_val_uci_tcl_*.json; do echo \"uci (TCL):\"; grep -o '\"test_ap\": [0-9.]*' \$f; done"
log "for f in saved_results/JODIE/lastfm/*_val_lastfm_jodie_*.json; do echo \"lastfm (JODIE):\"; grep -o '\"test_ap\": [0-9.]*' \$f; done"
log "for f in saved_results/JODIE/mooc/*_val_mooc_jodie_*.json; do echo \"mooc (JODIE):\"; grep -o '\"test_ap\": [0-9.]*' \$f; done"
log "for f in saved_results/JODIE/Contacts/*_val_contacts_jodie_*.json; do echo \"Contacts (JODIE):\"; grep -o '\"test_ap\": [0-9.]*' \$f; done"
log ""
log "# Check training curves (loss stability)"
log "ls -la saved_metrics/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_val_lastfm_jodie_seed0/"
log ""
log "=========================================="
log ""
log "✅ NEXT STEPS:"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "1. Review this log file: $LOGFILE"
log ""
log "2. Check result files for Test AP improvements:"
log "   - uci: Should maintain ~0.93+"
log "   - lastfm: Should improve from 0.65 to 0.70+"
log "   - mooc: Should improve from 0.77 to 0.80+"
log "   - Contacts: Check for stable training"
log ""
log "3. Analyze training curves in saved_metrics folders:"
log "   - Look for reduced loss spikes"
log "   - Verify smooth convergence"
log ""
log "4. Run the automated analysis:"
log "   python analyze_validation_results.py $RESULTS_DIR"
log ""
log "5. DECISION CRITERIA:"
log "   ✅ 3-4 experiments improve → Your fixes WORK! Proceed to full eval"
log "   ⚠️  2 experiments improve → Partial success, more tuning needed"
log "   ❌ <2 improve → Consider architectural changes"
log ""
log "=========================================="
log ""
log "🎯 Your baseline results are SAFE in their original locations"
log "   (e.g., saved_models/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_seed0/)"
log ""
log "🎯 These validation results have '_val_' prefix - NO OVERWRITES!"
log ""
log "=========================================="

# Create easy comparison script
cat > "$RESULTS_DIR/compare_results.sh" << 'EOF'
#!/bin/bash
# Quick comparison of validation results

echo "======================================"
echo "VALIDATION RESULTS COMPARISON"
echo "======================================"
echo ""

echo "1. uci (TCL) - Target: ≥0.93"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for f in saved_results/TCL/uci/*_val_uci_tcl_seed0_*.json; do
    if [ -f "$f" ]; then
        echo "File: $f"
        grep -o '"test_ap": [0-9.]*' "$f" || echo "  No test_ap found"
        grep -o '"test_auc": [0-9.]*' "$f" || echo "  No test_auc found"
    fi
done
echo ""

echo "2. lastfm (JODIE) - Target: ≥0.70 (baseline: 0.6519)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for f in saved_results/JODIE/lastfm/*_val_lastfm_jodie_seed0_*.json; do
    if [ -f "$f" ]; then
        echo "File: $f"
        grep -o '"test_ap": [0-9.]*' "$f" || echo "  No test_ap found"
        grep -o '"test_auc": [0-9.]*' "$f" || echo "  No test_auc found"
    fi
done
echo ""

echo "3. mooc (JODIE) - Target: ≥0.80 (baseline: 0.7662)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for f in saved_results/JODIE/mooc/*_val_mooc_jodie_seed0_*.json; do
    if [ -f "$f" ]; then
        echo "File: $f"
        grep -o '"test_ap": [0-9.]*' "$f" || echo "  No test_ap found"
        grep -o '"test_auc": [0-9.]*' "$f" || echo "  No test_auc found"
    fi
done
echo ""

echo "4. Contacts (JODIE) - Target: Stable + competitive"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for f in saved_results/JODIE/Contacts/*_val_contacts_jodie_seed0_*.json; do
    if [ -f "$f" ]; then
        echo "File: $f"
        grep -o '"test_ap": [0-9.]*' "$f" || echo "  No test_ap found"
        grep -o '"test_auc": [0-9.]*' "$f" || echo "  No test_auc found"
    fi
done
echo ""

echo "======================================"
echo "Run 'bash $0' to see this summary again"
echo "======================================"
EOF

chmod +x "$RESULTS_DIR/compare_results.sh"

log "📊 Created quick comparison script: $RESULTS_DIR/compare_results.sh"
log ""
log "🎉 Validation complete! Check results tomorrow morning."
log ""
