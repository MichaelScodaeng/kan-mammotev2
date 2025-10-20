#!/bin/bash
#
# Generate Comprehensive Results Table
# =====================================
# This script generates a long-format table with ALL results across:
# - All models, datasets, encoders
# - All negative sampling strategies (random, historical, inductive)
#
# Output: comprehensive_results_table.csv with columns:
#   Model | Dataset | Encoder | Strategy | Test_AP | Test_AUC | New_Node_AP | ...
#

set -e

echo "=========================================="
echo "COMPREHENSIVE RESULTS TABLE GENERATION"
echo "=========================================="
echo ""
echo "This will scan all saved_results/ files and create:"
echo "  1. comprehensive_results_table.csv (long format)"
echo "  2. test_ap_pivot_table.csv (pivot view)"
echo ""
echo "File patterns recognized:"
echo "  - Random: {model}_{encoder}_seed0_{timestamp}.json"
echo "  - Historical: historical_negative_sampling_{model}_{encoder}_seed0.json"
echo "  - Inductive: inductive_negative_sampling_{model}_{encoder}_seed0.json"
echo ""
echo "Output directory: completion_analysis/"
echo ""
echo "Press Enter to continue, Ctrl+C to abort..."
read

# Run the comprehensive table generation
python check_results_exist.py \
  --comprehensive_table \
  --output_dir completion_analysis

echo ""
echo "=========================================="
echo "✅ GENERATION COMPLETE!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  📊 completion_analysis/comprehensive_results_table.csv"
echo "  📊 completion_analysis/test_ap_pivot_table.csv"
echo ""
echo "Usage examples:"
echo ""
echo "  # View in terminal (first 20 rows)"
echo "  head -20 completion_analysis/comprehensive_results_table.csv | column -t -s,"
echo ""
echo "  # Count results per strategy"
echo "  cut -d',' -f4 completion_analysis/comprehensive_results_table.csv | sort | uniq -c"
echo ""
echo "  # Filter for specific model (JODIE)"
echo "  grep '^JODIE,' completion_analysis/comprehensive_results_table.csv"
echo ""
echo "  # Filter for specific dataset (wikipedia)"
echo "  grep ',wikipedia,' completion_analysis/comprehensive_results_table.csv"
echo ""
echo "  # Compare strategies for JODIE+wikipedia+time2vec"
echo "  grep 'JODIE,wikipedia,time2vec,' completion_analysis/comprehensive_results_table.csv"
echo ""
echo "=========================================="
