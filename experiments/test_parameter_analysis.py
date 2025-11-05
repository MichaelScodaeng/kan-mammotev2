#!/usr/bin/env python3
"""
Quick test of KAN-MAMMOTE Parameter Analysis Framework
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_parameter_analysis():
    """Test the parameter analysis framework"""
    print("🧪 Testing KAN-MAMMOTE Parameter Analysis Framework")
    
    try:
        from experiments.kan_mammote_parameter_analysis_v2 import KANMAMMOTEParameterAnalyzer
        
        # Create analyzer with test output directory
        analyzer = KANMAMMOTEParameterAnalyzer(output_dir='test_parameter_analysis')
        
        print(f"✅ Successfully created analyzer")
        print(f"   Parameter combinations: {len(analyzer.param_combinations)}")
        print(f"   Fixed config: {analyzer.fixed_config}")
        
        # Show some example parameter combinations
        print(f"\n📋 Example parameter combinations:")
        for i, combo in enumerate(analyzer.param_combinations[:5]):
            print(f"   {i+1}: {combo}")
        
        # Test FLOPs calculation
        from experiments.kan_mammote_parameter_analysis_v2 import calculate_kan_mammote_flops
        
        test_config = {'expert_dim': 128, 'mamba_d_state': 128, 'mamba_headdim': 32, 'n_layers': 2}
        flops = calculate_kan_mammote_flops(test_config)
        
        print(f"\n⚡ FLOPs calculation test:")
        print(f"   Config: {test_config}")
        print(f"   Total FLOPs: {flops['total']/1e9:.2f} GFLOPs")
        print(f"   Breakdown: {flops}")
        
        # Run test analysis (just first few configurations)
        print(f"\n🚀 Running test analysis...")
        analyzer.run_analysis(test_mode=True)
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Check results in: test_parameter_analysis/")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_analysis()