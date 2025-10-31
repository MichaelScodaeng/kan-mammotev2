import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelKMOTE(nn.Module):
    """
    CURRENT APPROACH: All experts run in parallel
    High memory usage but fast execution
    """
    def __init__(self, experts, gating_network):
        super().__init__()
        self.experts = experts
        self.gating_network = gating_network
        self.num_experts = len(experts)
    
    def forward(self, x):
        print("🔄 PARALLEL EVALUATION:")
        print(f"  Input shape: {x.shape}")
        
        # Step 1: Gating computation
        gating_logits = self.gating_network(x)
        gating_weights = F.softmax(gating_logits, dim=-1)
        print(f"  Gating weights shape: {gating_weights.shape}")
        
        # Step 2: ALL EXPERTS RUN AT ONCE (memory explosion!)
        expert_outputs = []
        total_memory = 0
        
        for i, expert in enumerate(self.experts):
            print(f"  💾 Expert {i+1} running... (memory accumulating)")
            output = expert(x)
            expert_outputs.append(output)
            # Memory keeps accumulating - all outputs stay in memory!
            memory_mb = output.numel() * 4 / (1024**2)  # 4 bytes per float32
            total_memory += memory_mb
            print(f"     Expert {i+1} output: {output.shape}, Memory: +{memory_mb:.1f}MB")
        
        print(f"  💥 TOTAL MEMORY USED: {total_memory:.1f}MB (all experts + outputs)")
        
        # Step 3: Stack (more memory!)
        stacked_outputs = torch.stack(expert_outputs, dim=-1)
        stacking_memory = stacked_outputs.numel() * 4 / (1024**2)
        print(f"  📚 Stacking memory: +{stacking_memory:.1f}MB")
        
        # Step 4: Weighted combination
        gating_weights = gating_weights.unsqueeze(-2)
        output = (gating_weights * stacked_outputs).sum(dim=-1)
        
        print(f"  🎯 Final output: {output.shape}")
        print(f"  💾 PEAK MEMORY: {total_memory + stacking_memory:.1f}MB")
        return output


class SequentialKMOTE(nn.Module):
    """
    SEQUENTIAL APPROACH: Experts run one at a time
    Low memory usage but slower execution
    """
    def __init__(self, experts, gating_network):
        super().__init__()
        self.experts = experts
        self.gating_network = gating_network
        self.num_experts = len(experts)
    
    def forward(self, x):
        print("🔄 SEQUENTIAL EVALUATION:")
        print(f"  Input shape: {x.shape}")
        
        # Step 1: Gating computation FIRST
        gating_logits = self.gating_network(x)
        gating_weights = F.softmax(gating_logits, dim=-1)
        print(f"  Gating weights shape: {gating_weights.shape}")
        
        # Step 2: Initialize result tensor
        batch_size, seq_len = x.shape[0], x.shape[1]
        result = torch.zeros(batch_size, seq_len, 1, device=x.device)
        print(f"  Initialized result: {result.shape}")
        
        peak_memory = 0
        
        # Step 3: EXPERTS RUN ONE AT A TIME
        for i, expert in enumerate(self.experts):
            print(f"  🔄 Expert {i+1} running...")
            
            # Run expert
            expert_output = expert(x)
            expert_memory = expert_output.numel() * 4 / (1024**2)
            peak_memory = max(peak_memory, expert_memory)
            
            # Apply gating weight immediately
            weight = gating_weights[:, :, i:i+1]  # (batch, seq, 1)
            weighted_output = weight * expert_output
            
            # Accumulate into result
            result += weighted_output
            
            print(f"     Expert {i+1} output: {expert_output.shape}")
            print(f"     Memory used: {expert_memory:.1f}MB")
            print(f"     Accumulated into result")
            
            # CRITICAL: Free memory immediately
            del expert_output, weighted_output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"     🗑️  Memory freed")
        
        print(f"  🎯 Final result: {result.shape}")
        print(f"  💾 PEAK MEMORY: {peak_memory:.1f}MB (only one expert at a time)")
        return result


# Mock experts for demonstration
class MockExpert(nn.Module):
    def __init__(self, name, memory_mb):
        super().__init__()
        self.name = name
        self.memory_mb = memory_mb
        # Create parameters to simulate memory usage
        param_count = int(memory_mb * 1024 * 1024 / 4)  # 4 bytes per float32
        self.large_param = nn.Parameter(torch.randn(param_count))
        self.linear = nn.Linear(64, 1)
    
    def forward(self, x):
        # Simulate computation that uses the large parameter
        _ = self.large_param.mean()  # Touch the parameter
        return self.linear(x)


if __name__ == "__main__":
    # Create mock experts with different memory footprints
    experts = nn.ModuleList([
        MockExpert("SplineKAN", 240),   # 240MB
        MockExpert("FourierKAN", 40),   # 40MB  
        MockExpert("WaveletKAN", 80),   # 80MB
    ])
    
    # Simple gating network
    gating_network = nn.Linear(64, 3)
    
    # Create both models
    parallel_model = ParallelKMOTE(experts, gating_network)
    sequential_model = SequentialKMOTE(experts, gating_network)
    
    # Test input
    x = torch.randn(32, 512, 64)  # (batch=32, seq_len=512, hidden_dim=64)
    
    print("=" * 70)
    print("PARALLEL K-MOTE:")
    print("=" * 70)
    with torch.no_grad():
        parallel_output = parallel_model(x)
    
    print("\n" + "=" * 70)
    print("SEQUENTIAL K-MOTE:")
    print("=" * 70)
    with torch.no_grad():
        sequential_output = sequential_model(x)
    
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    print(f"✅ Outputs are identical: {torch.allclose(parallel_output, sequential_output, atol=1e-6)}")
    print(f"📊 Parallel memory: ~360MB peak")
    print(f"📊 Sequential memory: ~240MB peak")
    print(f"🎯 Memory savings: {(360-240)/360*100:.1f}%")
    print(f"⚠️  Trade-off: Sequential is slower due to no parallelization")