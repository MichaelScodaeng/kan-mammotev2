import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the CombinedLeTE model from the LeTE.py file
try:
    from LeTE import CombinedLeTE
except ImportError:
    print("Error: Could not import CombinedLeTE. Make sure 'LeTE.py' is in the same directory.")
    sys.exit(1)

class LeTERegressor(nn.Module):
    """
    A wrapper to use CombinedLeTE for a scalar regression task.
    """
    def __init__(self, embedding_dim=64, p=0.5):
        super().__init__()
        self.time_encoder = CombinedLeTE(dim=embedding_dim, p=p)
        self.output_head = nn.Linear(embedding_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1: 
            t = t.unsqueeze(0)
        embeddings = self.time_encoder(t)
        output = self.output_head(embeddings)
        return output

def train_and_debug_model(model, t_data, y_true, model_name, epochs=100, lr=2e-4):
    """Train model with debugging information"""
    print(f"\n=== Debugging {model_name} ===")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    
    # Ensure proper input shapes
    if t_data.dim() == 1: 
        t_data = t_data.unsqueeze(0)
    if y_true.dim() == 1: 
        y_true = y_true.unsqueeze(0).unsqueeze(-1)
    
    print(f"Input shape: {t_data.shape}")
    print(f"Target shape: {y_true.shape}")
    
    # Check initial forward pass
    model.eval()
    with torch.no_grad():
        try:
            initial_output = model(t_data)
            print(f"Initial output shape: {initial_output.shape}")
            print(f"Initial output range: [{initial_output.min().item():.6f}, {initial_output.max().item():.6f}]")
            print(f"Initial output has NaN: {torch.isnan(initial_output).any().item()}")
            print(f"Initial output has Inf: {torch.isinf(initial_output).any().item()}")
        except Exception as e:
            print(f"ERROR in initial forward pass: {e}")
            return float('inf'), None

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        try:
            y_pred = model(t_data)
            loss = loss_fn(y_pred, y_true)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN/Inf loss at epoch {epoch+1}")
                print(f"    y_pred range: [{y_pred.min().item():.6f}, {y_pred.max().item():.6f}]")
                return float('inf'), None
            
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    total_grad_norm += grad_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: Loss = {loss.item():.6f}, Grad norm = {total_grad_norm:.6f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
        except Exception as e:
            print(f"    ERROR at epoch {epoch}: {e}")
            return float('inf'), None
    
    final_loss = losses[-1] if losses else float('inf')
    print(f"Final Loss: {final_loss:.6f}")
    
    # Final forward pass check
    model.eval()
    with torch.no_grad():
        final_output = model(t_data)
        print(f"Final output range: [{final_output.min().item():.6f}, {final_output.max().item():.6f}]")
        print(f"Final output has NaN: {torch.isnan(final_output).any().item()}")
        print(f"Final output has Inf: {torch.isinf(final_output).any().item()}")
    
    return final_loss, losses

def debug_spline_issue():
    print("=== Debugging Spline-based LeTE Issue ===")
    
    # Test on the swish function (where the issue is most visible)
    t = torch.linspace(-5, 5, 500)
    y_true = t * torch.sigmoid(t)  # Swish function
    
    # Normalize
    y_mean, y_std = y_true.mean(), y_true.std()
    y_norm = (y_true - y_mean) / y_std
    
    print(f"Target function range: [{y_true.min().item():.6f}, {y_true.max().item():.6f}]")
    print(f"Normalized target range: [{y_norm.min().item():.6f}, {y_norm.max().item():.6f}]")
    
    # Test different p values
    p_values = [1.0, 0.5, 0.0]  # Fourier, Combined, Spline
    names = ["Pure Fourier (p=1.0)", "Combined (p=0.5)", "Pure Spline (p=0.0)"]
    
    results = {}
    
    for p, name in zip(p_values, names):
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")
        
        model = LeTERegressor(embedding_dim=64, p=p)
        
        # Debug the LeTE encoder specifically
        print(f"\nLeTE encoder config:")
        print(f"  dim_fourier: {model.time_encoder.dim_fourier}")
        print(f"  dim_spline: {model.time_encoder.dim_spline}")
        print(f"  layer_norm: {model.time_encoder.layer_norm}")
        print(f"  scale: {model.time_encoder.scale}")
        
        final_loss, loss_history = train_and_debug_model(model, t, y_norm, name)
        results[name] = {'model': model, 'loss': final_loss, 'loss_history': loss_history}
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss curves
    ax = axes[0, 0]
    for name, result in results.items():
        if result['loss_history'] is not None:
            ax.plot(result['loss_history'], label=name)
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot predictions
    for i, (name, result) in enumerate(results.items()):
        ax = axes[0, 1] if i == 0 else axes[1, 0] if i == 1 else axes[1, 1]
        
        if result['model'] is not None:
            model = result['model']
            model.eval()
            with torch.no_grad():
                t_input = t.unsqueeze(0)
                y_pred_norm = model(t_input).squeeze()
                y_pred = y_pred_norm * y_std + y_mean  # Denormalize
                
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', label='Target', linewidth=2)
            ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), 'r--', label='Prediction', linewidth=2)
            ax.set_title(f'{name}\nFinal Loss: {result["loss"]:.6f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spline_debug_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, result in results.items():
        print(f"{name}: Final Loss = {result['loss']:.6f}")

if __name__ == '__main__':
    debug_spline_issue()