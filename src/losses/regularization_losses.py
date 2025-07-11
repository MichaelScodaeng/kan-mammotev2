# kan_mamote/src/losses/regularization_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Import KAN-MAMOTE specific modules to access parameters
from src.models.kan_mammote import KAN_MAMOTE_Model
from src.models.k_mote import K_MOTE
from src.layers.kan_base_layer import KANLayer
from src.layers.basis_functions import (
    FourierBasis, SplineBasis, GaussianKernelBasis, WaveletBasis
)

def sobolev_l2_loss(model: KAN_MAMOTE_Model, lambda_sobolev: float, timestamps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Sobolev L2 regularization loss for the KAN experts.
    Penalizes the L2 norm of the first and/or second derivatives of the expert's
    transformation functions to encourage smoothness and generalization.

    Args:
        model: The KAN_MAMOTE_Model instance.
        lambda_sobolev: The regularization strength.
        timestamps: Actual batch timestamps, shape (batch_size, seq_len, 1) or (batch_size, 1).

    Returns:
        torch.Tensor: Scalar Sobolev L2 loss.
    """
    if lambda_sobolev == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    total_sobolev_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    num_kan_layers = 0

    # Flatten timestamps for processing: (batch_size * seq_len, 1) or (batch_size, 1)
    if timestamps.dim() == 3:
        # Shape: (batch_size, seq_len, 1) -> (batch_size * seq_len, 1)
        timestamps_flat = timestamps.view(-1, 1)
    else:
        # Shape: (batch_size, 1)
        timestamps_flat = timestamps
    
    timestamps_flat = timestamps_flat.detach().requires_grad_(True)

    # Iterate through all KANLayer instances within the K-MOTE modules
    for k_mote_module in [model.k_mote_abs, model.k_mote_rel]:
        for expert_name in k_mote_module.expert_names:
            kan_layer = k_mote_module.experts[expert_name]

            try:
                # Forward pass through the KAN layer using actual batch timestamps
                output = kan_layer(timestamps_flat)  # (batch_size, D_time_per_expert)
                
                # Compute first derivatives with respect to timestamps
                if output.requires_grad:
                    # Sum over output dimensions for gradient computation
                    output_sum = output.sum()
                    
                    first_deriv = torch.autograd.grad(
                        outputs=output_sum,
                        inputs=timestamps_flat,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if first_deriv is not None:
                        # L2 norm of first derivative
                        total_sobolev_loss += torch.mean(first_deriv**2)
                        
                        # Second derivative
                        try:
                            second_deriv = torch.autograd.grad(
                                outputs=first_deriv.sum(),
                                inputs=timestamps_flat,
                                create_graph=False,
                                retain_graph=False,
                                allow_unused=True
                            )[0]
                            
                            if second_deriv is not None:
                                total_sobolev_loss += torch.mean(second_deriv**2)
                        except:
                            pass  # Skip second derivative if computation fails
                        
                num_kan_layers += 1
                
            except Exception as e:
                # Skip this layer if forward pass fails
                print(f"Warning: Skipping Sobolev loss for {expert_name}: {e}")
                continue
    
    if num_kan_layers == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    # Average over all KAN layers and apply regularization strength
    return lambda_sobolev * (total_sobolev_loss / num_kan_layers)


def total_variation_l1_loss(model: KAN_MAMOTE_Model, lambda_tv: float, timestamps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Total Variation L1 regularization loss for the KAN experts.
    Penalizes the L1 norm of the function's gradient to encourage piecewise
    constant/linear solutions and sharp, interpretable transitions.

    Args:
        model: The KAN_MAMOTE_Model instance.
        lambda_tv: The regularization strength.
        timestamps: Actual batch timestamps, shape (batch_size, seq_len, 1) or (batch_size, 1).

    Returns:
        torch.Tensor: Scalar Total Variation L1 loss.
    """
    if lambda_tv == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    total_tv_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    num_kan_layers = 0

    # Flatten timestamps for processing: (batch_size * seq_len, 1) or (batch_size, 1)
    if timestamps.dim() == 3:
        # Shape: (batch_size, seq_len, 1) -> (batch_size * seq_len, 1)
        timestamps_flat = timestamps.view(-1, 1)
    else:
        # Shape: (batch_size, 1)
        timestamps_flat = timestamps
    
    timestamps_flat = timestamps_flat.detach().requires_grad_(True)

    # Iterate through all KANLayer instances within the K-MOTE modules
    for k_mote_module in [model.k_mote_abs, model.k_mote_rel]:
        for expert_name in k_mote_module.expert_names:
            kan_layer = k_mote_module.experts[expert_name]

            try:
                # Forward pass through the KAN layer using actual batch timestamps
                output = kan_layer(timestamps_flat)  # (batch_size, D_time_per_expert)
                
                # Compute first derivatives with respect to timestamps
                if output.requires_grad:
                    # Sum over output dimensions for gradient computation
                    output_sum = output.sum()
                    
                    first_deriv = torch.autograd.grad(
                        outputs=output_sum,
                        inputs=timestamps_flat,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True
                    )[0]
                    
                    if first_deriv is not None:
                        # L1 norm of first derivative
                        total_tv_loss += torch.mean(torch.abs(first_deriv))
                        
                num_kan_layers += 1
                
            except Exception as e:
                # Skip this layer if forward pass fails
                print(f"Warning: Skipping TV loss for {expert_name}: {e}")
                continue

    if num_kan_layers == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    # Average over all KAN layers and apply regularization strength
    return lambda_tv * (total_tv_loss / num_kan_layers)


def moe_load_balancing_loss(expert_weights_abs: torch.Tensor, expert_weights_rel: torch.Tensor, lambda_moe: float) -> torch.Tensor:
    """
    Calculates the MoE load balancing loss.
    Encourages a balanced distribution of workload across experts by penalizing
    disparities in expert usage (to prevent expert collapse).

    Uses a simple and effective formulation: sum of squared mean expert probabilities.
    This penalizes experts that receive very low average weights and encourages uniform usage.

    Args:
        expert_weights_abs: Raw expert weights from k_mote_abs, shape (batch_size, seq_len, num_experts).
        expert_weights_rel: Raw expert weights from k_mote_rel, shape (batch_size, seq_len, num_experts).
        lambda_moe: The regularization strength.

    Returns:
        torch.Tensor: Scalar MoE load balancing loss.
    """
    if lambda_moe == 0:
        return torch.tensor(0.0, device=expert_weights_abs.device)

    # Combine weights from both K-MOTE modules for total load.
    # Stack them along a new dimension to treat them as separate tasks for load balancing.
    # (batch_size, seq_len, 2, num_experts)
    all_expert_weights = torch.stack([expert_weights_abs, expert_weights_rel], dim=2)
    
    # Average expert weights across batch and sequence length
    # (num_experts) - average probability of selection for each expert
    mean_expert_prob_across_tasks = all_expert_weights.mean(dim=(0, 1, 2)) 
    
    # Simple and effective MoE load balancing loss: sum of squared mean probabilities
    # This encourages uniform distribution across experts
    load_balance_loss = (mean_expert_prob_across_tasks ** 2).sum()

    return lambda_moe * load_balance_loss

# Add a combined loss function for convenience in train.py
def calculate_total_loss(
    main_task_loss: torch.Tensor,
    model: KAN_MAMOTE_Model,
    moe_losses_info: Tuple[torch.Tensor, torch.Tensor],
    timestamps: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Calculates the total loss including main task loss and regularization terms.

    Args:
        main_task_loss: The primary loss from the downstream task (e.g., MSE, CrossEntropy).
        model: The KAN_MAMOTE_Model instance.
        moe_losses_info: Tuple containing (abs_expert_weights_for_loss, rel_expert_weights_for_loss).
        timestamps: Actual batch timestamps for regularization, shape (batch_size, seq_len, 1) or (batch_size, 1).

    Returns:
        Tuple[torch.Tensor, dict]: The sum of all loss components and a dict with individual losses.
    """
    config = model.config
    
    # Regularization Losses using actual batch data
    sobolev_loss = sobolev_l2_loss(model, config.lambda_sobolev_l2, timestamps)
    tv_loss = total_variation_l1_loss(model, config.lambda_total_variation_l1, timestamps)
    
    abs_expert_weights, rel_expert_weights = moe_losses_info
    moe_loss = moe_load_balancing_loss(abs_expert_weights, rel_expert_weights, config.lambda_moe_load_balancing)

    total_loss = main_task_loss + sobolev_loss + tv_loss + moe_loss

    # For debugging/logging, you might want to return a dictionary of losses as well
    return total_loss, {
        "main_task_loss": main_task_loss.item(),
        "sobolev_loss": sobolev_loss.item(),
        "tv_loss": tv_loss.item(),
        "moe_loss": moe_loss.item(),
        "total_loss": total_loss.item()
    }