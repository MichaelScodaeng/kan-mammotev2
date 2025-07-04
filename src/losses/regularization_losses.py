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

def sobolev_l2_loss(model: KAN_MAMOTE_Model, lambda_sobolev: float) -> torch.Tensor:
    """
    Calculates the Sobolev L2 regularization loss for the KAN experts.
    Penalizes the L2 norm of the first and/or second derivatives of the expert's
    transformation functions to encourage smoothness and generalization.

    Args:
        model: The KAN_MAMOTE_Model instance.
        lambda_sobolev: The regularization strength.

    Returns:
        torch.Tensor: Scalar Sobolev L2 loss.
    """
    if lambda_sobolev == 0:
        return torch.tensor(0.0, device=model.config.device)

    total_sobolev_loss = torch.tensor(0.0, device=model.config.device)
    num_kan_layers = 0

    # Iterate through all KANLayer instances within the K-MOTE modules
    for k_mote_module in [model.k_mote_abs, model.k_mote_rel]:
        for expert_name in k_mote_module.expert_names:
            kan_layer = k_mote_module.experts[expert_name]
            basis_function = kan_layer.basis_function

            # We need to compute derivatives with respect to the input of the basis_function (x_prime)
            # x_prime is (batch_size, out_features)
            # We'll use a small fixed grid for approximation of the integral/sum of derivatives.
            
            # Create a dummy input for the basis function (represents x_prime range)
            # A range like [-5, 5] or determined by input transformation can be suitable.
            # Here, we'll assume a typical range for the transformed time input.
            dummy_x_prime = torch.linspace(-5.0, 5.0, 100).view(-1, 1).repeat(1, basis_function.output_dim).to(model.config.device)
            dummy_x_prime.requires_grad_(True)

            # --- First Derivative (L2 Norm) ---
            output = basis_function(dummy_x_prime) # (100, output_dim)
            
            # Compute first derivatives (d(output)/d(dummy_x_prime))
            # Gradients will be (100, output_dim)
            first_deriv = torch.autograd.grad(
                outputs=output,
                inputs=dummy_x_prime,
                grad_outputs=torch.ones_like(output),
                create_graph=True, # Important for computing second derivative
                retain_graph=True # Important for subsequent backward calls on the same graph
            )[0]
            
            # L2 norm of first derivative
            total_sobolev_loss += torch.mean(first_deriv**2)
            
            # --- Second Derivative (L2 Norm) ---
            # Only compute if the first derivative exists
            if first_deriv is not None:
                second_deriv = torch.autograd.grad(
                    outputs=first_deriv,
                    inputs=dummy_x_prime,
                    grad_outputs=torch.ones_like(first_deriv),
                    create_graph=False, # No need for higher-order derivatives for this loss
                    retain_graph=True # Keep graph if other losses need it, but generally not needed for this
                )[0]
                if second_deriv is not None:
                    total_sobolev_loss += torch.mean(second_deriv**2)
            
            num_kan_layers += 1
    
    if num_kan_layers == 0:
        return torch.tensor(0.0, device=model.config.device)

    # Average over all KAN layers and apply regularization strength
    return lambda_sobolev * (total_sobolev_loss / num_kan_layers)


def total_variation_l1_loss(model: KAN_MAMOTE_Model, lambda_tv: float) -> torch.Tensor:
    """
    Calculates the Total Variation L1 regularization loss for the KAN experts.
    Penalizes the L1 norm of the function's gradient to encourage piecewise
    constant/linear solutions and sharp, interpretable transitions.

    Args:
        model: The KAN_MAMOTE_Model instance.
        lambda_tv: The regularization strength.

    Returns:
        torch.Tensor: Scalar Total Variation L1 loss.
    """
    if lambda_tv == 0:
        return torch.tensor(0.0, device=model.config.device)

    total_tv_loss = torch.tensor(0.0, device=model.config.device)
    num_kan_layers = 0

    # Iterate through all KANLayer instances within the K-MOTE modules
    for k_mote_module in [model.k_mote_abs, model.k_mote_rel]:
        for expert_name in k_mote_module.expert_names:
            kan_layer: KANLayer = k_mote_module.experts[expert_name]
            basis_function = kan_layer.basis_function

            # Create a dummy input range for derivative calculation
            dummy_x_prime = torch.linspace(-5.0, 5.0, 100).view(-1, 1).repeat(1, basis_function.output_dim).to(model.config.device)
            dummy_x_prime.requires_grad_(True)

            output = basis_function(dummy_x_prime) # (100, output_dim)
            
            # Compute first derivatives (d(output)/d(dummy_x_prime))
            first_deriv = torch.autograd.grad(
                outputs=output,
                inputs=dummy_x_prime,
                grad_outputs=torch.ones_like(output),
                create_graph=True, # Need to retain graph for subsequent calls in a complex setup if sharing `dummy_x_prime`
                retain_graph=True # For multiple gradient calls if needed
            )[0]
            
            # L1 norm of first derivative
            total_tv_loss += torch.mean(torch.abs(first_deriv))
            
            num_kan_layers += 1

    if num_kan_layers == 0:
        return torch.tensor(0.0, device=model.config.device)

    # Average over all KAN layers and apply regularization strength
    return lambda_tv * (total_tv_loss / num_kan_layers)


def moe_load_balancing_loss(expert_weights_abs: torch.Tensor, expert_weights_rel: torch.Tensor, lambda_moe: float) -> torch.Tensor:
    """
    Calculates the MoE load balancing loss.
    Encourages a balanced distribution of workload across experts by penalizing
    disparities in expert usage (to prevent expert collapse).

    Formula (from common MoE papers, simplified):
    Loss = sum_i (importance_i * log(importance_i)) where importance_i = mean_over_batch(weight_i)
    Or often: (mean_over_batch(weights)).pow(2) * num_experts
    A common formulation is: (sum_j p_j * log p_j) - (sum_j p_j) log (sum_j p_j)
    More commonly: Sum(p_j * log(p_j)) over experts.
    Another common one (from GShard, Switch Transformers):
    (expert_load_sum * expert_router_prob_sum).sum() where sums are over batch/tokens.

    Let's use a standard formulation for this:
    L_balance = N * sum_{i=1 to num_experts} (p_i * log(p_i)) - N * log(sum_{i=1 to num_experts} p_i)
    where p_i = sum over batch (expert_weights_i) / sum over batch,experts (expert_weights)
    Simpler: N * (sum_i p_i log p_i - (sum_i p_i) log(sum_i p_i)) where p_i is sum over all batch/seq.
    Even simpler from gMLP: (router_prob_sum * router_prob_sum.transpose()).sum() where router_prob_sum = mean(weights, 0)

    A simple and widely used form (from GShard/Switch Transformers) is:
    `loss = (mean_expert_weights.sum() * (mean_expert_weights_squared_sum - mean_expert_weights_sum_sq)) / num_experts`
    This encourages router probabilities to be uniform and ensures all experts are used.
    A simpler version often used is `sum_i (mean_expert_weight_i)^2` or `(expert_weights.mean(dim=(0,1)))^2.sum()`
    This penalizes experts that receive very low average weights.
    Let's use a common and effective version:
    `loss = (num_experts * mean_expert_prob^2).sum() - mean_expert_prob.sum()^2` (simplified from Switch Transformer variant)

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

    # Compute load balancing loss as per common implementations (e.g., Switch Transformers simplified)
    # L_balance = num_experts * sum(p_i^2) - (sum(p_i))^2
    # This pushes p_i towards uniformity.
    loss = (mean_expert_prob_across_tasks.sum() * (mean_expert_prob_across_tasks**2).sum()) * self.config.num_experts - (mean_expert_prob_across_tasks.sum()**2)
    
    # The GShard/Switch-Transformer version for load balancing:
    # l_aux = (p_router * p_expert).sum()
    # where p_router = weights.mean(0) (avg prob of router dispatching to expert i)
    # p_expert = (batch_expert_mask * loss_per_example).sum(0) / batch_expert_mask.sum(0) (avg loss for expert i)
    # A simpler form is often used: sum_i (mean_prob_i * mean_load_i)
    
    # A very common and simple MoE load balancing loss (from https://github.com/lucidrains/switching-transformers-pytorch/blob/main/switching_transformers_pytorch/switching_transformers_pytorch.py)
    # This is (mean_expert_prob_across_tasks * expert_load).sum() where expert_load is a proxy.
    # Simpler version:
    load_balance_loss = (mean_expert_prob_across_tasks * mean_expert_prob_across_tasks).sum()

    return lambda_moe * load_balance_loss

# Add a combined loss function for convenience in train.py
def calculate_total_loss(
    main_task_loss: torch.Tensor,
    model: KAN_MAMOTE_Model,
    moe_losses_info: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, dict]:
    """
    Calculates the total loss including main task loss and regularization terms.

    Args:
        main_task_loss: The primary loss from the downstream task (e.g., MSE, CrossEntropy).
        model: The KAN_MAMOTE_Model instance.
        moe_losses_info: Tuple containing (abs_expert_weights_for_loss, rel_expert_weights_for_loss).

    Returns:
        torch.Tensor: The sum of all loss components.
    """
    config = model.config
    
    # Regularization Losses
    sobolev_loss = sobolev_l2_loss(model, config.lambda_sobolev_l2)
    tv_loss = total_variation_l1_loss(model, config.lambda_total_variation_l1)
    
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