# kan_mamote/src/losses/simple_losses.py

import torch
from typing import Tuple

def moe_load_balancing_loss(expert_weights_abs: torch.Tensor, expert_weights_rel: torch.Tensor, lambda_moe: float) -> torch.Tensor:
    """
    Simple MoE load balancing loss to prevent expert collapse.
    This is the only regularization that's really necessary for MoE architectures.
    """
    if lambda_moe == 0:
        return torch.tensor(0.0, device=expert_weights_abs.device)

    # Combine weights from both K-MOTE modules
    all_expert_weights = torch.stack([expert_weights_abs, expert_weights_rel], dim=2)
    
    # Average expert weights across batch and sequence length
    mean_expert_prob = all_expert_weights.mean(dim=(0, 1, 2)) 
    
    # Encourage uniform distribution across experts
    load_balance_loss = (mean_expert_prob ** 2).sum()

    return lambda_moe * load_balance_loss


def simple_total_loss(
    main_task_loss: torch.Tensor,
    moe_losses_info: Tuple[torch.Tensor, torch.Tensor],
    lambda_moe: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    Simple loss calculation with only MoE load balancing.
    
    Args:
        main_task_loss: Your downstream task loss
        moe_losses_info: (abs_expert_weights, rel_expert_weights)
        lambda_moe: MoE regularization strength
    
    Returns:
        total_loss, loss_dict
    """
    abs_expert_weights, rel_expert_weights = moe_losses_info
    moe_loss = moe_load_balancing_loss(abs_expert_weights, rel_expert_weights, lambda_moe)
    
    total_loss = main_task_loss + moe_loss
    
    return total_loss, {
        "main_task_loss": main_task_loss.item(),
        "moe_loss": moe_loss.item(),
        "total_loss": total_loss.item()
    }


def feature_extractor_loss(main_task_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Pure feature extractor mode - no regularization at all.
    Let the main task loss guide the time embedding learning.
    """
    return main_task_loss, {
        "main_task_loss": main_task_loss.item(),
        "total_loss": main_task_loss.item()
    }


def flexible_feature_loss(
    main_task_loss: torch.Tensor,
    model,
    moe_losses_info: Tuple[torch.Tensor, torch.Tensor],
    timestamps: torch.Tensor,
    lambda_moe: float = 0.01,
    lambda_sobolev: float = 0.0,
    lambda_tv: float = 0.0
) -> Tuple[torch.Tensor, dict]:
    """
    Flexible loss for feature extraction with optional regularization.
    You can enable/disable each regularization component as needed.
    
    Args:
        main_task_loss: Your downstream task loss
        model: KAN_MAMOTE_Model (only needed if using Sobolev/TV regularization)
        moe_losses_info: (abs_expert_weights, rel_expert_weights)
        timestamps: Batch timestamps (only needed if using Sobolev/TV regularization)
        lambda_moe: MoE load balancing strength (recommended: 0.01)
        lambda_sobolev: Smoothness regularization strength (0.0 = disabled)
        lambda_tv: Sharp transition regularization strength (0.0 = disabled)
    
    Returns:
        total_loss, loss_dict
    """
    loss_dict = {"main_task_loss": main_task_loss.item()}
    total_loss = main_task_loss
    
    # MoE Load Balancing (almost always recommended for MoE)
    if lambda_moe > 0:
        abs_expert_weights, rel_expert_weights = moe_losses_info
        moe_loss = moe_load_balancing_loss(abs_expert_weights, rel_expert_weights, lambda_moe)
        total_loss += moe_loss
        loss_dict["moe_loss"] = moe_loss.item()
    
    # Optional: Smoothness regularization for gradual temporal changes
    if lambda_sobolev > 0:
        from src.losses.regularization_losses import sobolev_l2_loss
        sobolev_loss = sobolev_l2_loss(model, lambda_sobolev, timestamps)
        total_loss += sobolev_loss
        loss_dict["sobolev_loss"] = sobolev_loss.item()
    
    # Optional: Sharp transition regularization for regime changes
    if lambda_tv > 0:
        from src.losses.regularization_losses import total_variation_l1_loss
        tv_loss = total_variation_l1_loss(model, lambda_tv, timestamps)
        total_loss += tv_loss
        loss_dict["tv_loss"] = tv_loss.item()
    
    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict


def recommended_feature_loss(
    main_task_loss: torch.Tensor,
    moe_losses_info: Tuple[torch.Tensor, torch.Tensor],
    lambda_moe: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    Recommended loss for most feature extraction use cases.
    Uses only MoE load balancing - minimal but effective regularization.
    """
    return simple_total_loss(main_task_loss, moe_losses_info, lambda_moe)
