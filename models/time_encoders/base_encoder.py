"""
Base Time Encoder Interface

Provides abstract base class for all time encoders in the KAN-MAMMOTE framework.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union


class BaseTimeEncoder(nn.Module, ABC):
    """
    Abstract base class for all time encoders.
    
    This class defines the common interface that all time encoders must implement,
    ensuring consistency across different encoding strategies.
    """
    
    def __init__(self, time_dim: int, device: str = 'cpu'):
        """
        Initialize the base time encoder.
        
        Args:
            time_dim: Output dimension of time encoding
            device: Device to place the encoder on ('cpu' or 'cuda')
        """
        super().__init__()
        self.time_dim = time_dim
        self.device = device
    
    @abstractmethod
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        The unified forward pass for all time encoders.
        
        Args:
            t_abs (torch.Tensor): Absolute timestamps.
            t_rel (torch.Tensor): Relative time deltas.
        
        Returns:
            torch.Tensor: The final time embedding.
        """
        pass
    
    def to_device(self, device: str):
        """Move encoder to specified device."""
        self.device = device
        return self.to(device)
    
    def get_config(self) -> dict:
        """Return configuration dictionary for reproducibility."""
        return {
            'type': self.__class__.__name__,
            'time_dim': self.time_dim,
            'device': self.device
        }
