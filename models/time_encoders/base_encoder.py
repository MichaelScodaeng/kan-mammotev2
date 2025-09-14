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
    def forward(self, timestamps: torch.Tensor, time_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode timestamps into embeddings.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len)
                       containing absolute timestamps
            time_deltas: Optional tensor of same shape as timestamps containing
                        relative time differences. If None, may be computed internally.
        
        Returns:
            Time embeddings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
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
