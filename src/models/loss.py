import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PrivacyAwareLoss(nn.Module):
    """Loss function with privacy-preserving capabilities"""
    
    def __init__(
        self,
        base_criterion: str = "cross_entropy",
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_multiplier: float = 1.0
    ):
        super(PrivacyAwareLoss, self).__init__()
        self.base_criterion = base_criterion
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        
        self.criterion_map = {
            "cross_entropy": self._cross_entropy_loss,
            "mse": self._mse_loss,
            "binary": self._binary_cross_entropy_loss
        }
        
        if base_criterion not in self.criterion_map:
            raise ValueError(f"Unsupported criterion: {base_criterion}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute privacy-aware loss
        
        Returns:
            - privacy-preserving loss value
            - privacy metrics dictionary
        """
        # Compute base loss
        base_loss = self.criterion_map[self.base_criterion](predictions, targets)
        
        if self.training:
            # Add noise to loss gradient
            noisy_loss = self._add_gradient_noise(base_loss)
            
            # Calculate privacy cost
            privacy_cost = self._compute_privacy_cost(len(targets))
            
            return noisy_loss, privacy_cost
        
        return base_loss, {"privacy_cost": 0.0}
    
    def _cross_entropy_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross entropy loss"""
        return F.cross_entropy(predictions, targets)
    
    def _mse_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean squared error loss"""
        return F.mse_loss(predictions, targets)
    
    def _binary_cross_entropy_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross entropy loss"""
        return F.binary_cross_entropy_with_logits(predictions, targets)
    
    def _add_gradient_noise(self, loss: torch.Tensor) -> torch.Tensor:
        """Add noise to loss gradient for privacy"""
        noise_scale = self.noise_multiplier * math.sqrt(
            2 * math.log(1.25/self.delta)
        ) / self.epsilon
        
        noise = torch.normal(
            mean=0,
            std=noise_scale,
            size=loss.shape,
            device=loss.device
        )
        
        return loss + noise
    
    def _compute_privacy_cost(self, batch_size: int) -> Dict[str, float]:
        """Compute privacy cost for current batch"""
        # Simplified privacy cost calculation
        privacy_cost = (
            self.noise_multiplier
            * math.sqrt(2 * math.log(1.25/self.delta))
            * batch_size
        ) / self.epsilon
        
        return {
            "privacy_cost": float(privacy_cost),
            "epsilon_used": float(self.epsilon),
            "noise_scale": float(self.noise_multiplier)
        }