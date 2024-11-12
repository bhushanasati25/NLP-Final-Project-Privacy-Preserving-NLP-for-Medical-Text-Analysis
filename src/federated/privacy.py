import torch
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class PrivacyParams:
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float

class PrivacyMechanism:
    """Implementation of differential privacy mechanisms for federated learning"""
    
    def __init__(self, params: PrivacyParams):
        self.params = params
        self.privacy_budget_spent = 0.0
        
    def add_noise_to_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to gradients"""
        # Clip gradients by norm
        grad_norm = torch.norm(gradients)
        gradients = gradients * min(1, self.params.max_grad_norm / (grad_norm + 1e-6))
        
        # Add calibrated noise
        noise = torch.normal(
            mean=0,
            std=self.params.noise_multiplier * self.params.max_grad_norm,
            size=gradients.shape,
            device=gradients.device
        )
        
        return gradients + noise
    
    def privatize_model_update(
        self,
        model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Apply privacy mechanism to model updates"""
        private_updates = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                private_updates[name] = self.add_noise_to_gradients(param.grad)
        
        return private_updates
    
    def compute_privacy_spent(
        self,
        num_steps: int,
        batch_size: int,
        data_size: int
    ) -> Dict[str, float]:
        """Compute privacy budget spent using moments accountant"""
        q = batch_size / data_size  # Sampling ratio
        
        # Privacy loss computation
        eps = (
            2 * np.sqrt(num_steps * np.log(1/self.params.delta))
            * self.params.noise_multiplier
            * q
        )
        
        self.privacy_budget_spent += eps
        
        return {
            'epsilon_spent': eps,
            'total_epsilon_spent': self.privacy_budget_spent,
            'remaining_budget': max(0, self.params.epsilon - self.privacy_budget_spent)
        }
    
    def check_privacy_guarantee(self) -> Tuple[bool, str]:
        """Check if privacy guarantees are maintained"""
        if self.privacy_budget_spent > self.params.epsilon:
            return False, "Privacy budget exceeded"
        return True, "Privacy guarantees maintained"