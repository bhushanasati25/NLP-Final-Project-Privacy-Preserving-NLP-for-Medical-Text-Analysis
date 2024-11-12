import numpy as np
from typing import Dict, List
import torch

class PrivacyUtils:
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.privacy_budget_spent = 0.0
    
    def add_noise_to_gradients(
        self,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Add Gaussian noise to gradients
        """
        grad_norm = torch.norm(gradients)
        
        # Clip gradients
        gradients = gradients * min(1, self.max_grad_norm / (grad_norm + 1e-6))
        
        # Add noise
        noise = torch.normal(
            mean=0,
            std=self.noise_multiplier * self.max_grad_norm,
            size=gradients.shape,
            device=gradients.device
        )
        
        return gradients + noise
    
    def calculate_privacy_spent(
        self,
        num_samples: int,
        batch_size: int,
        epochs: int
    ) -> Dict:
        """
        Calculate privacy budget spent
        """
        # Calculate number of compositions
        num_steps = num_samples * epochs // batch_size
        
        # Calculate privacy spent using moments accountant
        eps = (
            2 * np.sqrt(num_steps * np.log(1/self.delta))
            * self.noise_multiplier
        )
        
        self.privacy_budget_spent += eps
        
        return {
            'epsilon_spent': eps,
            'total_epsilon_spent': self.privacy_budget_spent,
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier
        }
    
    def check_privacy_guarantee(self) -> Dict:
        """
        Check if privacy guarantees are maintained
        """
        status = 'good'
        message = 'Privacy guarantees maintained'
        
        if self.privacy_budget_spent > self.epsilon:
            status = 'exceeded'
            message = 'Privacy budget exceeded'
            
        return {
            'status': status,
            'message': message,
            'budget_spent': self.privacy_budget_spent,
            'budget_limit': self.epsilon,
            'remaining_budget': max(0, self.epsilon - self.privacy_budget_spent)
        }