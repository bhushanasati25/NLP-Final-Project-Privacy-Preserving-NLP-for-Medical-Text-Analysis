import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    """Self-attention layer with privacy-preserving capabilities"""
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(input_dim, input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Linear transformations
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        
        # Reshape and transform output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )
        output = self.output_layer(context)
        
        return output

class PrivacyLayer(nn.Module):
    """Layer implementing privacy-preserving mechanisms"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0
    ):
        super(PrivacyLayer, self).__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Add noise during training
            return self._add_noise(x)
        return x
    
    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for differential privacy"""
        # Clip gradients by norm
        grad_norm = torch.norm(x)
        x = x * min(1, self.clip_norm / (grad_norm + 1e-6))
        
        # Calculate noise scale
        noise_scale = (
            2 * self.clip_norm * math.sqrt(2 * math.log(1.25/self.delta))
        ) / self.epsilon
        
        # Add noise
        noise = torch.normal(
            mean=0,
            std=noise_scale,
            size=x.shape,
            device=x.device
        )
        
        return x + noise