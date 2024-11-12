import torch
import numpy as np
from typing import Dict, Any
import os
import json
from datetime import datetime

class ModelUtils:
    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        save_dir: str,
        metrics: Dict = None
    ) -> str:
        """
        Save model checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        filepath = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')
        torch.save(checkpoint, filepath)
        return filepath
    
    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: str = None
    ) -> Dict:
        """
        Load model checkpoint
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint
    
    @staticmethod
    def calculate_model_size(model: torch.nn.Module) -> Dict:
        """
        Calculate model size and parameters
        """
        param_size = 0
        param_count = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_count += param.numel()
            param_size += param.numel() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'parameter_count': param_count,
            'model_size_mb': size_mb,
            'parameter_size_bytes': param_size,
            'buffer_size_bytes': buffer_size
        }