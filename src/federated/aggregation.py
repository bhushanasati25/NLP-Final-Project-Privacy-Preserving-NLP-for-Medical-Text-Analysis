import torch
from typing import List, Dict, Any
import numpy as np

class FederatedAggregator:
    """Implements various federated aggregation strategies"""
    
    def __init__(self, strategy: str = "fedavg"):
        self.strategy = strategy
        self.supported_strategies = {
            "fedavg": self._federated_averaging,
            "weighted": self._weighted_averaging,
            "median": self._coordinate_wise_median,
            "trimmed_mean": self._trimmed_mean
        }
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using selected strategy"""
        if self.strategy not in self.supported_strategies:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        return self.supported_strategies[self.strategy](client_updates, client_weights)
    
    def _federated_averaging(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Implement FedAvg algorithm"""
        if client_weights is None:
            client_weights = [1/len(client_updates)] * len(client_updates)
            
        aggregated_update = {}
        
        for name in client_updates[0].keys():
            aggregated_update[name] = torch.zeros_like(
                client_updates[0][name]
            )
            
            for update, weight in zip(client_updates, client_weights):
                aggregated_update[name] += weight * update[name]
        
        return aggregated_update
    
    def _weighted_averaging(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Implement weighted averaging based on client data size"""
        if client_weights is None:
            raise ValueError("Weights required for weighted averaging")
            
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w/total_weight for w in client_weights]
        
        return self._federated_averaging(client_updates, normalized_weights)
    
    def _coordinate_wise_median(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Implement coordinate-wise median aggregation"""
        aggregated_update = {}
        
        for name in client_updates[0].keys():
            stacked_updates = torch.stack(
                [update[name] for update in client_updates]
            )
            aggregated_update[name] = torch.median(stacked_updates, dim=0)[0]
        
        return aggregated_update
    
    def _trimmed_mean(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None,
        trim_ratio: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Implement trimmed mean aggregation"""
        aggregated_update = {}
        num_clients = len(client_updates)
        num_trim = int(num_clients * trim_ratio)
        
        for name in client_updates[0].keys():
            stacked_updates = torch.stack(
                [update[name] for update in client_updates]
            )
            sorted_updates, _ = torch.sort(stacked_updates, dim=0)
            trimmed_updates = sorted_updates[num_trim:-num_trim]
            aggregated_update[name] = torch.mean(trimmed_updates, dim=0)
        
        return aggregated_update