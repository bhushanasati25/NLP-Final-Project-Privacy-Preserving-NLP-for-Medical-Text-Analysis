import torch
import copy

class FederatedServer:
    def __init__(self, model, num_clients):
        self.global_model = model
        self.num_clients = num_clients
    
    def aggregate_models(self, client_models):
        """Aggregate client models using FedAvg"""
        global_dict = self.global_model.state_dict()
        
        for k in global_dict.keys():
            global_dict[k] = torch.stack([
                client_models[i][k].float() 
                for i in range(len(client_models))
            ], 0).mean(0)
        
        self.global_model.load_state_dict(global_dict)