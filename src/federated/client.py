import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FederatedClient:
    def __init__(self, model, dataset, client_id):
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, epochs=5, batch_size=32):
        """Train the model locally"""
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()