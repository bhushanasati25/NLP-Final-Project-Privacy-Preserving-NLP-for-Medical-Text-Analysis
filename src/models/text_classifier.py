import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=5):
        super(SimpleTextClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)