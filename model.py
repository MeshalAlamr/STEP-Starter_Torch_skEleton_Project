import torch
from torch import nn

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear( , ),
            nn.BatchNorm1d(),
            nn.ReLU(),     
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    

    
    
    

    
