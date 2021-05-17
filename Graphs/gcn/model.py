import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        pass
        self.layer_1 = nn.Linear(5,6)

    def forward(self,x):
        print(x.shape)
        return x