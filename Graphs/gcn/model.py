import math
import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,in_features, out_features,bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_params()
    
    def _reset_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # Inplace
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = input @ self.weight
        output = adj @ support

        if self.bias is not None:
            return output + self.bias
        
        return output


class GCN(nn.Module):
    def __init__(self,infeat, nhid, nclass,dropout = .50):
        super().__init__()
        self.gc1 = GraphConv(infeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self,x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)