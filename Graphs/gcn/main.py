import torch
import numpy as np
from model import GCN
import torch.optim as optim
import torch.nn.functional as F
from utils import load_data,accuracy

torch.manual_seed(42)
np.random.seed(42)

adj, features, labels, idx_train, idx_val, idx_test = load_data() 

model = GCN(features.shape[1],16,labels.max().item()+1)

opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for i in range(200):
    opt.zero_grad()
    model.train()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])    
    loss_train.backward()
    opt.step()
    
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if True:
        model.eval()
        output = model(features, adj)
        print ("Epoch: {:02d}, Train Loss: {:04f}, Train Acc: {:04f}, Val Loss: {:04f}, Val Acc: {:04f}".format(i + 1,loss_train.item(), acc_train.item(),loss_val.item(), acc_val.item()))

model.eval()
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])

print ("\n\nTest Loss: {:04f}, Test Acc: {:04f}".format(loss_test, acc_test))
