import torch

from utils import load_data

adj, features, labels, idx_train, idx_val, idx_test = load_data()

print (adj, features)