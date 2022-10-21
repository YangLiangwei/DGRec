import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import WeightedRandomSampler
import random
import pdb

class NegativeSampler:
    def __init__(self, args, dataloader, device):
        self.num_user = dataloader.user_number
        self.num_item = dataloader.item_number
        self.adj = dataloader.train_csr
        self.batch_size = args.batch_size
        self.neg_number = args.neg_number
        self.device = device

    def random_sample(self, users):
        weight = torch.ones((users.shape[0], self.num_item)).to(self.device)
        mask = torch.tensor(self.adj[users.cpu()].toarray()).bool().to(self.device)
        weight[mask] = 0.0
        items = WeightedRandomSampler(weight, self.neg_number, replacement = False)
        items = torch.tensor(list(items)).reshape(-1)
        return items
