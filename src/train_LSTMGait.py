import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from models.LSTMGait import LSTMGait

np.random.seed(42)
torch.manual_seed(42)

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long) 
        return x, y


