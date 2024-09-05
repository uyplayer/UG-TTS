# attention.py
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, values):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights, values)
        return context, weights
