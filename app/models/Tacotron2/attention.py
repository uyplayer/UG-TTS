import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, values):
        if query is None:
            raise ValueError("The hidden state (query) is None, which is required for attention mechanism.")
        query = query.unsqueeze(1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), values)

        return context, attention_weights


if __name__ == '__main__':
    batch_size = 16
    sequence_length = 199
    hidden_dim = 1024
    attention = Attention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, sequence_length, hidden_dim)
    values = torch.randn(batch_size, sequence_length, hidden_dim)
    # query.shape torch.Size([16, 1024])
    # keys.shape torch.Size([16, 199, 1024])
    # values.shape torch.Size([16, 199, 1024])
    # context.shape torch.Size([16, 1, 1024])
    # attention_weights.shape torch.Size([16, 199])
    context, attention_weights = attention(query, keys, values)
    print("Query shape:", query.shape)
    print("Keys shape:", keys.shape)
    print("Values shape:", values.shape)
    print("Context shape:", context.shape)
    print("Attention Weights shape:", attention_weights.shape)
    print("\nContext:\n", context)
    print("\nAttention Weights:\n", attention_weights)
