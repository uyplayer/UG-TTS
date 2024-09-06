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
        #  query torch.Size([16, 512])  .   values ,keys   torch.Size([16, 310, 1024])
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.bmm(attention_weights.transpose(1, 2), values)  # (batch_size, 1, hidden_dim)
        return context, attention_weights

if __name__ == '__main__':
    batch_size = 16
    sequence_length = 199
    hidden_dim = 1024
    attention = Attention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, sequence_length, hidden_dim)
    values = torch.randn(batch_size, sequence_length, hidden_dim)
    context, attention_weights = attention(query, keys, values)
    print("Query shape:", query.shape)
    print("Keys shape:", keys.shape)
    print("Values shape:", values.shape)
    print("Context shape:", context.shape)
    print("Attention Weights shape:", attention_weights.shape)
    print("\nContext:\n", context)
    print("\nAttention Weights:\n", attention_weights)
