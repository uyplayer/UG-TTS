import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Va = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, values):
        # Query: (batch_size, hidden_dim)
        # Keys: (batch_size, seq_len, hidden_dim)
        # Values: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = keys.size()
        query = query.unsqueeze(1).repeat(1, seq_len, 1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        return context, attention_weights


if __name__ == '__main__':
    batch_size = 16
    seq_len = 10
    hidden_dim = 512

    attention = Attention(hidden_dim)
    query = torch.randn(batch_size, hidden_dim)
    keys = torch.randn(batch_size, seq_len, hidden_dim)
    values = torch.randn(batch_size, seq_len, hidden_dim)

    context, attention_weights = attention(query, keys, values)
    print("Context shape:", context.shape)
    print("Attention Weights shape:", attention_weights.shape)
