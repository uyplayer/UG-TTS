# decoder.py
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden_state):
        output, (hn, cn) = self.lstm(x, hidden_state)
        output = self.fc(self.dropout(output))
        return output, (hn, cn)
