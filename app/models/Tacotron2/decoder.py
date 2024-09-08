import torch
from torch import nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention_transform = nn.Linear(input_dim, input_dim)
        self.output_dim = output_dim
    def forward(self, x, attention_context):
        attention_context = self.attention_transform(attention_context)
        rnn_input = x + attention_context
        rnn_output, _ = self.rnn(rnn_input)
        mel_output = self.fc(rnn_output)
        mel_output = mel_output.permute(0, 2, 1)
        mel_output = self.adjust_output_shape(mel_output)
        return mel_output

    def adjust_output_shape(self, mel_output):
        batch_size, output_dim, seq_len = mel_output.size()
        target_seq_len = self.output_dim
        if seq_len < target_seq_len:
            padding_size = target_seq_len - seq_len
            mel_output = F.pad(mel_output, (0, padding_size), mode='constant', value=0)
        elif seq_len > target_seq_len:
            mel_output = mel_output[:, :, :target_seq_len]
        return mel_output



