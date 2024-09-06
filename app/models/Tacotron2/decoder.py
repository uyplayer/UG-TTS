import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, hidden_dim, rnn_hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_dim, output_dim)

    def forward(self, context, hidden_state):
        rnn_input = context.squeeze(1)
        rnn_output, (hidden_state, cell_state) = self.rnn(rnn_input.unsqueeze(1), (hidden_state.unsqueeze(0), torch.zeros_like(hidden_state.unsqueeze(0))))
        mel_output = self.fc(rnn_output.squeeze(1))
        return mel_output, hidden_state.squeeze(0)

if __name__ == '__main__':
    batch_size = 16
    hidden_dim = 512
    rnn_hidden_dim = 512
    output_dim = 80
    decoder = Decoder(hidden_dim, rnn_hidden_dim, output_dim)
    context = torch.randn(batch_size, 1, hidden_dim)
    hidden_state = torch.randn(batch_size, rnn_hidden_dim)
    mel_output, new_hidden_state = decoder(context, hidden_state)
    print("Context shape:", context.shape)
    print("Hidden State shape:", hidden_state.shape)
    print("Mel Output shape:", mel_output.shape)
    print("New Hidden State shape:", new_hidden_state.shape)
