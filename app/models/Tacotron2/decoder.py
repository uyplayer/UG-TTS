import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, decoder_input, attention_context):
        rnn_input = torch.cat((decoder_input, attention_context), dim=2)
        rnn_output, (hidden_state, cell_state) = self.rnn(rnn_input)
        mel_output = self.fc(rnn_output)
        return mel_output, hidden_state, cell_state

if __name__ == '__main__':
    batch_size = 16
    hidden_dim = 512
    output_dim = 80

    decoder = Decoder(input_dim=1024, hidden_dim=512, output_dim=80)
    context = torch.randn(batch_size, 1, hidden_dim)
    decoder_input = torch.randn(batch_size, 1, hidden_dim)

    mel_output, new_hidden_state, new_cell_state = decoder(decoder_input, context)
    print("Mel Output shape:", mel_output.shape)
    print("New Hidden State shape:", new_hidden_state.shape)
    print("New Cell State shape:", new_cell_state.shape)
