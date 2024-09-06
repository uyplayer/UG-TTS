import torch
from torch import nn
from app.models.Tacotron2.text_encoder import TextEncoder
from app.models.Tacotron2.attention import Attention
from app.models.Tacotron2.decoder import Decoder

class Tacotron2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Tacotron2, self).__init__()
        self.encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.decoder = Decoder(hidden_dim, hidden_dim, output_dim)
        self.hidden_state_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden_state):
        encoder_output = self.encoder(x, hidden_state)
        hidden_state = self.hidden_state_projection(hidden_state)
        context, attention_weights = self.attention(hidden_state, encoder_output, encoder_output)
        mel_output, hidden_state = self.decoder(context, hidden_state)
        return mel_output, attention_weights

if __name__ == '__main__':
    pass
