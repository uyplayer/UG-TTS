import torch
from torch import nn
from app.models.Tacotron2.text_encoder import TextEncoder
from app.models.Tacotron2.attention import Attention
from app.models.Tacotron2.decoder import Decoder


class Tacotron2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Tacotron2, self).__init__()
        self.encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)
        self.attention = Attention(hidden_dim * 2)
        self.decoder = Decoder(hidden_dim, hidden_dim, output_dim)


    def forward(self, x, device):
        lengths = [len(seq) for seq in x]
        encoder_output = self.encoder(x, lengths).to(device) # (batch_size, seq_len, hidden_dim)
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, 1, encoder_output.size(2)).to(device)  # (batch_size, 1, hidden_dim)
        query = torch.zeros(batch_size, encoder_output.size(2)).to(device)   # (batch_size, hidden_dim)
        attention_context, attention_weights = self.attention(query, encoder_output, encoder_output)
        decoder_output = self.decoder(decoder_input, attention_context)
        return decoder_output, attention_weights


if __name__ == '__main__':
    pass
