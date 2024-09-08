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
        self.decoder = Decoder(hidden_dim*2, hidden_dim , output_dim)

    def forward(self, x, device):
        lengths = [len(seq) for seq in x]
        encoder_output = self.encoder(x, lengths).to(device)  # (batch_size, seq_len, hidden_dim)
        batch_size = x.size(0)
        seq_len = encoder_output.size(1)
        hidden_dim = encoder_output.size(2)
        decoder_input = torch.zeros(batch_size, seq_len, hidden_dim).to(device)
        query = torch.zeros(batch_size, hidden_dim).to(device)
        attention_context, attention_weights = self.attention(query, encoder_output,encoder_output)
        attention_context = attention_context.unsqueeze(1).repeat(1, seq_len, 1)
        # decoder_input.shape torch.Size([16, 235, 1024])
        # attention_context  torch.Size([16, 235, 1024])
        mel_output = self.decoder(decoder_input, attention_context)
        return mel_output


if __name__ == '__main__':
    pass
