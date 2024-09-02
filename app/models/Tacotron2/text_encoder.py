import torch
from torch import nn
from app.text_processing.text_cleaner import TextCleaner
from app.phoneme_preprocessing.phoneme_conversion import Phoneme

class TextEncoder(nn.Module):
    """
    TextEncoder for Tacotron2
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return x

if __name__ == '__main__':


    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_text(text)

    phoneme_processor = Phoneme()
    phonemes = phoneme_processor.phoneme(cleaned_text)

    phoneme_sequence = phoneme_processor.phoneme_to_sequence(phonemes)
    phoneme_sequence = torch.tensor(phoneme_sequence).unsqueeze(0)
    print(list(cleaned_text))
    print(len(list(cleaned_text)))
    print(list(phoneme_sequence))
    print(phoneme_sequence)
    print(f"phonemes = {len(phonemes)},phoneme_sequence.shape = {phoneme_sequence.shape},cleaned_text = {len(cleaned_text)}")

