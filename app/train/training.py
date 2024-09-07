import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.cbook import pts_to_midstep
from torch.utils.data import DataLoader
from app.models.Tacotron2 import tacotron2
from app.dataset.common_voice import TextMelDataset, collate_fn
from app.text_processing.text_cleaner import TextCleaner
from app.phoneme_preprocessing.phoneme_conversion import Phoneme
from common.path_config import model_saving_dir
import librosa

def train(model, dataloader, criterion, optimizer, device,batch_size,hidden_dim, num_epochs=10):
    name = str(model._get_name())
    model_saving_path = os.path.join(model_saving_dir,name)
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (wave_paths, phoneme_sequences) in enumerate(dataloader):
            phoneme_sequences = phoneme_sequences.to(device)
            optimizer.zero_grad()
            mel_output, _ = model(phoneme_sequences,device)
            mel_targets = get_mel_targets(wave_paths).to(device)
            loss = criterion(mel_output, mel_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        torch.save(model.state_dict(), f'{model_saving_path}/tacotron2_epoch_{epoch + 1}.pth')


def get_mel_targets(wave_paths):
    mel_targets = []
    for path in wave_paths:
        mel_spectrogram = load_and_convert_to_mel(path)
        mel_targets.append(mel_spectrogram)
    return torch.stack(mel_targets)


def load_and_convert_to_mel(wave_path):

    y, sr = librosa.load(wave_path, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=80)
    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
    return mel_spectrogram


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_dir = r"E:\MachineLearning\news_data\dev_clips"
    txt_file_path = r"E:\MachineLearning\news_data\dev.txt"
    phoneme_processor = Phoneme()
    cleaner = TextCleaner()
    dataset = TextMelDataset(clip_dir, txt_file_path, cleaner, phoneme_processor)
    hidden_dim = 512
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    model = tacotron2.Tacotron2(vocab_size=200, embedding_dim=256, hidden_dim=512, output_dim=80).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, dataloader, criterion, optimizer, device,batch_size,hidden_dim, num_epochs=50)
