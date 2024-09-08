import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.cbook import pts_to_midstep
from torch.utils.data import DataLoader

from app.audio.audio_processing import AudioProcess
from app.models.Tacotron2 import tacotron2
from app.dataset.common_voice import TextMelDataset, collate_fn
from app.text_processing.text_cleaner import TextCleaner
from app.phoneme_preprocessing.phoneme_conversion import Phoneme
from app.train.mel_data import get_mel_targets
from common.path_config import model_saving_dir

warnings.simplefilter(action='ignore', category=UserWarning)

def train(model, dataloader, criterion, optimizer, device, batch_size, hidden_dim, num_epochs=10):
    name = str(model.__class__.__name__)
    audio_processor = AudioProcess()
    model_saving_path = os.path.join(model_saving_dir, name)
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (wave_paths, phoneme_sequences) in enumerate(dataloader):
            phoneme_sequences = phoneme_sequences.to(device)
            optimizer.zero_grad()
            # mel_output.shape torch.Size([16, 250, 298])
            mel_output  = model(phoneme_sequences, device)
            # torch.Size([16, 250, 250])
            mel_targets = get_mel_targets(wave_paths, audio_processor, target_length=250,n_mels=250).to(device)
            loss = criterion(mel_output, mel_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        torch.save(model.state_dict(), os.path.join(model_saving_path, f'tacotron2_epoch_{epoch + 1}.pth'))


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
    model = tacotron2.Tacotron2(vocab_size=200, embedding_dim=256, hidden_dim=512, output_dim=250).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, dataloader, criterion, optimizer, device,batch_size,hidden_dim, num_epochs=50)
