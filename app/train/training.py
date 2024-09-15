import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.audio.audio_processing import AudioProcess
from app.models.Tacotron2 import tacotron2
from app.dataset.common_voice import TextMelDataset, collate_fn
from app.text_processing.text_cleaner import TextCleaner
from app.phoneme_preprocessing.phoneme_conversion import Phoneme
from app.train.mel_data import get_mel_targets
from common.path_config import model_saving_dir

warnings.simplefilter(action='ignore', category=UserWarning)


def validate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    audio_processor = AudioProcess()  # Ensure audio processor is available in validate function
    with torch.no_grad():
        for wave_paths, phoneme_sequences in dataloader:
            phoneme_sequences = phoneme_sequences.to(device)
            mel_output = model(phoneme_sequences, device)
            mel_targets = get_mel_targets(wave_paths, audio_processor, target_length=250, n_mels=250).to(device)
            loss = criterion(mel_output, mel_targets)
            val_loss += loss.item()
    return val_loss / len(dataloader)


def train(model, train_loader, val_loader, criterion, optimizer, device, batch_size, hidden_dim, num_epochs=10):
    name = str(model.__class__.__name__)
    audio_processor = AudioProcess()
    model_saving_path = os.path.join(model_saving_dir, name)
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, (wave_paths, phoneme_sequences) in enumerate(train_loader):
            phoneme_sequences = phoneme_sequences.to(device)
            optimizer.zero_grad()
            mel_output = model(phoneme_sequences, device)
            mel_targets = get_mel_targets(wave_paths, audio_processor, target_length=250, n_mels=250).to(device)
            if torch.isnan(phoneme_sequences).any():
                print(" phoneme_sequences is none ")
            if torch.isnan(mel_output).any():
                print(" mel_output is none ")
            if torch.isnan(mel_targets).any():
                print(" mel_targets is none ")
            loss = criterion(mel_output, mel_targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Found NaN or Inf in loss, stopping training.")
                return  # Stop training if nan or inf is found
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Validate after every epoch
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_saving_path, f'tacotron2_epoch_{epoch + 1}.pth'))
            print(f'Model saved at epoch {epoch + 1}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths to the dataset
    clip_dir = r"E:\MachineLearning\news_data\validated_clips"
    txt_file_path = r"E:\MachineLearning\news_data\validated.txt"

    # Initialize the phoneme processor and text cleaner
    phoneme_processor = Phoneme()
    cleaner = TextCleaner()

    # Create the dataset
    dataset = TextMelDataset(clip_dir, txt_file_path, cleaner, phoneme_processor)

    # Split dataset into train (80%), validation (10%) and test (10%)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model parameters
    hidden_dim = 512
    batch_size = 64
    model = tacotron2.Tacotron2(vocab_size=200, embedding_dim=256, hidden_dim=512, output_dim=250).to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, batch_size, hidden_dim, num_epochs=50)
