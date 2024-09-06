import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from app.text_processing.text_cleaner import TextCleaner
from app.phoneme_preprocessing.phoneme_conversion import Phoneme


class TextMelDataset(Dataset):
    def __init__(self, wave_dir, txt_file, text_cleaner, phoneme_processor):
        self.wave_dir = wave_dir
        self.txt_file = txt_file
        self.text_cleaner = text_cleaner
        self.phoneme_processor = phoneme_processor

        if not os.path.exists(self.wave_dir):
            raise FileNotFoundError(f'{self.wave_dir} does not exist')

        self.data = []
        with open(self.txt_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wave, text = line.strip().split("|")
                self.data.append((wave, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wave_file, text = self.data[idx]
        wave_path = os.path.join(self.wave_dir, wave_file + ".wav")
        cleaned_text = self.text_cleaner.clean_text(text)
        phonemes = self.phoneme_processor.phoneme(cleaned_text)
        phoneme_sequence = self.phoneme_processor.phoneme_to_sequence(phonemes)
        return wave_path, torch.tensor(phoneme_sequence, dtype=torch.long)


def collate_fn(batch):
    wave_paths, phoneme_sequences = zip(*batch)
    phoneme_sequences_padded = pad_sequence(phoneme_sequences, batch_first=True, padding_value=0)
    return wave_paths, phoneme_sequences_padded


if __name__ == '__main__':
    clip_dir = r"E:\MachineLearning\news_data\dev_clips"
    txt_file_path = r"E:\MachineLearning\news_data\dev.txt"
    phoneme_processor = Phoneme()
    cleaner = TextCleaner()
    td = TextMelDataset(clip_dir, txt_file_path, cleaner, phoneme_processor)
    batch_size = 16
    dataloader = DataLoader(td, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for wave_path, phoneme_sequence in dataloader:
        print(f"音频文件路径: {wave_path}")
        print(f"音素序列: {phoneme_sequence.shape}")
