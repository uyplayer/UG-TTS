import torch


def get_mel_targets(wave_paths, audio_processor, target_length=250, n_mels=250):
    mel_targets = []
    for path in wave_paths:
        audio, sr = audio_processor.load_audio(path, sr=22050)
        audio = audio_processor.preprocess_audio(audio, sr, target_sr=22050)
        mel_spectrogram = audio_processor.extract_mel_spectrogram(audio, sr, n_mels=n_mels)
        if mel_spectrogram is None:
            print(f"Warning: Mel spectrogram is None for file {path}")
            continue
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        # Normalize each spectrogram
        mel_mean = mel_spectrogram.mean(dim=1, keepdim=True)
        mel_std = mel_spectrogram.std(dim=1, keepdim=True)
        mel_spectrogram = (mel_spectrogram - mel_mean) / (mel_std + 1e-6)
        # Pad or crop the spectrogram
        mel_spectrogram = pad_or_crop(mel_spectrogram, target_length)
        mel_targets.append(mel_spectrogram)
    if len(mel_targets) == 0:
        print("Error: No mel_targets generated")
        return None
    return torch.stack(mel_targets)


def pad_or_crop(mel_spectrogram, target_length):
    current_length = mel_spectrogram.shape[1]
    if current_length > target_length:
        mel_spectrogram = mel_spectrogram[:, :target_length]
    elif current_length < target_length:
        pad_size = target_length - current_length
        mel_spectrogram = torch.cat([mel_spectrogram, torch.zeros(mel_spectrogram.shape[0], pad_size)], dim=1)
    return mel_spectrogram
