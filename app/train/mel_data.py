import torch


def get_mel_targets(wave_paths, audio_processor, target_length=250,n_mels=250):
    mel_targets = []
    for path in wave_paths:
        audio, sr = audio_processor.load_audio(path, sr=22050)
        audio = audio_processor.preprocess_audio(audio, sr, target_sr=22050)
        mel_spectrogram = audio_processor.extract_mel_spectrogram(audio, sr, n_mels=n_mels)
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        mel_spectrogram = pad_or_crop(mel_spectrogram, target_length)

        mel_targets.append(mel_spectrogram)
    return torch.stack(mel_targets)


def pad_or_crop(mel_spectrogram, target_length):
    current_length = mel_spectrogram.shape[1]
    if current_length > target_length:
        mel_spectrogram = mel_spectrogram[:, :target_length]
    elif current_length < target_length:
        # 填充
        pad_size = target_length - current_length
        mel_spectrogram = torch.cat([mel_spectrogram, torch.zeros(mel_spectrogram.shape[0], pad_size)], dim=1)
    return mel_spectrogram
