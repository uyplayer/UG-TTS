import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr


class AudioResampler:
    def __init__(self, sample_rate=None):
        self.sample_rate = sample_rate

    def resample(self, audio, orig_sr):
        if self.sample_rate is None:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)


class AudioNormalizer:
    def __init__(self, min_level_db=None, max_norm=None):
        self.min_level_db = min_level_db
        self.max_norm = max_norm

    def normalize(self, audio):
        audio = np.clip(audio, -1, 1)
        if self.max_norm is not None:
            audio = audio / max(np.max(np.abs(audio)), 1e-8) * self.max_norm
        return audio


class AudioTrimmer:
    def __init__(self, do_trim_silence=False, trim_db=60):
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db

    def trim(self, audio):
        if not self.do_trim_silence:
            return audio
        # 静音修剪逻辑
        return librosa.effects.trim(audio, top_db=self.trim_db)[0]


class AudioFeatureExtractor:
    def __init__(self, num_mels=80, fft_size=1024):
        self.num_mels = num_mels
        self.fft_size = fft_size

    def extract_features(self, audio):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_mels=self.num_mels, n_fft=self.fft_size)
        return mel_spectrogram


class AudioFrameSplitter:
    def __init__(self, frame_length_ms=50, frame_shift_ms=12.5):
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms

    def split(self, audio):
        frame_length = int(self.frame_length_ms / 1000 * len(audio))
        frame_shift = int(self.frame_shift_ms / 1000 * len(audio))
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_shift)
        return frames


class AudioSTFT:
    def __init__(self, fft_size=1024):
        self.fft_size = fft_size

    def compute(self, audio):
        stft = librosa.stft(audio, n_fft=self.fft_size)
        return stft


class GriffinLim:
    def __init__(self, griffin_lim_iters=60):
        self.griffin_lim_iters = griffin_lim_iters

    def reconstruct(self, magnitude, phase=None):
        audio = librosa.istft(magnitude, hop_length=512, win_length=1024, window='hann', length=None)
        return audio


class AudioDenoiser:
    def __init__(self):
        pass

    def denoise(self, audio):
        return nr.reduce_noise(y=audio)


class AudioSaver:
    def __init__(self):
        pass

    def save(self, audio, filename, sample_rate):
        sf.write(filename, audio, sample_rate)


class AudioProcessor:
    def __init__(self, config):
        self.resampler = AudioResampler(sample_rate=config.get('sample_rate'))
        self.normalizer = AudioNormalizer(
            min_level_db=config.get('min_level_db'),
            max_norm=config.get('max_norm'))
        self.trimmer = AudioTrimmer(
            do_trim_silence=config.get('do_trim_silence'),
            trim_db=config.get('trim_db'))
        self.feature_extractor = AudioFeatureExtractor(
            num_mels=config.get('num_mels'),
            fft_size=config.get('fft_size'))
        self.frame_splitter = AudioFrameSplitter(
            frame_length_ms=config.get('frame_length_ms'),
            frame_shift_ms=config.get('frame_shift_ms'))
        self.stft = AudioSTFT(fft_size=config.get('fft_size'))
        self.griffin_lim = GriffinLim(griffin_lim_iters=config.get('griffin_lim_iters'))
        self.denoiser = AudioDenoiser()
        self.saver = AudioSaver()

    def process(self, audio, orig_sr):
        audio = self.resampler.resample(audio, orig_sr)
        audio = self.trimmer.trim(audio)
        audio = self.normalizer.normalize(audio)
        features = self.feature_extractor.extract_features(audio)
        frames = self.frame_splitter.split(audio)
        stft = self.stft.compute(audio)
        reconstructed_audio = self.griffin_lim.reconstruct(stft)
        denoised_audio = self.denoiser.denoise(reconstructed_audio)
        return features, frames, stft, denoised_audio

    def save(self, audio, filename, sample_rate):
        self.saver.save(audio, filename, sample_rate)


if __name__ == "__main__":
    # 配置示例
    config = {
        'sample_rate': 22050,
        'num_mels': 80,
        'fft_size': 1024,
        'frame_length_ms': 50,
        'frame_shift_ms': 12.5,
        'do_trim_silence': True,
        'trim_db': 60,
        'griffin_lim_iters': 60,
        'max_norm': 1.0,
        'min_level_db': -100
    }

    processor = AudioProcessor(config)
    audio, sr = librosa.load('input.wav', sr=None)
    features, frames, stft, denoised_audio = processor.process(audio, orig_sr=sr)
    processor.save(denoised_audio, 'output.wav', sample_rate=sr)
