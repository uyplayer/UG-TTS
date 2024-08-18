import os.path

import numpy as np
import librosa
import soundfile as sf
import torch
from common.log_utils import get_logger
from scipy import signal
from common.path_config import data_dir

logger = get_logger("ug_tts")


class AudioProcess(object):
    _instance = None

    def __new__(cls):
        logger.info("AudioProcess is initialized")
        if cls._instance is None:
            cls._instance = super(AudioProcess, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        pass

    def load_audio(self, file_path, sr=None):
        """
        从文件中加载音频数据。
        :param file_path: 音频文件的路径
        :param sr: 目标采样率，如果为 None，则不进行重采样
        :return: 音频数据和采样率
        """
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate

    def save_audio(self, file_path, audio, sample_rate):
        """
        将音频数据保存到文件。
        :param file_path: 目标文件路径
        :param audio: 音频数据
        :param sample_rate: 采样率
        """
        sf.write(file_path, audio, sample_rate)

    def preprocess_audio(self, audio, sr, target_sr=None, noise_reduction=True, normalize=True, compression=True,
                         time_stretch=None):
        """
        对音频进行预处理，包括去噪、归一化、采样率转换、动态范围压缩和时间伸缩。
        :param audio: 音频数据
        :param sr: 采样率
        :param target_sr: 目标采样率，如果为 None，则不进行采样率转换
        :param noise_reduction: 是否进行去噪处理
        :param normalize: 是否进行归一化
        :param compression: 是否进行动态范围压缩
        :param time_stretch: 时间伸缩因子，如果为 None，则不进行时间伸缩
        :return: 处理后的音频数据
        """
        if normalize:
            audio = self._normalize_audio(audio)

        if noise_reduction:
            audio = self._reduce_noise(audio, sr)

        if target_sr is not None and target_sr != sr:
            audio = self._resample_audio(audio, sr, target_sr)

        if compression:
            audio = self._apply_compression(audio)

        if time_stretch is not None:
            audio = self._time_stretch(audio, time_stretch)

        return audio

    def trim_silence(self, audio, sr):
        """
        去除音频前后的静音部分。
        :param audio: 音频数据
        :param sr: 采样率
        :return: 去除静音后的音频数据
        """
        audio_trimmed, _ = librosa.effects.trim(audio)
        return audio_trimmed

    import librosa

    def extract_mel_spectrogram(self, audio, sr, n_mels=80, n_fft=2048, hop_length=512):
        """
        从音频数据中提取 Mel 频谱。

        :param audio: 音频数据（numpy 数组）
        :param sr: 采样率（整数）
        :param n_mels: Mel 频谱的 Mel 帧数（整数）
        :param n_fft: FFT 点数（整数）
        :param hop_length: 窗口滑动步长（整数）
        :return: Mel 频谱（numpy 数组）
        """
        # 确保 audio 是一个一维 numpy 数组
        if not isinstance(audio, np.ndarray) or audio.ndim != 1:
            raise ValueError("Audio data should be a one-dimensional numpy array.")

        # 调用 librosa 提取 Mel 频谱
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                         hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db

    def extract_mfcc(self, audio, sr, n_mfcc=13):
        """
        从音频数据中提取 MFCC 特征。

        :param audio: 音频数据（numpy 数组）
        :param sr: 采样率（整数）
        :param n_mfcc: MFCC 特征的数量（整数）
        :return: MFCC 特征（numpy 数组）
        """
        # 调用 librosa 提取 MFCC 特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        return mfcc

    def apply_data_augmentation(self, audio):
        """
        对音频进行数据增强，如添加噪声、改变音高、时间伸缩等。
        :param audio: 音频数据
        :return: 增强后的音频数据
        """
        # 示例: 添加白噪声
        noise = np.random.randn(len(audio))
        audio = audio + 0.005 * noise
        # 更多增强操作可以添加到这里
        return audio

    def spectrogram_to_waveform(self, mel_spectrogram, sr, method='griffinlim'):
        """
        将 Mel 频谱转换回音频波形。
        :param mel_spectrogram: Mel 频谱
        :param sr: 采样率
        :param method: 转换方法 ('griffinlim', 'waveglow', 'melgan')
        :return: 音频数据
        """
        if method == 'griffinlim':
            # 使用 Griffin-Lim 算法
            audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sr)
        elif method == 'waveglow':
            # 使用 WaveGlow 神经声码器
            raise NotImplementedError("WaveGlow not implemented")
        elif method == 'melgan':
            # 使用 MelGAN 神经声码器
            raise NotImplementedError("MelGAN not implemented")
        else:
            raise ValueError("Unsupported method")
        return audio

    def _normalize_audio(self, audio):
        """
        对音频进行归一化处理。
        :param audio: 音频数据
        :return: 归一化后的音频数据
        """
        return audio / np.max(np.abs(audio))

    def _reduce_noise(self, audio, sr):
        """
        对音频进行去噪处理。
        :param audio: 音频数据
        :param sr: 采样率
        :return: 去噪后的音频数据
        """
        cutoff_freq = 3000
        b, a = signal.butter(4, cutoff_freq / (0.5 * sr), btype='low')
        audio_denoised = signal.filtfilt(b, a, audio)
        return audio_denoised

    def _resample_audio(self, audio, sr, target_sr):
        """
        对音频进行采样率转换。
        :param audio: 音频数据
        :param sr: 当前采样率
        :param target_sr: 目标采样率
        :return: 重新采样后的音频数据
        """
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio_resampled

    def _apply_compression(self, audio, threshold=0.5, ratio=4):
        """
        对音频进行动态范围压缩。
        :param audio: 音频数据
        :param threshold: 压缩阈值
        :param ratio: 压缩比
        :return: 压缩后的音频数据
        """
        audio_compressed = np.copy(audio)
        # 压缩超过阈值的部分
        audio_compressed[audio > threshold] = threshold + (audio[audio > threshold] - threshold) / ratio
        return audio_compressed

    def _time_stretch(self, audio, stretch_factor):
        """
        对音频进行时间伸缩。
        :param audio: 音频数据
        :param stretch_factor: 时间伸缩因子
        :return: 伸缩后的音频数据
        """
        # 确保 audio 是 numpy 数组
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio data must be a numpy array")

        # 确保 stretch_factor 是数字
        if not isinstance(stretch_factor, (int, float)):
            raise TypeError("Stretch factor must be a number")

        # 确保 stretch_factor 大于0
        if stretch_factor <= 0:
            raise ValueError("Stretch factor must be greater than 0")

        # 调用 time_stretch 函数
        try:
            return librosa.effects.time_stretch(audio, rate=stretch_factor)
        except TypeError as e:
            raise RuntimeError(f"Error in time_stretch: {e}")


if __name__ == "__main__":
    audio_processor = AudioProcess()

    # 测试音频加载和保存
    file_path = os.path.join(data_dir, "LJ001-0001.wav")
    audio, sr = audio_processor.load_audio(file_path)
    print(f"Loaded audio with sample rate {sr}")

    output_path = os.path.join(data_dir, "output_audio.wav")
    audio_processor.save_audio(output_path, audio, sr)
    print(f"Saved audio to {output_path}")

    # 测试音频预处理
    preprocessed_audio = audio_processor.preprocess_audio(audio, sr, target_sr=16000, noise_reduction=True,
                                                          normalize=True, compression=True, time_stretch=1.2)
    print(f"Preprocessed audio with new sample rate: 16000")

    # 测试去除静音
    trimmed_audio = audio_processor.trim_silence(preprocessed_audio, sr)
    print(f"Trimmed silence from audio")

    # 测试提取 Mel 频谱
    mel_spectrogram = audio_processor.extract_mel_spectrogram(trimmed_audio, sr)
    print(f"Extracted Mel spectrogram with shape: {mel_spectrogram.shape}")

    # 测试提取 MFCC 特征
    mfcc = audio_processor.extract_mfcc(trimmed_audio, sr)
    print(f"Extracted MFCC with shape: {mfcc.shape}")

    # 测试数据增强
    augmented_audio = audio_processor.apply_data_augmentation(trimmed_audio)
    print(f"Applied data augmentation")

    # 测试 Mel 频谱转换回波形
    waveform = audio_processor.spectrogram_to_waveform(mel_spectrogram, sr, method='griffinlim')
    print(f"Converted Mel spectrogram back to waveform")

    # 验证 Librosa 模块
    print(dir(librosa))
