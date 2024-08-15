import numpy as np
import librosa
import scipy
import soundfile as sf
from pydub import AudioSegment
from scipy import signal


class AudioProcessor:

    def __init__(self,
                 sample_rate=None,
                 resample=False,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 ref_level_db=None,
                 fft_size=1024,
                 power=None,
                 preemphasis=0.0,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 spec_gain=20,
                 stft_pad_mode='reflect',
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 do_sound_norm=False,
                 stats_path=None,
                 verbose=True,
                 **_):
        """
        初始化 AudioProcessor 类的实例，设置音频处理的各种参数。

        :param sample_rate: 音频采样率（Hz）。如果为 None，则使用音频文件的默认采样率。
        :param resample: 是否对音频进行重采样。如果为 True，则会根据 sample_rate 参数重采样音频。
        :param num_mels: 梅尔频率滤波器组的数量。如果为 None，则使用默认值。
        :param min_level_db: 最小级别的分贝数（dB），用于对信号进行归一化。如果为 None，则使用默认值。
        :param frame_shift_ms: 帧移的时间长度（毫秒），用于计算 STFT。如果为 None，则使用默认值。
        :param frame_length_ms: 帧长度的时间长度（毫秒），用于计算 STFT。如果为 None，则使用默认值。
        :param hop_length: 帧移的样本数（通常等于 frame_shift_ms）。如果为 None，则计算为采样率和帧移的比例。
        :param win_length: 帧长度的样本数（通常等于 frame_length_ms）。如果为 None，则计算为采样率和帧长度的比例。
        :param ref_level_db: 参考的分贝级别（dB），用于对信号进行归一化。如果为 None，则使用默认值。
        :param fft_size: FFT 的大小（样本点数），用于计算 STFT。默认为 1024。
        :param power: 功率谱的计算方式。如果为 None，则使用默认值。
        :param preemphasis: 预加重系数。用于信号预处理，防止高频噪声。如果为 0.0，则不进行预加重。
        :param signal_norm: 信号归一化的方式。如果为 None，则使用默认值。
        :param symmetric_norm: 对称归一化的设置。如果为 None，则使用默认值。
        :param max_norm: 最大归一化的设置。如果为 None，则使用默认值。
        :param mel_fmin: 梅尔频率滤波器组的最小频率（Hz）。如果为 None，则使用默认值。
        :param mel_fmax: 梅尔频率滤波器组的最大频率（Hz）。如果为 None，则使用默认值。
        :param spec_gain: 频谱增益因子，用于增强频谱的音量。如果为 20，则为默认值。
        :param stft_pad_mode: STFT 时的填充模式。可选值有 'reflect', 'constant', 'edge' 等。
        :param clip_norm: 是否对归一化后的信号进行裁剪。如果为 True，则会裁剪。
        :param griffin_lim_iters: Griffin-Lim 算法的迭代次数，用于频谱到时域的重建。如果为 None，则使用默认值。
        :param do_trim_silence: 是否进行静音修剪。如果为 True，则会去除音频中的静音部分。
        :param trim_db: 静音修剪的阈值（dB）。低于该值的部分会被去除。
        :param do_sound_norm: 是否对音频进行声音归一化。如果为 True，则会对音频进行归一化处理。
        :param stats_path: 声音归一化的统计信息路径。如果为 None，则不使用统计信息。
        :param verbose: 是否打印详细的日志信息。如果为 True，则打印详细信息。
        :param _: 其他额外的参数（如果有的话）。
        """

        self.sample_rate = sample_rate
        self.resample = resample
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.hop_length = hop_length
        self.win_length = win_length
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.max_norm = max_norm
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.spec_gain = spec_gain
        self.stft_pad_mode = stft_pad_mode
        self.clip_norm = clip_norm
        self.griffin_lim_iters = griffin_lim_iters
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.stats_path = stats_path
        self.verbose = verbose

        members = vars(self)
        if verbose:
            print(" Audio Processor 类参数 :")
            for key, value in members.items():
                print(" | > {}:{}".format(key, value))

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def normalize(self, S):
        # 归一化实现
        pass

    def denormalize(self, S):
        # 反归一化实现
        pass

    def _build_mel_basis(self):
        return librosa.filters.mel(
            self.sample_rate,
            self.fft_size,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def apply_preemphasis(self, x):
        return scipy.signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        # 实现静音检测
        pass

    def trim_silence(self, wav):
        # 实现静音修剪
        pass

    def load_wav(self, filename, sr=None):
        # 加载音频文件
        pass

    def save_wav(self, wav, path):
        # 保存音频文件
        pass

    @staticmethod
    def mulaw_encode(wav, qc):
        # 量化编码
        pass

    @staticmethod
    def mulaw_decode(wav, qc):
        # 解码
        pass
