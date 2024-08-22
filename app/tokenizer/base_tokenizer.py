from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer
import warnings
from common.log_utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text):
        """
        对输入文本进行分词。
        :param text: 输入的文本字符串
        :return: 分词后的结果
        """
        pass

    @abstractmethod
    def encoding(self, text):
        """
        对输入文本进行编码，将文本转换为模型的输入格式。
        :param text: 输入的文本字符串
        :return: 编码后的结果
        """
        pass

    @abstractmethod
    def decoding(self, encoded):
        """
        对编码后的结果进行解码，将其转换回原始文本。
        :param encoded: 编码后的输入
        :return: 解码后的文本字符串
        """
        pass

    @abstractmethod
    def encoding_character_based(self, text):
        """
        基于字符对输入文本进行编码。
        :param text: 输入的文本字符串
        :return: 编码后的结果，按字符进行编码
        """
        pass

    @abstractmethod
    def decoding_character_based(self, encoded):
        """
        基于字符对编码结果进行解码。
        :param encoded: 编码后的输入 Tensor
        :return: 解码后的字符列表
        """
        pass

    @abstractmethod
    def verbose_each_char(self, text):
        """
        打印每一个字符对应的编码
        :param text:
        :return:
        """
        pass

    @abstractmethod
    def verbose_each_word(self, text):
        """
        打印每一个单词对应的编码
        :param text:
        :return:
        """
        pass
