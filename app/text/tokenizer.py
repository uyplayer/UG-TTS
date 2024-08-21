import torch
from transformers import AutoTokenizer
import warnings

from common.log_utils import get_logger

# 设置日志记录器
logger = get_logger(__name__)
# 忽略用户警告
warnings.filterwarnings("ignore", category=UserWarning)


class PreTrainedTokenizer(object):
    _instance = None

    def __new__(cls):
        """
        单例模式实现，确保 PreTrainedTokenizer 只会被初始化一次。
        """
        logger.info(f"PreTrainedTokenizer is initialized")
        if cls._instance is None:
            cls._instance = super(PreTrainedTokenizer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        初始化 tokenizer，加载预训练的模型。
        """
        self.model_name = "xlm-roberta-base"  # 预训练模型的名称
        self.model = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)  # 加载模型
        logger.info(f"{self.model_name} model is loaded")

    def tokenize(self, text):
        """
        对输入文本进行分词。
        :param text: 输入的文本字符串
        :return: 分词后的结果
        """
        return self.model.tokenize(text)

    def encoding(self, text):
        """
        对输入文本进行编码，将文本转换为模型的输入格式。
        :param text: 输入的文本字符串
        :return: 编码后的结果
        """
        return self.model.encode(text)

    def decoding(self, encoded):
        """
        对编码后的结果进行解码，将其转换回原始文本。
        :param encoded: 编码后的输入
        :return: 解码后的文本字符串
        """
        return self.model.decode(encoded)

    def encoding_character_based(self, text):
        """
        基于字符对输入文本进行编码。
        :param text: 输入的文本字符串
        :return: 编码后的结果，按字符进行编码
        """
        text = text.to(self.device) if isinstance(text, torch.Tensor) else text
        char_encoded = [self.model.encode(char) for char in text]  # 对每个字符进行编码
        char_encoded = [torch.tensor(enc) if not isinstance(enc, torch.Tensor) else enc for enc in char_encoded]
        return torch.cat(char_encoded, dim=0)  # 将编码结果合并为一个 Tensor

    def decoding_character_based(self, encoded):
        """
        基于字符对编码结果进行解码。
        :param encoded: 编码后的输入 Tensor
        :return: 解码后的字符列表
        """
        encoded = encoded.to('cpu') if isinstance(encoded, torch.Tensor) else encoded
        decoded_chars = [self.model.decode(encoded[i:i + 1]) for i in range(encoded.size(0))]  # 解码每个字符
        return decoded_chars


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = PreTrainedTokenizer()  # 创建 PreTrainedTokenizer 实例
    res = t.tokenize(text)  # 对文本进行分词
    print(res)
    r1 = t.encoding_character_based(text)  # 基于字符进行编码
    print(r1)
    r2 = t.decoding_character_based(t.encoding_character_based(text))  # 基于字符进行解码
    print(r2)
    print(len(r1), len(r2))  # 打印编码和解码后的长度
