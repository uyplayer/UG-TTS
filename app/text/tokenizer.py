import torch
from transformers import AutoTokenizer
import warnings

from common.log_utils import get_logger

logger = get_logger("ug_tts")
warnings.filterwarnings("ignore", category=UserWarning)


class PreTrainedTokenizer(object):
    _instance = None

    def __new__(cls):
        logger.info(f"PreTrainedTokenizer is initialized")
        if cls._instance is None:
            cls._instance = super(PreTrainedTokenizer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model_name = "xlm-roberta-base"
        self.model = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        logger.info(f"{self.model_name} model is loaded")

    def tokenize(self, text):
        return self.model.tokenize(text)

    def encoding(self, text):
        return self.model.encode(text)

    def decoding(self, encoded):
        return self.model.decode(encoded)

    def encoding_character_based(self, text):
        text = text.to(self.device) if isinstance(text, torch.Tensor) else text
        char_encoded = [self.model.encode(char) for char in text]
        char_encoded = [torch.tensor(enc) if not isinstance(enc, torch.Tensor) else enc for enc in char_encoded]
        return torch.cat(char_encoded, dim=0)

    def decoding_character_based(self, encoded):
        encoded = encoded.to('cpu') if isinstance(encoded, torch.Tensor) else encoded
        decoded_chars = [self.model.decode(encoded[i:i + 1]) for i in range(encoded.size(0))]
        return decoded_chars


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = PreTrainedTokenizer()
    res = t.tokenize(text)
    print(res)
    r1 = t.encoding_character_based(text)
    print(r1)
    r2 = t.decoding_character_based(t.encoding_character_based(text))
    print(r2)
    print(len(r1),len(r2))
