from transformers import AutoTokenizer
from app.tokenizer.base_tokenizer import BaseTokenizer
import torch
import warnings
from abc import ABC
from common.log_utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class FairseqXLMRTokenizer(BaseTokenizer, ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('pytorch/fairseq:main', 'xlmr.large', weights_only=True)
        logger.info(f"Device type is {self.device.type}")
        logger.info(f"Xlmr large model is loaded")
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

    def tokenize(self, text):
        # Tokenize the text using the AutoTokenizer
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def encoding(self, text):
        text = text.to(self.device) if isinstance(text, torch.Tensor) else text
        return self.model.encode(text)

    def decoding(self, encoded):
        encoded = encoded.to('cpu') if isinstance(encoded, torch.Tensor) else encoded
        return self.model.decode(encoded)

    def verbose_each_char(self, text):
        for char in text:
            encoded_char = self.model.encode(char)
            decoded_char = self.model.decode(encoded_char)
            print(f"Character: {char} -> Encoded: {encoded_char.tolist()} -> Decoded: {decoded_char}")

    def verbose_each_word(self, text):
        for char in text:
            encoded_char = self.tokenizer.encode(char)
            decoded_char = self.tokenizer.decode(encoded_char)
            print(f"Character: {char} -> Encoded: {encoded_char} -> Decoded: {decoded_char}")

    def encoding_character_based(self, text):
        char_encoded = [self.model.encode(char) for char in text]
        return torch.cat(char_encoded, dim=0)

    def decoding_character_based(self, encoded):
        decoded_chars = [self.model.decode(encoded[i:i + 1]) for i in range(encoded.size(0))]
        return decoded_chars


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = FairseqXLMRTokenizer()
    print(t.tokenize(text))
    en = t.encoding(text)
    print(en)
    print(t.decoding(en))
    t.verbose_each_char(list(text))
    t.verbose_each_word(text.split(" "))
    c_e = t.encoding_character_based(text)
    print(c_e)
    print(t.decoding_character_based(c_e))
