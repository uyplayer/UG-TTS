from transformers import AutoTokenizer
from app.tokenizer.base_tokenizer import BaseTokenizer
import torch
import warnings
from abc import ABC
from common.log_utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class XLMRobertaTokenizer(BaseTokenizer, ABC):

    def __init__(self):
        logger.info(f"XLMRobertaTokenizer is initialized")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
        logger.info(f"Device type is {self.device.type}")
        logger.info(f"XLM-RoBERTa model is loaded")

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encoding(self, text):
        return self.tokenizer.encode(text, return_tensors='pt').to(self.device)

    def decoding(self, encoded):
        encoded = encoded.cpu() if encoded.is_cuda else encoded
        return self.tokenizer.decode(encoded.squeeze().tolist(), skip_special_tokens=True)

    def verbose_each_char(self, text):
        for char in text:
            encoded_char = self.tokenizer.encode(char)
            decoded_char = self.tokenizer.decode(encoded_char)
            print(f"Character: {char} -> Encoded: {encoded_char} -> Decoded: {decoded_char}")

    def verbose_each_word(self, text):
        for char in text:
            encoded_char = self.tokenizer.encode(char)
            decoded_char = self.tokenizer.decode(encoded_char)
            print(f"Character: {char} -> Encoded: {encoded_char} -> Decoded: {decoded_char}")

    def encoding_character_based(self, text):
        char_encoded_tensors = [torch.tensor(encoded_char) for encoded_char in
                                [self.tokenizer.encode(char) for char in text]]
        char_encoded = torch.cat(char_encoded_tensors, dim=0)
        return char_encoded

    def decoding_character_based(self, encoded):
        decoded_chars = [self.tokenizer.decode(encoded[i:i + 1]) for i in range(encoded.size(0))]
        return decoded_chars


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = XLMRobertaTokenizer()
    print(t.tokenize(text))
    en = t.encoding(text)
    print(en)
    print(t.decoding(en))
    t.verbose_each_char(list(text))
    t.verbose_each_word(text.split(" "))
    c_e = t.encoding_character_based(text)
    print(c_e)
    print(t.decoding_character_based(c_e))
