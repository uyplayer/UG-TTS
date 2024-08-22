import unittest
from enum import Enum, unique
import torch
from transformers import AutoTokenizer

from app.tokenizer.FairseqXLMRTokenizer import FairseqXLMRTokenizer
from app.tokenizer.XLMRobertaTokenizer import XLMRobertaTokenizer
from app.tokenizer.base_tokenizer import BaseTokenizer
from common.log_utils import get_logger

logger = get_logger(__name__)


@unique
class TokenizerType(Enum):
    XLM_ROBERTA_BASE = "xlm-roberta-base"
    PYTORCH_FAIRSEQ_XLMR_LARGE = "pytorch/fairseq:main_xlmr.large"


class Tokenizer(object):
    def __init__(self, tokenizer_type: TokenizerType):
        if tokenizer_type == TokenizerType.XLM_ROBERTA_BASE:
            self.tokenizer = XLMRobertaTokenizer()
        elif tokenizer_type == TokenizerType.PYTORCH_FAIRSEQ_XLMR_LARGE:
            self.tokenizer = FairseqXLMRTokenizer()
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encoding(self, text):
        return self.tokenizer.encoding(text)

    def decoding(self, encoded):
        return self.tokenizer.decoding(encoded)

    def encoding_character_based(self, text):
        return self.tokenizer.encoding_character_based(text)

    def decoding_character_based(self, encoded):
        return self.tokenizer.decoding_character_based(encoded)

    def verbose_each_char(self, text):
        return self.tokenizer.verbose_each_char(list(text))

    def verbose_each_word(self, text):
        return self.tokenizer.verbose_each_word(text)


if __name__ == "__main__":
    # XLM_ROBERTA_BASE
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = Tokenizer(TokenizerType.XLM_ROBERTA_BASE)
    print(t.tokenize(text))
    en = t.encoding(text)
    print(en)
    print(t.decoding(en))
    t.verbose_each_char(list(text))
    t.verbose_each_word(text.split(" "))
    c_e = t.encoding_character_based(text)
    print(c_e)
    print(t.decoding_character_based(c_e))

    # PYTORCH_FAIRSEQ_XLMR_LARGE
    t = Tokenizer(TokenizerType.PYTORCH_FAIRSEQ_XLMR_LARGE)
    print(t.tokenize(text))
    en = t.encoding(text)
    print(en)
    print(t.decoding(en))
    t.verbose_each_char(list(text))
    t.verbose_each_word(text.split(" "))
    c_e = t.encoding_character_based(text)
    print(c_e)
    print(t.decoding_character_based(c_e))
