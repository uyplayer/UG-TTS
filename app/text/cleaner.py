import re
from config.alphabet import punctuation, character
from common.log_utils import get_logger
from config.special_symbols import money_symbols_mapping, punctuation_mapping
from app.text.num2str import num2str

logger = get_logger("ug_tts")


class TextCleaner(object):
    _instance = None

    def __new__(cls):
        logger.info("TextCleaner is initialized")
        if cls._instance is None:
            cls._instance = super(TextCleaner, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        pass

    def clean_text(self, text):
        text = self._normalize_punctuation(text)
        text = self._normalize_currency(text)
        text = self._normalize_number(text)
        text = self._remove_extra_spaces(text)
        text = self._remove_invalid_characters(text)
        return text

    def _normalize_punctuation(self, text):
        for punc, replacement in punctuation_mapping.items():
            text = text.replace(punc, replacement)
        return text

    def _normalize_currency(self, text):
        for symbol, translation in money_symbols_mapping.items():
            text = text.replace(symbol, translation)
        return text

    def _normalize_number(self, text):
        text = re.sub(r'\d+', lambda x: num2str(int(x.group())), text)
        return text

    def _remove_extra_spaces(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _remove_invalid_characters(self, text):
        valid_characters = ''.join(character)
        valid_punctuation = ''.join(punctuation)
        valid_symbols = r'\s\d'
        valid_pattern = f'[^{valid_characters}{valid_punctuation}{valid_symbols}]'

        text = re.sub(valid_pattern, '', text)
        return text


if __name__ == '__main__':
    text = "  ئايدا 1111 ئىككى! قېتىم...$ fff "
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_text(text)
    print(cleaned_text)
