import re
from config.alphabet import punctuation, character
from common.log_utils import get_logger
from config.dictionary import english_dictionary
from config.special_symbols import money_symbols_mapping, punctuation_mapping
from app.cleaner.num2str import num2str

logger = get_logger(__name__)


def _normalize_punctuation(text):
    for punc, replacement in punctuation_mapping.items():
        text = text.replace(punc, replacement)
    return text


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
        text = text.upper()
        text = _normalize_punctuation(text)
        text = self._normalize_currency(text)
        text = self._normalize_number(text)
        text = self._normalize_english_words(text)
        text = self._remove_extra_spaces(text)
        text = self._remove_invalid_characters(text)
        return text

    @classmethod
    def _normalize_currency(cls, text):
        for symbol, translation in money_symbols_mapping.items():
            text = text.replace(symbol, translation)
        return text

    @classmethod
    def _normalize_number(cls, text):
        text = re.sub(r'\d+', lambda x: num2str(int(x.group())), text)
        return text

    @classmethod
    def _normalize_english_words(cls, text):

        def replace_match(match):
            word = match.group(0)
            if word in english_dictionary.keys():
                return english_dictionary.get(word.upper(), word)
            else:
                characters = list(word)
                replaced_characters = [english_dictionary.get(char.upper(), char) for char in characters]
                return ''.join(replaced_characters)

        text = re.sub(r'\b[A-Za-z]+\b', replace_match, text)
        return text

    @classmethod
    def _remove_extra_spaces(cls, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def _remove_invalid_characters(cls, text):
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
