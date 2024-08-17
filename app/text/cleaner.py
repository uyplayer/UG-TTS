import re
from config.alphabet import punctuation


class TextCleaner(object):
    def __init__(self):
        pass

    def clean_text(self, text):
        pass

    def _normalize_punctuation(self, text):
        pass

    def _normalize_currency(self):
        pass

    def _normalize_number(self):
        pass

    def _remove_extra_spaces(self, text):
        # 替换连续多个空格为一个空格
        # 去除句首和句尾的空格
        pass

    def _remove_invalid_characters(self, text):
        pass


if __name__ == '__main__':
    text = "  ئايدا  ئىككى! قېتىم...  "
