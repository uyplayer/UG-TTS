# -*- coding: utf-8 -*-

from config.alphabet import character_all, punctuation, vowels_all, consonants_all


def do_symbols(characters, phonemes, punctuations='!\'(),-.:;؟؟؟', pad='_', eos='~', bos='^'):
    """
    生成符号和音素列表。
    :param characters: 所有字符的集合
    :param phonemes: 音素的集合
    :param punctuations: 标点符号的集合
    :param pad: 填充符号
    :param eos: 句子结束符号
    :param bos: 句子开始符号
    :return: 符号列表和音素列表
    """
    # 对音素进行排序并添加 "@" 前缀以确保唯一性
    _phonemes_sorted = sorted(list(phonemes))
    _arpabet = ['@' + s for s in _phonemes_sorted]

    # 生成符号列表，包括填充符号、句子结束符、句子开始符号、所有字符和音素
    _symbols = [pad, eos, bos] + list(characters) + _arpabet

    # 生成音素列表，包括填充符号、句子结束符、句子开始符号、所有音素和标点符号
    _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)

    return _symbols, _phonemes


# 定义符号和音素的默认值
_pad = '_'
_eos = '~'
_bos = '^'
_characters = character_all
_punctuations = punctuation
_vowels = vowels_all
_consonants = consonants_all
_phonemes = _vowels + _consonants

# 调用 do_symbols 函数生成符号和音素列表
symbols, phonemes = do_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

if __name__ == '__main__':
    # 打印符号和音素的数量及其内容
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print(phonemes)
