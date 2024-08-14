# -*- coding: utf-8 -*-
'''
Defines the set of symbols used in text input to the model for Uyghur language.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from static.alphabet import character,vowels,consonants,punctuation,_vowels,_pulmonic_consonants

def make_symbols(characters, phonemes, punctuations='!\'(),-.:;? ', pad='_', eos='~', bos='^'):
    _phonemes_sorted = sorted(list(phonemes))

    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = ['@' + s for s in _phonemes_sorted]

    # Export all symbols:
    _symbols = [pad, eos, bos] + list(characters) + _arpabet
    _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)

    return _symbols, _phonemes

_pad = '_'
_eos = '~'
_bos = '^'
_characters = "".join(character)
_punctuations = "".join(punctuation)

# Define Uyghur phonemes
_vowels = "".join(_vowels)
_pulmonic_consonants = "".join(_pulmonic_consonants)
# 超段特征符号 (维吾尔语通常不需要)
suprasegmentals = ''
# 其他符号 (维吾尔语通常不需要)
other_symbols = ''
# 变音符号 (维吾尔语通常不需要)
diacritics = ''
_phonemes = _vowels + _pulmonic_consonants + suprasegmentals + other_symbols + diacritics

symbols, phonemes = make_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

if __name__ == '__main__':
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print(phonemes)
