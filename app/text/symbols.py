# -*- coding: utf-8 -*-

from config.alphabet import character_all, punctuation, vowels_all, consonants_all


def do_symbols(characters, phonemes, punctuations='!\'(),-.:;؟؟؟', pad='_', eos='~', bos='^'):
    _phonemes_sorted = sorted(list(phonemes))

    # Prepend "@" to phonemes to ensure uniqueness:
    _arpabet = ['@' + s for s in _phonemes_sorted]

    # Export all symbols:
    _symbols = [pad, eos, bos] + list(characters) + _arpabet
    _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)

    return _symbols, _phonemes


_pad = '_'
_eos = '~'
_bos = '^'
_characters = character_all


_punctuations = punctuation


_vowels = vowels_all
_consonants = consonants_all
_phonemes = _vowels + _consonants

symbols, phonemes = do_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)

if __name__ == '__main__':
    print(" > TTS symbols {}".format(len(symbols)))
    print(symbols)
    print(" > TTS phonemes {}".format(len(phonemes)))
    print(phonemes)
