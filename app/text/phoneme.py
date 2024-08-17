from common.log_utils import get_logger
from config.alphabet import international_to_ipa_all, character, vowels, consonants

logger = get_logger("ug_tts")


class Phoneme(object):
    _instance = None

    def __new__(cls):
        logger.info(f"Phoneme is initialized")
        if cls._instance is None:
            cls._instance = super(Phoneme, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self._build_phoneme_map()

    def _build_phoneme_map(self):
        for i, phoneme in enumerate(international_to_ipa_all.values()):
            self.phoneme_to_id[phoneme] = i
            self.id_to_phoneme[i] = phoneme

    def _find_phoneme(self, char):
        # todo 实现查找char对应音素的逻辑
        return "None"

    def phoneme(self, text):
        character_list = list(text)
        index = 0
        phoneme_res = []
        while index < len(character_list):
            char = character_list[index]
            phoneme = None
            if char in vowels:
                # 处理元音“ئ”的特殊情况
                if char == "ئ":
                    if index + 1 < len(character_list):
                        head = char + character_list[index + 1]
                        phoneme = self._find_phoneme(head)
                        index += 2
                    else:
                        phoneme = self._find_phoneme(char)
                        index += 1
                else:
                    phoneme = self._find_phoneme(char)
                    index += 1
            elif char in consonants:
                # 辅音处理
                phoneme = self._find_phoneme(char)
                index += 1
            else:
                phoneme = char
                index += 1
            if phoneme is not None:
                phoneme_res.append(phoneme)
        return phoneme_res

    def phoneme_to_sequence(self, phonemes):
        sequence = [self.phoneme_to_id[p] for p in phonemes if p in self.phoneme_to_id]
        return sequence

    def sequence_to_phoneme(self, sequence):
        ph = ""
        phonemes = ''.join([self.id_to_phoneme[i] for i in sequence])
        return phonemes


if __name__ == '__main__':
    phoneme_processor = Phoneme()
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    print(set(list(text)))
    phonemes = phoneme_processor.phoneme(text)
    print(phonemes)
    # sequence = phoneme_processor.phoneme_to_sequence(phonemes)
    # reconstructed_phonemes = phoneme_processor.sequence_to_phoneme(sequence)
    # logger.info(f"Original text: {text}")
    # logger.info(f"Phonemes: {phonemes}")
    # logger.info(f"Sequence: {sequence}")
    # logger.info(f"Reconstructed phonemes: {reconstructed_phonemes}")
