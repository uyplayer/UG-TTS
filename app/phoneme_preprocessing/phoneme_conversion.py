
from config.alphabet import international_to_ipa_all, character, vowels, consonants, international_to_ipa, punctuation
from common.log_utils import get_logger
logger = get_logger(__name__)


class Phoneme(object):
    _instance = None

    def __new__(cls):
        """
        确保 Phoneme 类为单例模式。
        """
        logger.info(f"Phoneme is initialized")
        if cls._instance is None:
            cls._instance = super(Phoneme, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        初始化 Phoneme 实例，包括构建音素到 ID 的映射。
        """
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self._build_phoneme_map()

    def _build_phoneme_map(self):
        """
        构建音素到 ID 的映射以及 ID 到音素的映射。
        """
        all_chars = list(international_to_ipa_all.values()) + punctuation
        for i, phoneme in enumerate(all_chars):
            self.phoneme_to_id[phoneme] = i
            self.id_to_phoneme[i] = phoneme
        logger.debug(f"Phoneme to ID map: {self.phoneme_to_id}")
        logger.debug(f"ID to phoneme map: {self.id_to_phoneme}")

    def _find_phoneme(self, char):
        """
        查找字符对应的音素，如果未找到则返回字符本身。
        :param char: 字符
        :return: 对应的音素或字符本身
        """
        phoneme = international_to_ipa.get(char)
        if phoneme is None:
            # logger.warning(f"Phoneme not found for character: {char}")
            phoneme = char
        return phoneme

    def phoneme(self, text):
        """
        将文本转换为音素序列。
        :param text: 输入文本
        :return: 音素列表
        """
        character_list = list(text)
        index = 0
        phoneme_res = []
        while index < len(character_list):
            char = character_list[index]
            phoneme = None
            if char in vowels:
                if char == "ئ":
                    if index + 1 < len(character_list):
                        head = char + character_list[index + 1]
                        phoneme = self._find_phoneme(head)
                        if phoneme != head:  # 如果 head 不是音素本身
                            index += 2
                        else:
                            phoneme = self._find_phoneme(char)
                            index += 1
                    else:
                        phoneme = self._find_phoneme(char)
                        index += 1
                else:
                    phoneme = self._find_phoneme(char)
                    index += 1
            elif char in consonants:
                phoneme = self._find_phoneme(char)
                index += 1
            else:
                phoneme = char
                index += 1

            if phoneme is not None:
                phoneme_res.append(phoneme)
                logger.debug(f"Processed character: {char}, Phoneme: {phoneme}")

        logger.info(f"Generated phonemes for symbol: {phoneme_res}")
        return phoneme_res

    def phoneme_to_sequence(self, phonemes):
        """
        将音素列表转换为音素 ID 序列。
        :param phonemes: 音素列表
        :return: 音素 ID 序列
        """
        sequence = [self.phoneme_to_id[p] for p in phonemes if p in self.phoneme_to_id]
        logger.info(f"Phoneme sequence: {sequence}")
        return sequence

    def sequence_to_phoneme(self, sequence):
        """
        将音素 ID 序列转换回音素列表。
        :param sequence: 音素 ID 序列
        :return: 重建的音素列表
        """
        phonemes = ' '.join([self.id_to_phoneme[i] for i in sequence])
        logger.info(f"Reconstructed phonemes: {phonemes}")
        return phonemes


if __name__ == '__main__':

    phoneme_processor = Phoneme()
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    phonemes = phoneme_processor.phoneme(text)
    sequence = phoneme_processor.phoneme_to_sequence(phonemes)
    reconstructed_phonemes = phoneme_processor.sequence_to_phoneme(sequence)

    logger.info(f"Original symbol: {text}")
    logger.info(f"Phonemes: {phonemes}")
    logger.info(f"Sequence: {sequence}")
    logger.info(f"Reconstructed phonemes: {reconstructed_phonemes}")
    print(f"phonemes = {len(phonemes)}")
    print(f"sequence = {len(sequence)}")
    print(f"reconstructed_phonemes = {len(reconstructed_phonemes)}")