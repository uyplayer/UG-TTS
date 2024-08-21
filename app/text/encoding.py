import warnings
import torch
from common.log_utils import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


class PreTrainedEncodeDecode(object):
    _instance = None

    def __new__(cls):
        logger.info(f"PreTrainedEncodeDecode is initialized")
        if cls._instance is None:
            cls._instance = super(PreTrainedEncodeDecode, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('pytorch/fairseq:main', 'xlmr.large', weights_only=True)
        logger.info(f"Device type is {self.device.type}")
        logger.info(f"Xlmr large model is loaded")
        self.model.to(self.device)

    def encoding(self, text):
        text = text.to(self.device) if isinstance(text, torch.Tensor) else text
        return self.model.encode(text)

    def decoding(self, encoded):
        encoded = encoded.to('cpu') if isinstance(encoded, torch.Tensor) else encoded
        return self.model.decode(encoded)

    def encode_each_char(self, text):
        for char in text:
            encoded_char = self.model.encode(char)
            decoded_char = self.model.decode(encoded_char)
            print(f"Character: {char} -> Encoded: {encoded_char.tolist()} -> Decoded: {decoded_char}")

    def encoding_character_based(self, text):
        char_encoded = [self.model.encode(char) for char in text]
        return torch.cat(char_encoded, dim=0)

    def decoding_character_based(self, encoded):
        decoded_chars = [self.model.decode(encoded[i:i + 1]) for i in range(encoded.size(0))]
        return decoded_chars


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = PreTrainedEncodeDecode()
    char_encoded = t.encoding_character_based(text)
    print("Character-based encoding:", char_encoded.tolist())
    decoded_text = t.decoding_character_based(char_encoded)
    print("Character-based decoded text:", decoded_text)
