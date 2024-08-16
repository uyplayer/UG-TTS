import warnings
import torch
from common.log_utils import get_logger
logger = get_logger("ug_tts")

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
        logger.info(f"xlmr.large model is loaded")
        self.model.to(self.device)

    def encoding(self, text):
        text = text.to(self.device) if isinstance(text, torch.Tensor) else text
        return self.model.encode(text)

    def decoding(self, encoded):
        encoded = encoded.to('cpu') if isinstance(encoded, torch.Tensor) else encoded
        return self.model.decode(encoded)


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = PreTrainedEncodeDecode()
    res = t.encoding(text)
    print(res)
