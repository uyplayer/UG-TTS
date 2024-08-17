import epitran
from common.log_utils import get_logger

logger = get_logger("ug_tts")


class Phoneme (object):
    _instance = None

    def __new__(cls):
        logger.info(f"Phoneme is initialized")
        if cls._instance is None:
            cls._instance = super(Phoneme , cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        pass


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."

