import epitran
from common.log_utils import get_logger

logger = get_logger("ug_tts")


class IPA(object):
    _instance = None

    def __new__(cls):
        logger.info(f"IPA is initialized")
        if cls._instance is None:
            cls._instance = super(IPA, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.ipa = epitran.Epitran('uig-Arab')

    def transliterate(self, text):
        return self.ipa.transliterate(text)


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    ipa = IPA()
    res = ipa.transliterate(text)
    print(res)
