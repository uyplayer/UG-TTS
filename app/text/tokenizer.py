from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class PreTrainedTokenizer(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PreTrainedTokenizer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model_name = "xlm-roberta-base"
        self.model = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def tokenize(self, text):
        return self.model.tokenize(text)

    def encoding(self, text):
        return self.model.encode(text)

    def decoding(self, encoded):
        return self.model.decode(encoded)


if __name__ == "__main__":
    text = "ئايدا ئىككى قېتىم دەرسكە كەلمىگەن ئوقۇغۇچىلار دەرستىن چېكىندۈرۈلىدۇ."
    t = PreTrainedTokenizer()
    res = t.tokenize(text)
    print(res)
    print(t.encoding(text))
    print(t.decoding(t.encoding(text)))
