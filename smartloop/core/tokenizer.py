from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from smartloop.core import Config, DefaultConfig


class SentenceTokenizer:
    """
        Tokenize sentences to sequences
    """

    max_input_length: int

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.cfg = DefaultConfig()

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, self.max_input_length, padding='post', truncating='post')
