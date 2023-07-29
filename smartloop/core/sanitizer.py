from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from nltk import PorterStemmer

from smartloop.core import LANG_MAPS


class Sanitizer:
    """
        Remove punctuations and stop words
    """

    def __init__(self, lang='en'):
        self.lang = lang

    def transform(self, data):
        reg_exp_tokenizer = RegexpTokenizer(r'\w+')
        words = stopwords.words(LANG_MAPS[self.lang])
        tokens = reg_exp_tokenizer.tokenize(data.lower())

        temp = []
        stemmer = PorterStemmer()

        for token in tokens:
            if not token in words:
                temp.append(stemmer.stem(token))

        if len(temp) > 0:
            return ' '.join(temp)

        return data
