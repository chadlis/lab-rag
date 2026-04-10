import string

from nltk.stem import PorterStemmer

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
STEMMER = PorterStemmer()


def tokenize(text: str) -> list[str]:
    return text.lower().translate(PUNCTUATION_TABLE).split()


def filter_and_stem(text: str, stopwords: set[str], stemmer=STEMMER) -> list[str]:
    return [stemmer.stem(t) for t in tokenize(text) if t not in stopwords]
