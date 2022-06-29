from nltk import word_tokenize

from modules.data_processor.denoiser import expand_contractions


def tokenize(text):
    text = expand_contractions(text)
    return word_tokenize(text)
