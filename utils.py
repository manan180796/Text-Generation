import numpy as np


class Vocablury(object):
    def __init__(self, text: str, file_path: str = None):
        if not file_path:
            self.vocab = sorted(set(text))
            self.char_to_index = {c: i for i, c in enumerate(self.vocab)}
            self.index_to_char = np.array(self.vocab)


class Text_Vectorizer(object):
    def __init__(self, text: str, vocablury: Vocablury = None):
        self.vocablury = vocablury
        if not self.vocablury:
            self.vocablury = Vocablury(text=text)

    def get_vocablury(self):
        return self.vocablury

    def vectorize(self, text):
        return np.array([self.vocablury.char_to_index[c] for c in text])
