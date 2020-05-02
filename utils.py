import numpy as np


class Vocablury(object):
    def __init__(self, text: str = None, file_path: str = None):
        if not file_path:
            self.vocab = sorted(set(text))
            self.char_to_index = {c: i for i, c in enumerate(self.vocab)}
            self.index_to_char = np.array(self.vocab)

    def __len__(self):
        return len(self.vocab)


class Text_Vectorizer(object):
    def __init__(self, text: str = None, vocablury: Vocablury = None):
        self.vocablury = vocablury
        if not self.vocablury:
            self.vocablury = Vocablury(text=text)

    def get_vocablury(self) -> Vocablury:
        return self.vocablury

    def vectorize(self, text) -> np.array:
        return np.array([self.vocablury.char_to_index[c] for c in text])
