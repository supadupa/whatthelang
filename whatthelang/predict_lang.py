import fasttext
from os import path
import re

MODEL_FILE = path.join(path.dirname(__file__), 'model', 'lid.176.ftz')


class WhatTheLang(object):
    def __init__(self):
        self.model_file = MODEL_FILE
        self.model = self.load_model()

    def load_model(self):
        return fasttext.load_model(self.model_file)

    def _clean_up(self, txt):
        txt = re.sub(r"\b\d+\b", "", txt)
        return txt

    def _flatten(self, pred):
        return [
            self.lang_from_label(item[0]) if item else None
            for item in pred
        ]

    @staticmethod
    def lang_from_label(label):
        return label.replace('__label__', '')

    def known_langs(self):
        return [
            self.lang_from_label(label)
            for label in self.model.get_labels()
        ]

    def predict_lang(self, inp):
        if type(inp) != list:
            cleaned_txt = self._clean_up(inp)
            if not cleaned_txt:
                return None
            pred, confidence = self.model.predict([cleaned_txt])
            return self._flatten(pred)[0]
        else:
            batch = [self._clean_up(i) for i in inp]
            pred, confidence = self.model.predict(batch)
            return self._flatten(pred)

    def pred_prob(self, inp):
        if type(inp) != list:
            inp = self._clean_up(inp)
            return self.model.predict_proba([inp])
        return self.model.predict_proba(inp)
