from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from collections import Counter

import numpy as np
import unidecode
import pandas as pd
import string
import re
import os
import shutil
import pickle 

negative_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{'}

positive_emoticons = {'=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)'}


class Lowercase(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [s.lower() for s in x]

    def fit(self, x, y=None):
        return self


class RemoveTone(BaseEstimator, TransformerMixin):
    def remove_tone(self, s):
        return unidecode.unidecode(s)

    def transform(self, x):
        return [self.remove_tone(s) for s in x]

    def fit(self, x, y=None):
        return self


class CountEmoticons(BaseEstimator, TransformerMixin):
    def count_emoticon(self, s):
        positive_count = 0
        negative_count = 0
        for emoticon in positive_emoticons:
            positive_count += s.count(emoticon)
        for emoticon in negative_emoticons:
            negative_count += s.count(emoticon)
        return positive_count, negative_count

    def transform(self, x):
        return [self.count_emoticon(s) for s in x]

    def fit(self, x, y=None):
        return self


class RemoveDuplicate(BaseEstimator, TransformerMixin):
    def transform(self, x):
        result = []
        for s in x:
            s = re.sub(r'([a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
            s = re.sub(r'([a-z][a-z])\1+', lambda m: m.group(1), s, flags=re.IGNORECASE)
            result.append(s)
        return result

    def fit(self,x, y=None):
        return self


class Tokenrize(BaseEstimator, TransformerMixin):
    def pun_num(self, s):
        for token in s.split():
            if token in string.punctuation:
                if token == '.':
                    s = s
                else:
                    s = s.replace(token, 'punc')
            else:
                s = s
        return s

    def transform(self, x):
        return [self.pun_num(tokenize(s, format='text')) for s in x]

    def fit(self, x, y=None):
        return self