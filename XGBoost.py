from underthesea.word_tokenize.regex_tokenize import tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
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

estimator_C = 0.72
lower_tfidf__ngram_range = (1, 4)
with_tone_char__ngram_range = (1, 5)
remove_tone__tfidf__ngram_range = (1, 3)


pipeline = Pipeline(
    steps=[
        ('features', FeatureUnion([
            ('lower_tfidf', Pipeline([
                ('lower', Lowercase()),
                ('tfidf', TfidfVectorizer(ngram_range=lower_tfidf__ngram_range, norm='l2', min_df=2))])),
            ('with_tone_char',
             TfidfVectorizer(ngram_range=with_tone_char__ngram_range, norm='l2', min_df=2, analyzer='char')),
            ('remove_tone', Pipeline([
                ('remove_tone', RemoveTone()),
                ('lower', Lowercase()),
                ('tfidf', TfidfVectorizer(ngram_range=remove_tone__tfidf__ngram_range, norm='l2', min_df=2))])),
            ('emoticons', CountEmoticons())
        ])),
        ('estimator', XGBClassifier())
        # ('estimator', SVC(kernel='linear', C=estimator_C, class_weight='balanced', verbose=True))
    ]
)

df = pd.read_csv('combine.csv')
X = df['sentence'].values
Y = df['label'].values
index = np.arange(X.shape[0])
np.random.shuffle(index)
X = X[index]
Y = Y[index]
X_train = X[:-10000]
Y_train = Y[:-1000]
X_test = X[-10000:]
Y_test = Y[-10000:]

pipeline.fit(X_train, Y_train)

train_acc = (pipeline.predict(X_train) == Y_train).astype(int).mean()
test_acc = (pipeline.predict(X_test) == Y_test).astype(int).mean()

print("Train acc: ", train_acc, " Test acc: ", test_acc)

with open('model_xgboost.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
    