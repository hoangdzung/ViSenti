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

from preprocessing import * 

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
    