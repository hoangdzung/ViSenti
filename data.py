import pandas as pd
import numpy as np 

def getdata():
    df = pd.read_csv('combine.csv')
    X = df['sentence'].values
    Y = df['label'].values
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    X_train = X[:-10000]
    Y_train = Y[:-10000]

    bad_kw = ['địt', 'cặc', 'đéo', 'dái', 'buồi', 'lìn', 'chán', 'buồn', 'nản', 'mệt', 'tệ', 'kém', 'yếu', 'dốt', 'lười', 'ốm', 'ghét', 'hận', 'thù','xấu', 'đụ']
    good_kw = ['xinh', 'đẹp', 'khỏe', 'mạnh', 'vui', 'tươi', 'ngon', 'tốt', 'giỏi', 'siêu', 'khủng', 'thông minh', 'xuất sắc', 'bền']
    bad_kw = bad_kw *20
    bad_labels = [0] * len(bad_kw)
    good_kw = good_kw *20
    good_labels = [1] * len(good_kw)
    X_train = np.concatenate([X_train, np.array(good_kw), np.array(bad_kw)])
    Y_train = np.concatenate([Y_train, np.array(good_labels), np.array(bad_labels)])
    X_test = X[-10000:]
    Y_test = Y[-10000:]
    return (X_train, Y_train), (X_test, Y_test)