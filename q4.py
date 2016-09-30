#! /usr/bin/env python
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import svm


def load_dataset():
    X = []
    y = []
    with open('wine.data') as f:
        for line in f:
            data = line.strip().split(',')
            label = int(data[0])
            sample = [float(x) for x in data[1:]]
            X.append(sample)
            y.append(label)
    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()
    # normalize each feature
    X = preprocessing.scale(X)

    clf = svm.SVC(kernel='rbf')
    # 5-fold
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print scores


if __name__ == '__main__':
    main()
