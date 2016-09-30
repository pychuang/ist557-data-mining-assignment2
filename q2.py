#! /usr/bin/env python
import math
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

    for power in xrange(-4, 5):
        c = math.pow(10, power)
        clf = svm.SVC(C=c)
        # 5-fold
        scores = cross_validation.cross_val_score(clf, X, y, cv=5)
        print "C = %f\tscores = %s"  % (c , scores)


if __name__ == '__main__':
    main()
