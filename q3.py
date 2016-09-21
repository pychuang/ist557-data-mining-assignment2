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
    X = preprocessing.normalize(X, axis=0)

    # 5-fold
    for train_index, test_index in cross_validation.KFold(n=len(X), n_folds=5, shuffle=True):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        #print 'TRAIN:', train_index
        #print 'TEST:', test_index
        best_score = 0
        best_scores = None
        best_model = None

        for power in xrange(-4, 5):
            c = math.pow(10, power)
            clf = svm.SVC(C=c).fit(X_train, y_train)
            # 5-fold
            scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
            avg_score = sum(scores)/len(scores)
            if best_score < avg_score:
                best_score = avg_score
                best_scores = scores
                best_c = c
                best_model = clf

        test_scores = best_model.score(X_test, y_test)
        print "best C = %d\nbest scores = %s\ntest scores = %s\n" % (best_c, best_scores, test_scores)


if __name__ == '__main__':
    main()
