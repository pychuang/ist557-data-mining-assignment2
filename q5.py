#! /usr/bin/env python
import math
import numpy as np
from sklearn import cross_validation
from sklearn import grid_search
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

    param_grid = [
        {'C': [math.pow(10, power) for power in xrange(-4, 5)], 'gamma': [math.pow(2, power) for power in xrange(-4, 5)]}
    ]
    print 'param_grid:', param_grid

    # 5-fold
    for train_index, test_index in cross_validation.KFold(n=len(X), n_folds=5, shuffle=True):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        #print 'TRAIN:', train_index
        #print 'TEST:', test_index

        gs = grid_search.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, cv=5, verbose=1)
        gs.fit(X_train, y_train)
        best_params = gs.best_params_
        best_score = gs.best_score_

        test_score = gs.best_estimator_.score(X_test, y_test)
        print "best C = %s\nbest score = %f\ntest score = %f\n" % (best_params, best_score, test_score)


if __name__ == '__main__':
    main()
