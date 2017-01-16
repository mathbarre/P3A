#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:46:30 2017

@author: mathieu
"""
import numpy as np
from sklearn.cross_validation import KFold

class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        
    def get_params(self,deep =True):
        out = dict()
        out['n_folds']= self.n_folds
        out['stacker'] = self.stacker
        out['base_models'] = self.base_models
        return out
        
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        self.folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(self.folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                self.stacker.fit(S_train,y)
                
    def predict(self,T):
        T = np.array(T)
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(self.folds)))
            for j, (train_idx, test_idx) in enumerate(self.folds):
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
        
        
 