#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 18:36:42 2017

@author: mathieubarre
"""
from sklearn.linear_model import Lasso
import numpy as np


class AdaptativeLasso():
    

    
    
    def __init__(self,alpha,n_iter):
        self.alpha = alpha
        self.n_iter = n_iter
        
    def get_params(self,deep =True):
        out = dict()
        out['alpha']= self.alpha
        out['n_iter']= self.n_iter
        return out
        
    def fit(self,X,y):
        gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)
        n_samples, n_features = X.shape

        self.weights = np.ones(n_features)
        n_lasso_iterations = self.n_iter

        for k in range(n_lasso_iterations):
            X_w = X / self.weights
            clf = Lasso(alpha=self.alpha)
            clf.fit(X_w, y)
            coef_ = clf.coef_ / self.weights
            self.weights = gprime(coef_)
        self.clf = Lasso(self.alpha);
        self.clf.fit(X / self.weights,y)
        
            
    def predict(self,X):
        return self.clf.predict(X / self.weights)