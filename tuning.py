#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:59:46 2017

@author: mathieubarre
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
from sklearn.preprocessing import StandardScaler

def RMSE(y,y_pred):
    return(np.sqrt(sum((y-y_pred)**2)/len(y)))

RMSE_Score = make_scorer(RMSE)

TARGET = 'SalePrice'


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

ntrain = train.shape[0]
ntest = test.shape[0]

## Preprocessing ##

y_train = np.log(train[TARGET])


train.drop([TARGET], axis=1, inplace=True)


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())



#creating matrices for sklearn:

x_train = np.array(all_data[:train.shape[0]])

x_test = np.array(all_data[train.shape[0]:])

#print(np.mean(cross_val_score(AdaBoostRegressor(n_estimators= 200,learning_rate=1.0),x_train,y_train,cv = 5,scoring = RMSE_Score)))





from adaptativeLasso import AdaptativeLasso

res_AL = []
res_AL3 = []
res_L = []
alph = np.arange(0.00001,0.001,0.00001)
for l in alph:
    aL = AdaptativeLasso(alpha = l,n_iter=1)
    aL3 = AdaptativeLasso(alpha = l,n_iter=3)
    res_AL3.append(np.mean(cross_val_score(aL3,x_train,y_train,cv = 5,scoring = RMSE_Score)))
    res_AL.append(np.mean(cross_val_score(aL,x_train,y_train,cv = 5,scoring = RMSE_Score)))
    res_L.append(np.mean(cross_val_score(Lasso(l),x_train,y_train,cv = 5,scoring = RMSE_Score)))
    
plt.plot(alph,res_AL)
plt.plot(alph,res_AL3)
plt.plot(alph,res_L)









