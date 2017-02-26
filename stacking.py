#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:41:55 2017

@author: mathieubarre
"""


# Stacking Starter based on Allstate Faron's Script
#https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
# Preprocessing from Alexandru Papiu
#https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
from Ensemble import Ensemble


class Discretize():
    

    
    
    def __init__(self,quantil):
        self.quantil = quantil

        

    def fit(self,y):
        q=self.quantil

        s=np.arange(100/q+1)
        s=s*q
        self.table=np.percentile(y,s)
        return(self.table)
        
        
            
    def transform(self,y):
        table=self.table
        for i in range(len(y)):
            for j in range(len(table)):
                if (y.iloc[i]<table[j]):
                    y.iloc[i]=(table[j]+table[j-1])/2
                    break
        return(y)

TARGET = 'SalePrice'
NFOLDS = 4
SEED = 0
NROWS = None
SUBMISSION_FILE = 'sample_submission.csv'


## Load the data ##
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ntrain = train.shape[0]
ntest = test.shape[0]

## Preprocessing ##

y_train = np.log(train[TARGET])
train.drop([TARGET], axis=1, inplace=True)
"""


train["IsRegularLotShape"] = (train["LotShape"] == "Reg") * 1

    # Most properties are level; bin the other possibilities together
    # as "not level".
train["IsLandLevel"] = (train["LandContour"] == "Lvl") * 1

    # Most land slopes are gentle; treat the others as "not gentle".
train["IsLandSlopeGentle"] = (train["LandSlope"] == "Gtl") * 1

    # Most properties use standard circuit breakers.
train["IsElectricalSBrkr"] = (train["Electrical"] == "SBrkr") * 1

    # About 2/3rd have an attached garage.
train["IsGarageDetached"] = (train["GarageType"] == "Detchd") * 1

    # Most have a paved drive. Treat dirt/gravel and partial pavement
    # as "not paved".
train["IsPavedDrive"] = (train["PavedDrive"] == "Y") * 1

    # The only interesting "misc. feature" is the presence of a shed.
train["HasShed"] = (train["MiscFeature"] == "Shed") * 1.  

train['Yr'] = (train['YearBuilt']*train['YearRemodAdd'])
train['Overall'] = (train['OverallQual']*(train['OverallCond']))
train['Tt_Area'] = (train['TotalBsmtSF']/(train['GrLivArea']))
train["RecentRemodel"] = (train["YearRemodAdd"] == train["YrSold"]) * 1
train["VeryNewHouse"] = (train["YearBuilt"] == train["YrSold"]) * 1
train["Remodeled"] = (train["YearRemodAdd"] != train["YearBuilt"]) * 1


train["IsRegularLotShape"] = (train["LotShape"] == "Reg") * 1

    # Most properties are level; bin the other possibilities together
    # as "not level".
train["IsLandLevel"] = (test["LandContour"] == "Lvl") * 1

    # Most land slopes are gentle; treat the others as "not gentle".
test["IsLandSlopeGentle"] = (test["LandSlope"] == "Gtl") * 1

    # Most properties use standard circuit breakers.
test["IsElectricalSBrkr"] = (test["Electrical"] == "SBrkr") * 1

    # About 2/3rd have an attached garage.
test["IsGarageDetached"] = (test["GarageType"] == "Detchd") * 1

    # Most have a paved drive. Treat dirt/gravel and partial pavement
    # as "not paved".
test["IsPavedDrive"] = (test["PavedDrive"] == "Y") * 1

    # The only interesting "misc. feature" is the presence of a shed.
test["HasShed"] = (test["MiscFeature"] == "Shed") * 1.  

test['Yr'] = (test['YearBuilt']*test['YearRemodAdd'])
test['Overall'] = (test['OverallQual']*(test['OverallCond']))
test['Tt_Area'] = (test['TotalBsmtSF']/(test['GrLivArea']))
test["RecentRemodel"] = (test["YearRemodAdd"] == test["YrSold"]) * 1
test["VeryNewHouse"] = (test["YearBuilt"] == test["YrSold"]) * 1
test["Remodeled"] = (test["YearRemodAdd"] != test["YearBuilt"]) * 1
"""



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

dis = Discretize(15)
dis.fit(all_data['GrLivArea'].copy())
all_data['discretize'] = pd.to_numeric(dis.transform(all_data['GrLivArea'].copy()))
dis = Discretize(5)
dis.fit(all_data['LotArea'].copy())
all_data['discretize1'] = pd.to_numeric(dis.transform(all_data['LotArea'].copy()))
dis = Discretize(3)
dis.fit(all_data['YearBuilt'].copy())
all_data['discretize2'] = pd.to_numeric(dis.transform(all_data['YearBuilt'].copy()))

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.8]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())



#creating matrices for sklearn:

x_train = np.array(all_data[:train.shape[0]])
x_test = np.array(all_data[train.shape[0]:])





kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

ab_params = {
'base_estimator':DecisionTreeRegressor(max_depth = 14,max_features = 0.2),
'n_estimators': 200,
'learning_rate':1.0,

}
    
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 13,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.2,
    'max_depth': 14,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.6,
    'silent': 1,
    'subsample': 0.5,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 3,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.0005
}

ab = SklearnWrapper(clf = AdaBoostRegressor,seed=SEED,params=ab_params)
xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

ab_oof_train, ab_oof_test = get_oof(ab)
xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("AB-CV: {}".format(sqrt(mean_squared_error(y_train, ab_oof_train))))
print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))


x_train = np.concatenate((ab_oof_train,xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((ab_oof_test,xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)


print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=2000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = gbdt.predict(dtest)
saleprice = np.exp(submission['SalePrice'])
submission['SalePrice'] = saleprice
submission.to_csv('stacker.csv', index=None)


