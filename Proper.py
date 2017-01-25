# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 08:16:58 2016

@author: mathieu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import log
from sklearn import preprocessing,cross_validation,metrics,linear_model,ensemble,neighbors
from xgboost.sklearn import XGBRegressor
from Ensemble import Ensemble
from preprocess import Preprocess

class LassoXGB():

    def __init__(self,l):
        self.l = l
        
    def get_params(self,deep=True):
        out = dict()
        out['l'] = self.l
        return out

    def fit(self, X, y):
       self.clf1 = linear_model.Lasso(alpha = 0.0008)
       self.clf1.fit(X,y)
       self.clf2 = XGBRegressor(max_depth=2,learning_rate=0.08,n_estimators= 955,subsample = 0.96)
       self.clf2.fit(X,y)


    def predict(self, X):
        return self.l*(self.clf2.predict(X))+(1-self.l)*(np.array(pd.DataFrame(self.clf1.predict(X))[0]))

train = pd.read_csv('train.csv',delimiter = ',',index_col = 0)
train = train.loc[train['GrLivArea'] < 5500,:]
train = train.loc[train['LotArea'] < 65000,:]

Prep = Preprocess()

X = train.iloc[:,0:len(train.columns)-1]
y = train[['SalePrice']]
y = np.log(y)

Prep.fit(X,y)
X1 = Prep.transform(X,False)
"""
X_cat.drop(['Utilities_NoSeWa','HouseStyle_2.5Fin','PoolQC_Fa','MiscFeature_TenC','Heating_OthW','Exterior2nd_Other',\
            'Condition2_RRAe','Condition2_RRNn','HouseStyle_2.5Fin','Condition2_RRAn','Electrical_Mix','RoofMatl_Membran',\
            'RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Exterior1st_Stone','Heating_Floor','GarageQual_Ex'],\
            axis = 1 , inplace =True)
"""




clf = linear_model.Ridge(alpha =15,fit_intercept = True)

#les coeffs ont été trouvés en en faisant varier 1 avec les 2 autres fixés
clf2 = XGBRegressor(max_depth=2,learning_rate=0.08,n_estimators= 955,subsample = 0.96)


#clf4 = neighbors.KNeighborsRegressor(weights = 'distance')


clf1 = linear_model.Lasso(alpha = 0.0004)
clf1.fit(X1,y)

clf3 = ensemble.RandomForestRegressor(n_estimators=100,max_depth = 15)


clf5 = Ensemble(5,linear_model.Lasso(alpha = 0.0005),[clf,clf1,clf2,clf3])

"""
clf = regularization.logReg(l=0.00001)
clf.fit(X1,y)
"""

def RMSE(y,y_pred):
    return(np.sqrt(sum((y-y_pred)**2)/len(y)))
    

    
RMSE_Score = metrics.make_scorer(RMSE)


def test(model):
    return(np.mean(cross_validation.cross_val_score(model,X1,np.array(y[0]),cv = 10,scoring=RMSE_Score)))

def plot_errorXGB(l):
    res = []
    for alpha in l:
        res.append(test(XGBRegressor(max_depth=2,learning_rate=0.9,n_estimators= alpha)))
    plt.plot(l,res)


def plot_errorKnn(l):
    res = []
    for n in l:
        res.append(test(neighbors.KNeighborsRegressor(n_neighbors = n,weights = 'distance')))
    plt.plot(l,res)
    plt.show()
    
def plot_errorL(l):
    res = []
    for alpha in l:
        res.append(test(linear_model.Ridge(alpha=alpha)))
    plt.plot(l,res)
    plt.show()

def plot_error(l):
    res = []
    for alpha in l:
        res.append(test(Ensemble(5,linear_model.Lasso(alpha = alpha),[clf,clf1,clf2,clf3])))
    plt.plot(l,res)
    plt.show()
    
def result(clf1,clf2):
    test = pd.read_csv('test.csv',delimiter= ',',index_col = 0)
    Xt = Prep.transform(test,True)

    #Z= pd.DataFrame(np.exp(0.6*(clf2.predict(Xt))+0.4*(np.array(pd.DataFrame(clf1.predict(Xt))[0])))-1)
    
    #Z = pd.DataFrame(0.15*np.exp(clf1.predict(Xt))+ 0*np.exp(clf2.predict(Xt)) + 0.85*np.array(pd.DataFrame(np.exp(clf.predict(Xt)))[0]) - 1) 
    
    Z= pd.DataFrame(np.exp(clf1.predict(Xt)))
    Z.index = range(1461,2920)
    Z.columns = ['SalePrice']
    Z.index.rename('id',inplace =True)
    Z.to_csv('RIDGE_17_01.csv',sep=',')
    

