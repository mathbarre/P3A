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



class LassoXGB():

    def __init__(self,l):
        self.l = l
        
    def get_params(self,deep=True):
        out = dict()
        out['l'] = self.l
        return out

    def fit(self, X, y):
       self.clf1 = linear_model.Lasso(alpha = 0.0005)
       self.clf1.fit(X,y)
       self.clf2 = XGBRegressor(max_depth=2,learning_rate=0.08,n_estimators= 1518,subsample = 0.94)
       self.clf2.fit(X,y)


    def predict(self, X):
        return self.l*(self.clf2.predict(X))+(1-self.l)*(np.array(pd.DataFrame(self.clf1.predict(X))[0]))

train = pd.read_csv('train.csv',delimiter = ',',index_col = 0)
X = train.iloc[:,0:len(train.columns)-1]
X.drop('MoSold',axis = 1,inplace = True)
X.drop('MiscVal',axis = 1,inplace = True)
X.drop('MiscFeature',axis = 1,inplace = True)  
X.drop('SaleType',axis = 1,inplace = True)  
X.drop('Fence',axis = 1,inplace = True)  
X.drop('PoolQC',axis = 1,inplace = True)  
X.drop('3SsnPorch',axis = 1,inplace = True)  
X.drop('PavedDrive',axis = 1,inplace = True) 
X.drop('GarageFinish',axis = 1,inplace = True)
X.drop('GarageType',axis = 1,inplace = True)
X.drop('FireplaceQu',axis = 1,inplace = True)
X.drop('HalfBath',axis = 1,inplace = True)
X.drop('Electrical',axis = 1,inplace = True)
X.drop('BsmtFinType2',axis = 1,inplace = True)
X.drop('BsmtFinType1',axis = 1,inplace = True)
X.drop('MasVnrType',axis = 1,inplace = True)
X.drop('Exterior2nd',axis = 1,inplace = True)
X.drop('RoofStyle',axis = 1,inplace = True)
X.drop('BldgType',axis = 1,inplace = True)
X.drop('Condition2',axis = 1,inplace = True)
X.drop('LandSlope',axis = 1,inplace = True)
X.drop('LandContour',axis = 1,inplace = True)
X.drop('LotShape',axis = 1,inplace = True)
X.drop('Alley',axis = 1,inplace = True)
X.drop('Street',axis = 1,inplace = True)
X.drop('MSSubClass',axis = 1,inplace = True)
X.drop('LotFrontage',axis = 1,inplace = True)

X.loc[X['ExterQual'] == 'Ex','ExterQual1'] = 5
X.loc[X['ExterQual'] == 'Gd','ExterQual1'] = 4
X.loc[X['ExterQual'] == 'TA','ExterQual1'] = 3
X.loc[X['ExterQual'] == 'Fa','ExterQual1'] = 2
X.loc[X['ExterQual'] == 'Po','ExterQual1'] = 1

X.loc[X['ExterCond'] == 'Ex','ExterCond1'] = 5
X.loc[X['ExterCond'] == 'Gd','ExterCond1'] = 4
X.loc[X['ExterCond'] == 'TA','ExterCond1'] = 3
X.loc[X['ExterCond'] == 'Fa','ExterCond1'] = 2
X.loc[X['ExterCond'] == 'Po','ExterCond1'] = 1

X['ExterQual1'] = pd.Series(X['ExterQual1'],dtype = int)
X['ExterCond1'] = pd.Series(X['ExterCond1'],dtype = int)

Neigh = ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',\
         'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel',\
         'NAmes','NoRidge','NPkVill','NridgHt','NWAmes','OldTown',\
         'SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker']
for name in Neigh :
    X.loc[X['Neighborhood'] == name,'Neigh'] = np.mean(train.loc[X['Neighborhood'] == name,'SalePrice'])
    X.loc[X['Neighborhood'] == name,'Neighsd'] = np.var(train.loc[X['Neighborhood'] == name,'SalePrice'])
    
Zone = ['A','C','FV','I','RH','RL','RP','RM']
for name in Zone :
    X.loc[X['MSZoning'] == name,'Zone'] = np.nanmean(train.loc[X['MSZoning'] == name,'SalePrice'])

for i in np.arange(1,11) :
     X.loc[X['OverallQual'] == i,'OverQ'] = np.nanmean(train.loc[X['OverallQual'] == i,'SalePrice'])
     
Func = ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal']
for name in Func:
    X.loc[X['Functional'] == name,'Func'] = np.nanmean(train.loc[X['Functional'] == name,'SalePrice'])
    X.loc[X['Functional'] == name,'Func1'] = np.var(train.loc[X['Functional'] == name,'SalePrice'])
    
    
    
    



y = train[['SalePrice']]
y = np.array(y,dtype ='float')
y=(log(y+1))
y = pd.DataFrame(y)

X_numeric = X.select_dtypes(include= ['int','float'])



#X_numeric['Bedroom_Area'] = (X_numeric['BedroomAbvGr']/(X_numeric['GrLivArea'])+1)

X_numeric['Yr'] = (X_numeric['YearBuilt']*X_numeric['YearRemodAdd'])
X_numeric['Overall'] = (X_numeric['OverallQual']*(X_numeric['OverallCond']))

X_numeric['Ext'] = (X_numeric['ExterQual1']*(X_numeric['ExterCond1']))

X_numeric.drop(['ExterQual1','ExterCond1'],axis = 1,inplace = True)


X_numeric['Tt_Area'] = (X_numeric['TotalBsmtSF']/(X_numeric['GrLivArea']))


X_numeric[['GrLivArea']] = (log(X_numeric[['GrLivArea']] + 1))
X_numeric[['LotArea']] = (log(X_numeric[['LotArea']] + 1))

X_numeric[['BsmtFinSF1']] = (log(X_numeric[['BsmtFinSF1']] + 1))
X_numeric[['TotalBsmtSF']] = (log(X_numeric[['TotalBsmtSF']] + 1))
X_numeric[['1stFlrSF']] = (log(X_numeric[['1stFlrSF']] + 1))
X_numeric[['2ndFlrSF']] = (log(X_numeric[['2ndFlrSF']] + 1))
X_numeric[['GarageArea']] = (log(X_numeric[['GarageArea']] + 1))
X_numeric[['OpenPorchSF']] = (log(X_numeric[['OpenPorchSF']] + 1))
X_numeric[['LowQualFinSF']] = (log(X_numeric[['LowQualFinSF']] + 1))



X_numeric.fillna(np.nanmean(X_numeric),inplace = True)
scaler = preprocessing.StandardScaler().fit(X_numeric)
X_numeric = pd.DataFrame(scaler.transform(X_numeric))

X_cat =  X.select_dtypes(include= ['object'])
"""
X_cat.loc[X_cat['PoolQC'].isnull(),'PoolQC'] = 'No'
X_cat.loc[X_cat['Fence'].isnull(),'Fence'] = 'No'
X_cat.loc[X_cat['MiscFeature'].isnull(),'MiscFeature'] = 'No'
X_cat.loc[X_cat['GarageCond'].isnull(),'GarageCond'] = 'No'
X_cat.loc[X_cat['GarageQual'].isnull(),'GarageQual'] = 'No'
X_cat.loc[X_cat['GarageFinish'].isnull(),'GarageFinish'] = 'No'
X_cat.loc[X_cat['GarageType'].isnull(),'GarageType'] = 'No'
X_cat.loc[X_cat['FireplaceQu'].isnull(),'FireplaceQu'] = 'No'
X_cat.loc[X_cat['BsmtFinType2'].isnull(),'BsmtFinType2'] = 'No'
X_cat.loc[X_cat['BsmtFinType1'].isnull(),'BsmtFinType1'] = 'No'
X_cat.loc[X_cat['BsmtExposure'].isnull(),'BsmtExposure'] = 'No'
X_cat.loc[X_cat['BsmtCond'].isnull(),'BsmtCond'] = 'No'
X_cat.loc[X_cat['BsmtQual'].isnull(),'BsmtQual'] = 'No'
X_cat.loc[X_cat['Alley'].isnull(),'Alley'] = 'No'
"""
summary_cat = (X_cat.describe())
for a in list(X_cat.columns) :
    X_cat.loc[X_cat[a].isnull(),a] = summary_cat[[a]].loc['top',a]
    
    
X_cat = pd.get_dummies(X_cat)

X_cat.index = range(1460)
X_cat.sort_index(axis=1,inplace = True)
X1 = pd.concat([X_numeric,X_cat],axis=1)



clf = linear_model.Ridge(alpha =10,fit_intercept = True)
clf.fit(X1,y)

#les coeffs ont été trouvés en en faisant varier 1 avec les 2 autres fixés
clf2 = XGBRegressor(max_depth=2,learning_rate=0.11,n_estimators= 617)
clf2.fit(X1,y)

#clf4 = neighbors.KNeighborsRegressor(weights = 'distance')


clf1 = linear_model.Lasso(alpha = 0.00035)
clf1.fit(X1,y)

clf3 = linear_model.ElasticNet(alpha = 0.00045,l1_ratio = 0.8)
clf3.fit(X1,y)

"""
clf = regularization.logReg(l=0.00001)
clf.fit(X1,y)
"""

def RMSE(y,y_pred):
    return(np.sqrt(sum((y-y_pred)**2)/len(y)))
    

    
RMSE_Score = metrics.make_scorer(RMSE)


def test(model):
    return(np.mean(cross_validation.cross_val_score(model,X1,np.array(y[0]),cv = 5,scoring=RMSE_Score)))

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
        res.append(test(linear_model.Lasso(alpha=alpha)))
    plt.plot(l,res)
    plt.show()

def plot_error(l):
    res = []
    for alpha in l:
        res.append(test(LassoXGB(l=alpha)))
    plt.plot(l,res)
    plt.show()
    
def result(clf1,clf2):
    test = pd.read_csv('test.csv',delimiter= ',',index_col = 0)
    test.drop('MoSold',axis = 1,inplace = True)
    test.drop('MiscVal',axis = 1,inplace = True)
    test.drop('MiscFeature',axis = 1,inplace = True)  
    test.drop('SaleType',axis = 1,inplace = True)  
    test.drop('Fence',axis = 1,inplace = True)  
    test.drop('PoolQC',axis = 1,inplace = True)  
    test.drop('3SsnPorch',axis = 1,inplace = True)  
    test.drop('PavedDrive',axis = 1,inplace = True) 
    test.drop('GarageFinish',axis = 1,inplace = True)
    test.drop('GarageType',axis = 1,inplace = True)
    test.drop('FireplaceQu',axis = 1,inplace = True)
    test.drop('HalfBath',axis = 1,inplace = True)
    test.drop('Electrical',axis = 1,inplace = True)
    test.drop('BsmtFinType2',axis = 1,inplace = True)
    test.drop('BsmtFinType1',axis = 1,inplace = True)
    test.drop('MasVnrType',axis = 1,inplace = True)
    test.drop('Exterior2nd',axis = 1,inplace = True)
    test.drop('RoofStyle',axis = 1,inplace = True)
    test.drop('BldgType',axis = 1,inplace = True)
    test.drop('Condition2',axis = 1,inplace = True)
    test.drop('LandSlope',axis = 1,inplace = True)
    test.drop('LandContour',axis = 1,inplace = True)
    test.drop('LotShape',axis = 1,inplace = True)
    test.drop('Alley',axis = 1,inplace = True)
    test.drop('Street',axis = 1,inplace = True)
    test.drop('MSSubClass',axis = 1,inplace = True)
    test.drop('LotFrontage',axis = 1,inplace = True)
    Xt_numeric = test.select_dtypes(include= ['int','float'])
    
    
    Xt_numeric['Yr'] = (Xt_numeric['YearBuilt']*Xt_numeric['YearRemodAdd'])
    Xt_numeric['Tt_Area'] = (Xt_numeric['TotalBsmtSF']/(Xt_numeric['GrLivArea']))
    
    Xt_numeric[['GrLivArea']] = (log(Xt_numeric[['GrLivArea']] + 1))
    Xt_numeric[['LotArea']] = (log(Xt_numeric[['LotArea']] + 1))
    Xt_numeric[['BsmtFinSF1']] = (log(Xt_numeric[['BsmtFinSF1']] + 1))
    Xt_numeric[['TotalBsmtSF']] = (log(Xt_numeric[['TotalBsmtSF']] + 1))
    Xt_numeric[['1stFlrSF']] = (log(Xt_numeric[['1stFlrSF']] + 1))
    Xt_numeric[['2ndFlrSF']] = (log(Xt_numeric[['2ndFlrSF']] + 1))
    Xt_numeric[['GarageArea']] = (log(Xt_numeric[['GarageArea']] + 1))
    Xt_numeric[['OpenPorchSF']] = (log(Xt_numeric[['OpenPorchSF']] + 1))
    Xt_numeric[['LowQualFinSF']] = (log(Xt_numeric[['LowQualFinSF']] + 1))

    
    Xt_numeric.fillna(np.nanmean(X_numeric),inplace = True)
    Xt_numeric = pd.DataFrame(scaler.transform(Xt_numeric))
    
    Xt_cat =  test.select_dtypes(include= ['object'])
    for a in list(Xt_cat.columns) :
        Xt_cat.loc[Xt_cat[a].isnull(),a] = summary_cat[[a]].loc['top',a]
    
    Xt_cat = pd.get_dummies(Xt_cat)
    
    Xt_cat['Utilities_NoSeWa'] = 0
    Xt_cat['HouseStyle_2.5Fin'] = 0
    Xt_cat['RoofMatl_ClyTile'] = 0
    Xt_cat['RoofMatl_Membran'] = 0
    Xt_cat[ 'RoofMatl_Metal'] = 0
    Xt_cat['RoofMatl_Roll'] = 0
    Xt_cat['Exterior1st_ImStucc'] = 0
    Xt_cat['Exterior1st_Stone'] = 0
    Xt_cat['Heating_Floor'] = 0
    Xt_cat['Heating_OthW'] = 0
    Xt_cat['GarageQual_Ex'] = 0

    
    Xt_cat.index = range(1461,2920)
    
    Xt_cat.sort_index(axis=1,inplace = True)
    Xt_numeric.index = range(1461,2920)
    Xt = pd.concat([Xt_numeric,Xt_cat],axis = 1)
    
    Z= pd.DataFrame(np.exp(0.2*(clf2.predict(Xt))+0.8*(np.array(pd.DataFrame(clf1.predict(Xt))[0])))-1)
    #Z = pd.DataFrame(0.15*np.exp(clf1.predict(Xt))+ 0*np.exp(clf2.predict(Xt)) + 0.85*np.array(pd.DataFrame(np.exp(clf.predict(Xt)))[0]) - 1) 
    #Z= pd.DataFrame(np.exp(clf1.predict(Xt))-1)
    Z.index = range(1461,2920)
    Z.columns = ['SalePrice']
    Z.index.rename('id',inplace =True)
    Z.to_csv('resultLassoXGB40_15_01.csv',sep=',')
    


