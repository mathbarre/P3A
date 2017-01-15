# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import log
from sklearn import preprocessing,cross_validation,metrics,linear_model,ensemble,neighbors
from xgboost.sklearn import XGBRegressor
from pysmac.optimize import fmin
from sklearn.cross_validation import cross_val_score




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
summary_cat = (X_cat.describe())



X_numeric.fillna(np.nanmean(X_numeric),inplace = True)
scaler = preprocessing.StandardScaler().fit(X_numeric)
X_numeric = pd.DataFrame(scaler.transform(X_numeric))

for a in list(X_cat.columns) :
    X_cat.loc[X_cat[a].isnull(),a] = summary_cat[[a]].loc['top',a]
    
    
X_cat = pd.get_dummies(X_cat)
X_cat.index = range(1460)
X_cat.sort_index(axis=1,inplace = True)
X1 = pd.concat([X_numeric,X_cat],axis=1)

def objective_function(x_int):
    objective_function.n_iterations += 1
    learning_rate,n_estimators,subsample,max_depth =x_int
    n_estimators = int(n_estimators)
    learning_rate = int(learning_rate)
    subsample = int(subsample)
    max_depth = int(max_depth)
    clf = XGBRegressor(n_estimators = n_estimators,learning_rate = learning_rate/100.0,max_depth = max_depth,subsample =subsample/100.0)
    #clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
    scores = cross_val_score(clf, X1, y, cv=5, scoring='neg_mean_squared_error')
    print objective_function.n_iterations, \
        ": n_estimators = ", n_estimators, \
        "\tlearning_rate = ",learning_rate/100.0, \
        "\tsubsample= ",subsample/100.0, \
        "\tmax_depth= ",max_depth, \
        "roc = ", np.sqrt(-np.mean(scores))
        
        #"\tmin_samples_leaf", min_samples_leaf, \
        #"\tmax_leaf_nodes= ",max_leaf_nodes, \
    return np.sqrt(-np.mean(scores))
    


objective_function.n_iterations = 0
xmin, fval = fmin(objective_function, 
                  #x0=(0,0), xmin=(-5, 0), xmax=(10, 15), 
                  x0_int=(1,500,80,2), xmin_int=(1,500,80,2), xmax_int=(15,1500,100,2), 
                  max_evaluations=50)