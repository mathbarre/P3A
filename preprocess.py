#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:03:30 2017

@author: mathieu
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocess():
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.summary_cat = pd.DataFrame()
        pass
    
    def fit(self,X,y):
        self.mean = dict()
        self.var = dict()
        self.Neigh = ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',\
         'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel',\
         'NAmes','NoRidge','NPkVill','NridgHt','NWAmes','OldTown',\
         'SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker']
        for name in self.Neigh :
             self.mean[name] = np.mean(y.loc[X['Neighborhood'] == name,'SalePrice'])
             self.var[name] = np.var(y.loc[X['Neighborhood'] == name,'SalePrice'])
    
        self.Zone = ['A','C','FV','I','RH','RL','RP','RM']
        for name in self.Zone :
            self.mean[name] = np.nanmean(y.loc[X['MSZoning'] == name,'SalePrice'])

        for i in np.arange(1,11) :
            self.mean[i] = np.nanmean(y.loc[X['OverallQual'] == i,'SalePrice'])
     
        self.Func = ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal']
        for name in self.Func:
            self.mean[name] = np.nanmean(y.loc[X['Functional'] == name,'SalePrice'])
            self.var[name] = np.var(y.loc[X['Functional'] == name,'SalePrice'])
        return self
    
    def transform(self,train,X_cat,test = False):
        X = train.copy()
        
        """
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
        """
        X.loc[X['ExterQual'] == 'Ex','ExterQual'] = 5
        X.loc[X['ExterQual'] == 'Gd','ExterQual'] = 4
        X.loc[X['ExterQual'] == 'TA','ExterQual'] = 3
        X.loc[X['ExterQual'] == 'Fa','ExterQual'] = 2
        X.loc[X['ExterQual'] == 'Po','ExterQual'] = 1
        
        X.loc[X['ExterCond'] == 'Ex','ExterCond'] = 5
        X.loc[X['ExterCond'] == 'Gd','ExterCond'] = 4
        X.loc[X['ExterCond'] == 'TA','ExterCond'] = 3
        X.loc[X['ExterCond'] == 'Fa','ExterCond'] = 2
        X.loc[X['ExterCond'] == 'Po','ExterCond'] = 1
            
            
        X['ExterQual'] = pd.Series(X['ExterQual'],dtype = int)
        X['ExterCond'] = pd.Series(X['ExterCond'],dtype = int)
        
        X_numeric = X.select_dtypes(include= ['int','float'])
        
        X_numeric['Yr'] = (X_numeric['YearBuilt']*X_numeric['YearRemodAdd'])
        X_numeric['Overall'] = (X_numeric['OverallQual']*(X_numeric['OverallCond']))
        X_numeric['Tt_Area'] = (X_numeric['TotalBsmtSF']/(X_numeric['GrLivArea']))
        
        X_numeric[['GrLivArea']] = (np.log(X_numeric[['GrLivArea']] + 1))
        X_numeric[['LotArea']] = (np.log(X_numeric[['LotArea']] + 1))
        X_numeric[['BsmtFinSF1']] = (np.log(X_numeric[['BsmtFinSF1']] + 1))
        X_numeric[['TotalBsmtSF']] = (np.log(X_numeric[['TotalBsmtSF']] + 1))
        X_numeric[['1stFlrSF']] = (np.log(X_numeric[['1stFlrSF']] + 1))
        X_numeric[['2ndFlrSF']] = (np.log(X_numeric[['2ndFlrSF']] + 1))
        X_numeric[['GarageArea']] = (np.log(X_numeric[['GarageArea']] + 1))
        X_numeric[['OpenPorchSF']] = (np.log(X_numeric[['OpenPorchSF']] + 1))
        X_numeric[['LowQualFinSF']] = (np.log(X_numeric[['LowQualFinSF']] + 1))
        X_numeric.fillna(X_numeric.mean(),inplace = True)
        
        if(not test):
            self.scaler.fit(X_numeric)
            
        X_numeric = pd.DataFrame(self.scaler.transform(X_numeric))
       
        if test :
            #and also add
            #X_cat.drop('MSSubClass_150',axis = 1,inplace = True)
            X_cat.index = range(1461,2920)
            X_numeric.index = range(1461,2920)
        else:
            X_cat.index = range(X_cat.shape[0])
            X_numeric.index = range(X_numeric.shape[0])
            
        X_cat.sort_index(axis=1,inplace = True)
        X1 = pd.concat([X_numeric,X_cat],axis=1)
        return X1
        
        
        
        
        
        