#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:03:30 2017

@author: mathieu
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

class Preprocess():
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.summary_cat = pd.DataFrame()
        pass
    
    def fit_cat(self,X,y):
        
        self.mean = dict()
        self.var = dict()
        self.Neigh = ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',\
         'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel',\
         'NAmes','NoRidge','NPkVill','NridgHt','NWAmes','OldTown',\
         'SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker']
        for name in self.Neigh :
            self.mean[name] = np.nanmean(y.loc[X['Neighborhood'] == name,'SalePrice'])
            self.var[name] = np.nanvar(y.loc[X['Neighborhood'] == name,'SalePrice'])
    
        self.Zone = ['FV','RH','RL','RM']
        for name in self.Zone :
            self.mean[name] = np.nanmean(y.loc[X['MSZoning'] == name,'SalePrice'])

     
        self.Func = ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev']
        for name in self.Func:
            self.mean[name] = np.mean(y.loc[X['Functional'] == name,'SalePrice'])
            self.var[name] = np.var(y.loc[X['Functional'] == name,'SalePrice'])
        
        return self
        
    def fit_num(self,X,y):
        

        for i in np.arange(1,11) :
            self.mean[i] = np.nanmean(y.loc[X['OverallQual'] == i,'SalePrice'])
        
        return self
    
    
    def transform_num(self,train,test = False):
        X = train.copy()
        
        X['OverallQuallMean'] = 0
        for i in np.arange(1,11) :
            X.loc[X['OverallQual'] == i,'OverallQuallMean'] = self.mean[i]
        """
        X['Neighmean'] = 0
        X['Neighvar'] = 0
        X['Zonemean'] = 0
        X['OverallQuallMean'] = 0
        X['Funcmean'] = 0
        X['Funcvar'] = 0 
        
        for name in self.Neigh :
             X.loc[X['Neighborhood'] == name,'Neighmean'] = self.mean[name] 
             X.loc[X['Neighborhood'] == name,'Neighvar'] = self.var[name] 
        
        self.Zone = ['FV','RH','RL','RM']
        for name in self.Zone :
            X.loc[X['MSZoning'] == name,'Zonemean'] = self.mean[name] 
        for i in np.arange(1,11) :
            X.loc[X['OverallQual'] == i,'OverallQuallMean'] = self.mean[i]
     
        self.Func = ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev']
        for name in self.Func:
            X.loc[X['Functional'] == name,'Funcmean'] = self.mean[name] 
            X.loc[X['Functional'] == name,'Funcvar'] = self.var[name] 
        
        
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
        """
        X.fillna(X.mean(),inplace = True)
        if not test :
            self.skewed_feats = X.apply(lambda x: skew(x.dropna())) #compute skewness
            self.skewed_feats = self.skewed_feats[self.skewed_feats > 0.75]
            self.skewed_feats = self.skewed_feats.index

        X[self.skewed_feats] = np.log1p(X[self.skewed_feats])
        if(not test):
            self.scaler.fit(X)
            
        X = pd.DataFrame(self.scaler.transform(X))
           
     
        X.index = X.index+1
        """    
        X_cat.sort_index(axis=1,inplace = True)
        X1 = pd.concat([X_numeric,X_cat],axis=1)
        """
        return X
        
    def transform_cat(self,train):
        X = train.copy()
        
        X['Neighmean'] = 0
        X['Neighvar'] = 0
        X['Zonemean'] = 0
        X['Funcmean'] = 0
        X['Funcvar'] = 0 
        for name in self.Neigh :
             X.loc[X['Neighborhood'] == name,'Neighmean'] = self.mean[name] 
             X.loc[X['Neighborhood'] == name,'Neighvar'] = self.var[name] 
        

        for name in self.Zone :
            X.loc[X['MSZoning'] == name,'Zonemean'] = self.mean[name] 
        

        for name in self.Func:
            X.loc[X['Functional'] == name,'Funcmean'] = self.mean[name] 
            X.loc[X['Functional'] == name,'Funcvar'] = self.var[name] 
        
        X.fillna(X.mean(),inplace = True)
        return X
        
