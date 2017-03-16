#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt



train = pd.read_csv('train.csv',delimiter = ',',index_col = 0)

varNames = np.array(train.dtypes.index)
#v =varNames[1]
#sns.set(style="darkgrid", color_codes=True)
#sns.stripplot(x="LotConfig", y="SalePrice", data=train.loc[:,['LotConfig','SalePrice']],jitter = True)
#plt.xticks(rotation=90);


#train.hist(figsize=(25, 30), bins=25)

print(train.shape)
#print(train.dtypes.index)

#Differenciation between missing values and absence of the specified feature

train.loc[train['Alley'].isnull(),'Alley'] = 'No'
train.loc[train['BsmtQual'].isnull(),'BsmtQual'] = 'No'
train.loc[train['BsmtCond'].isnull(),'BsmtCond'] = 'No'
train.loc[train['BsmtFinType1'].isnull(),'BsmtFinType1'] = 'No'
train.loc[train['BsmtExposure'].isnull(),'BsmtExposure'] = 'Unk'         
train.loc[train['BsmtFinType2'].isnull(),'BsmtFinType2'] = 'No'
train.loc[train['FireplaceQu'].isnull(),'FireplaceQu'] = 'No'
train.loc[train['GarageType'].isnull(),'GarageType'] = 'No'
train.loc[train['GarageYrBlt'].isnull(),'GarageYrBlt'] = 0
train.loc[train['GarageFinish'].isnull(),'GarageFinish'] = 'No'
train.loc[train['GarageQual'].isnull(),'GarageQual'] = 'No'
train.loc[train['GarageCond'].isnull(),'GarageCond'] = 'No'
train.loc[train['PoolQC'].isnull(),'PoolQC'] = 'No'
train.loc[train['Fence'].isnull(),'Fence'] = 'Unk'
train.loc[train['MiscFeature'].isnull(),'MiscFeature'] = 'Unk'
train.loc[train['MasVnrType'].isnull(),'MasVnrType'] = 'Unk'
          

#Dropping undifferenciating features
          

train.drop('Street',axis=1, inplace=True)
train.drop('Alley',axis=1, inplace=True)
train.drop('LotShape',axis=1, inplace=True)


#isNull = np.array(train.isnull())
#MV = np.zeros(train.shape[1])
#
#MV=[np.sum(isNull[:,i]) for i in range(train.shape[1])]
#
#varNames = np.array(train.dtypes.index)
#
#varWithMV = varNames[np.array(MV)>0]
#
#print(len(varWithMV))


#Adding Neighborhood standing

#SubC = np.unique(train['MSSubClass'])
#m = [np.mean(train.loc[train['MSSubClass']==n,'SalePrice']) for n in np.unique(train['MSSubClass'])]
#argS = np.argsort(m)
#SortedSubClass = {SubC[i] : np.argwhere(argS==i)[0,0]*20 for i in range(len(SubC))}
#
#train['SortedSubClass'] = [SortedSubClass[w] for w in train['MSSubClass']]
train.drop(['MSSubClass'], axis =1, inplace=True)

Neigh = np.unique(train['Neighborhood'])
m = [np.mean(train.loc[train['Neighborhood']==n,'SalePrice']) for n in Neigh]
NeighDict={}

percentiles = [np.percentile(m,25),np.percentile(m,50),np.percentile(m,75),np.percentile(m,100)]

for i in range(len(m)):
    if (m[i]<percentiles[0]):
        NeighDict[Neigh[i]] = 1
    elif (m[i]<percentiles[1]):
        NeighDict[Neigh[i]] = 2
    elif (m[i]<percentiles[2]):
        NeighDict[Neigh[i]] = 3
    elif (m[i]<=percentiles[3]):
        NeighDict[Neigh[i]] = 4
        
#print(NeighDict)



#adding the avg price per covered square feet in the NeighborHood

train['NeighStanding'] = [NeighDict[w] for w in train['Neighborhood']]

mSqFtPrice = [np.mean(train.loc[train['Neighborhood']==n,'SalePrice']/train.loc[train['Neighborhood']==n,'GrLivArea']) for n in Neigh]


NeighSqFtDict={}

percentiles = [np.percentile(mSqFtPrice,25),np.percentile(mSqFtPrice,50),np.percentile(mSqFtPrice,75),np.percentile(mSqFtPrice,100)]

for i in range(len(m)):
    if (mSqFtPrice[i]<percentiles[0]):
        NeighSqFtDict[Neigh[i]] = 1
    elif (mSqFtPrice[i]<percentiles[1]):
        NeighSqFtDict[Neigh[i]] = 2
    elif (mSqFtPrice[i]<percentiles[2]):
        NeighSqFtDict[Neigh[i]] = 3
    elif (mSqFtPrice[i]<=percentiles[3]):
        NeighSqFtDict[Neigh[i]] = 4
         
                     

train['SqFtPrice'] = [NeighSqFtDict[w] for w in train['Neighborhood']]


#taking the log of skewed variables

X_numeric = train.select_dtypes(include=['int','float'])

X_numeric[['GrLivArea']] = (np.log(X_numeric[['GrLivArea']] + 1))
X_numeric[['LotArea']] = (np.log(X_numeric[['LotArea']] + 1))
X_numeric[['BsmtFinSF1']] = (np.log(X_numeric[['BsmtFinSF1']] + 1))
X_numeric[['TotalBsmtSF']] = (np.log(X_numeric[['TotalBsmtSF']] + 1))
X_numeric[['1stFlrSF']] = (np.log(X_numeric[['1stFlrSF']] + 1))
X_numeric[['2ndFlrSF']] = (np.log(X_numeric[['2ndFlrSF']] + 1))
X_numeric[['GarageArea']] = (np.log(X_numeric[['GarageArea']] + 1))
X_numeric[['OpenPorchSF']] = (np.log(X_numeric[['OpenPorchSF']] + 1))
X_numeric[['LowQualFinSF']] = (np.log(X_numeric[['LowQualFinSF']] + 1))

X_numeric.fillna(X_numeric.mean(), inplace=True)

numVar = X_numeric.dtypes.index

train.loc[:,numVar] = X_numeric



         
#adding some discriminative features

train['OverallProd']=train['OverallQual']*train['OverallCond'] 

QuallDict={'Ex':10, 'Gd': 7, 'TA': 5,'Av':5, 'Fa': 3, 'Mn': 3, 'Po': 1, 'No':0, 'Unk': 0}

train['ExterQual']=[QuallDict[w] for w in train['ExterQual']]
#print(train['ExterQual'].describe)
train['ExterCond']=[QuallDict[w] for w in train['ExterCond']]
train['ExterOverall']= train['ExterQual']*train['ExterQual']

train['BsmtQual']=[QuallDict[w] for w in train['BsmtQual']]
train['BsmtCond']=[QuallDict[w] for w in train['BsmtCond']]
train['BsmtOverall']= train['BsmtQual']*train['BsmtQual']

train['BsmtExposure']=[QuallDict[w] for w in train['BsmtExposure']]
#
train['HeatingQC']=[QuallDict[w] for w in train['HeatingQC']]
#
train['KitchenQual']=[QuallDict[w] for w in train['KitchenQual']]
#
train['FireplaceQu']=[QuallDict[w] for w in train['FireplaceQu']]
#
train['GarageQual']=[QuallDict[w] for w in train['GarageQual']]
train['GarageCond']=[QuallDict[w] for w in train['GarageCond']]
train['GarageOverall'] = train['GarageQual']*train['GarageCond']
train['PoolQC']=[QuallDict[w] for w in train['PoolQC']]

QuallDict={'Typ':10,'Min1':8, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':2, 'Sev':1, 'Sal':0}

train['YrProduct'] = train['YearBuilt']*train['YearRemodAdd']
train['TotalArea'] = train['TotalBsmtSF']+train['GrLivArea']

Year_means_per_neighborhood = {}

for n in Neigh:
    YrSld = np.unique(train.loc[train['Neighborhood']==n, 'YrSold'])
    YrSalePrice = train.loc[train['Neighborhood']==n, ['YrSold','SalePrice']]
    for y in YrSld:
        #print(y)
        Year_means_per_neighborhood["_".join([n,str(y)])]= np.mean(YrSalePrice.loc[YrSalePrice['YrSold']==y,'SalePrice'])

X= np.array(train)
Aux = X[:,[7,72]]
PrPerNeighYr = []

for n,y in Aux:
    PrPerNeighYr.append(Year_means_per_neighborhood["_".join([n,str(y)])])
        
train['MeanPriceNeighYr'] = PrPerNeighYr
     
     
Year_min_per_neighborhood = {}

for n in Neigh:
    YrSld = np.unique(train.loc[train['Neighborhood']==n, 'YrSold'])
    YrSalePrice = train.loc[train['Neighborhood']==n, ['YrSold','SalePrice']]
    for y in YrSld:
        #print(y)
        Year_min_per_neighborhood["_".join([n,str(y)])]= np.min(YrSalePrice.loc[YrSalePrice['YrSold']==y,'SalePrice'])

X= np.array(train)
Aux = X[:,[7,72]]
PrPerNeighYr = []

for n,y in Aux:
    PrPerNeighYr.append(Year_min_per_neighborhood["_".join([n,str(y)])])
        
train['MinPriceNeighYr'] = PrPerNeighYr
     
     
Year_max_per_neighborhood = {}

for n in Neigh:
    YrSld = np.unique(train.loc[train['Neighborhood']==n, 'YrSold'])
    YrSalePrice = train.loc[train['Neighborhood']==n, ['YrSold','SalePrice']]
    for y in YrSld:
        #print(y)
        Year_max_per_neighborhood["_".join([n,str(y)])]= np.max(YrSalePrice.loc[YrSalePrice['YrSold']==y,'SalePrice'])

X= np.array(train)
Aux = X[:,[7,72]]
PrPerNeighYr = []

for n,y in Aux:
    PrPerNeighYr.append(Year_max_per_neighborhood["_".join([n,str(y)])])
        
train['MaxPriceNeighYr'] = PrPerNeighYr

     
Year_mean_per_neighborhood_sqft = {}
Year_max_per_neighborhood_sqft = {}
Year_min_per_neighborhood_sqft = {}


for n in Neigh:
    YrSld = np.unique(train.loc[train['Neighborhood']==n, 'YrSold'])
    YrSalePrice = train.loc[train['Neighborhood']==n, ['GrLivArea','OverallProd','YrSold','SalePrice']]
    for y in YrSld:
        #print(y)
        Year_mean_per_neighborhood_sqft["_".join([n,str(y)])]= np.mean(YrSalePrice.loc[YrSalePrice['YrSold']==y,'SalePrice']/(YrSalePrice.loc[YrSalePrice['YrSold']==y,'GrLivArea']))
        Year_max_per_neighborhood_sqft["_".join([n,str(y)])]= np.max(YrSalePrice.loc[YrSalePrice['YrSold']==y,'SalePrice']/(YrSalePrice.loc[YrSalePrice['YrSold']==y,'GrLivArea']))
        Year_min_per_neighborhood_sqft["_".join([n,str(y)])]= np.min(YrSalePrice.loc[YrSalePrice['YrSold']==y,'SalePrice']/(YrSalePrice.loc[YrSalePrice['YrSold']==y,'GrLivArea']))
                

X= np.array(train)
Aux = X[:,[7,72]]
PrPerNeighYr = []
MinPrPerNeighYr = []
MaxPrPerNeighYr = []


for n,y in Aux:
    PrPerNeighYr.append(Year_mean_per_neighborhood_sqft["_".join([n,str(y)])])
    MinPrPerNeighYr.append(Year_min_per_neighborhood_sqft["_".join([n,str(y)])])
    MaxPrPerNeighYr.append(Year_max_per_neighborhood_sqft["_".join([n,str(y)])])
    
        
train['MeanPriceNeighSqftYr'] = PrPerNeighYr
train['MinPriceNeighSqftYr'] = MinPrPerNeighYr
train['MaxPriceNeighSqftYr'] = MaxPrPerNeighYr


#train['Functional']= [QuallDict[w] for w in train['Functional']]

QuallDict={'GLQ':10, 'ALQ': 8, 'BLQ': 6,'Rec':4, 'LwQ': 3, 'Unf': 2, 'No':0}

train['BsmtFinType1']= [QuallDict[w] for w in train['BsmtFinType1']]





train_aux = train
catNames = train.select_dtypes(include=['object']).dtypes.index


for c in catNames:
    train[c] = train[c].astype('category').cat.codes
#train = pd.get_dummies(train_aux)

train.drop('MiscVal',axis = 1,inplace = True)
train.drop('MiscFeature',axis = 1,inplace = True)  
train.drop('Fence',axis = 1,inplace = True)  
train.drop('GarageQual',axis = 1,inplace = True)
train.drop('EnclosedPorch',axis = 1,inplace = True)
train.drop('BsmtFinType2',axis = 1,inplace = True)
train.drop('LandSlope',axis = 1,inplace = True)
train.drop('LandContour',axis = 1,inplace = True)




X_train = train.drop(['SalePrice'], axis=1)[:1100]
y_train = np.log(train['SalePrice'][:1100])
X_test = train.drop(['SalePrice'], axis=1)[1100:]
y_test = np.array(np.log(train['SalePrice'][1100:]))        


                      
clf = RandomForestRegressor(n_estimators=200,max_depth = 15)
clf2 = Lasso(alpha=0.0005)
clf3 = Ridge(alpha=0.001, fit_intercept=True)
clf4 = LinearRegression(normalize=True)

clf5 = Lasso(alpha=0.0005)



clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)




print(clf4.coef_)



y_pred_RF = clf.predict(X_test)
y_pred_Lasso = clf2.predict(X_test)
y_pred_Ridge = clf3.predict(X_test)
y_pred_Linear = clf4.predict(X_test)

y_pred = np.zeros(len(y_pred_RF))

percentiles = [np.percentile(y_test,25),np.percentile(y_test,50),np.percentile(y_test,75),np.percentile(y_test,100)]

for i in range(len(y_pred)):
    if (y_pred_Lasso[i]<percentiles[0]):
        y_pred[i] = y_pred_Lasso[i]
    elif (y_pred_Lasso[i]<percentiles[1]):
        y_pred[i] = (0.5*y_pred_Lasso[i]+0.5*y_pred_RF[i])
    else :
        y_pred[i] = (0.5*y_pred_Lasso[i]+0.5*y_pred_RF[i])

prox = np.zeros(4)
percentiles = [np.percentile(y_test,25),np.percentile(y_test,50),np.percentile(y_test,75),np.percentile(y_test,100)]

for i in range(len(y_test)):
    if(y_test[i]<percentiles[0]):
        if(np.abs(y_pred_Lasso[i]-y_test[i])<=np.abs(y_pred_Ridge[i]-y_test[i])):
            prox[0] =prox[0]+1
    elif (y_test[i]<percentiles[1]):
        if(np.abs(y_pred_Lasso[i]-y_test[i])<=np.abs(y_pred_Ridge[i]-y_test[i])):
            prox[1] =prox[1]+1
    elif (y_test[i]<percentiles[2]):
        if(np.abs(y_pred_Lasso[i]-y_test[i])<=np.abs(y_pred_Ridge[i]-y_test[i])):
            prox[2] =prox[2]+1
    elif (y_test[i]>=percentiles[2]):
        if(np.abs(y_pred_Lasso[i]-y_test[i])<=np.abs(y_pred_Ridge[i]-y_test[i])):
            prox[3] =prox[3]+1
print(prox)

def RMSE(y,y_pred):
    return(np.sqrt(sum((y-y_pred)**2)/len(y)))

RMSE_Score = make_scorer(RMSE)

#print(zip(y_pred_Ridge, zip(y_pred, y_test)))

print(RMSE((y_test), (y_pred)))

print(RMSE((y_pred_RF), (y_test)))
print(RMSE((y_pred_Lasso), (y_test)))
print(RMSE(y_pred_Ridge, y_test))
print(RMSE(y_pred_Linear, y_test))
#X_xr = np.array(train)
#print(X_xr[np.where((np.abs(y_pred_Lasso-y_test)<=np.abs(y_pred_Ridge-y_test)))])

