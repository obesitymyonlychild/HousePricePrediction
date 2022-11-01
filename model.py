#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:11:47 2020

@author: wufei
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from Help_functions import sklearn_Pvalue, RMSE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype



train_data=pd.read_csv("/Users/wufei/Desktop/kaggle/train_2.csv")
test_data=pd.read_csv("/Users/wufei/Desktop/kaggle/test_2.csv")

numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
numerical_features = numerical_features.drop("Id")

for column in numerical_features:
    new_col =  column + "squared"
    train_data[new_col] = train_data[column] ** 2
    test_data[new_col] = test_data[column]**2
    corr = train_data.corr()
    if corr.SalePrice[new_col] > corr.SalePrice[column]:
        train_data.drop([column], axis = 1)
        test_data.drop([column], axis = 1)
    else:
        train_data.drop([new_col], axis = 1)
        test_data.drop([new_col], axis = 1)

numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
numerical_features = numerical_features.drop("Id")

        
for column in numerical_features:
    new_col= column + "log"
    train_data[new_col] = np.log(train_data[column] + 1e-28)
    test_data[new_col] = np.log(test_data[column] + 1e-28)
    corr2 = train_data.corr()
    if corr2.SalePrice[new_col] > corr2.SalePrice[column]:
        train_data.drop([column], axis = 1)
        test_data.drop([column], axis = 1)
    else:
        train_data.drop([new_col], axis = 1)
        test_data.drop([new_col], axis = 1)


#get dummies:
categorical_features = train_data.select_dtypes(include = ["object"]).columns
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")

cat_features = test_data.select_dtypes(include = ["object"]).columns
num_features = test_data.select_dtypes(exclude = ["object"]).columns
train_num = train_data[numerical_features]
test_num = test_data[num_features]
test_num.shape
train_num.shape
train_cat = train_data[categorical_features]
test_cat = test_data[cat_features]
train_num = train_num.fillna(train_num.median())
test_num = test_num.fillna(test_num.median())


#2.6 turn category features to dummy
train_cat = pd.get_dummies(train_cat, drop_first=True)
test_cat = pd.get_dummies(test_cat, drop_first=True)

#2.7 join categorical and numerical features 
train_data_new = pd.concat([train_num, train_cat], axis = 1)
test_data_new = pd.concat([test_num, test_cat], axis = 1)
test_data_new.shape
train_data_new.shape

test_data_new.head(10)
train_data['SalePrice']
        
#remove collinear columns: 
corr_matrix = train_data_new.corr().abs()
corr_matrix2 = test_data_new.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper2 = corr_matrix2.where(np.triu(np.ones(corr_matrix2.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop2 = [column for column in upper2.columns if any(upper2[column] > 0.95)]
train_data_new.drop(to_drop, axis=1, inplace=True)
test_data_new.drop(to_drop2, axis=1, inplace=True)
train_data['SalePrice']
train_data_new['SalePrice'] = train_data['SalePrice']


train_data = train_data_new
test_data = test_data_new



train_data.columns.difference(test_data.columns)
test_data.columns.difference(train_data.columns)

train_data = train_data.drop(['YrSold'], axis = 1)
test_data = test_data.drop(['1stFlrSFsquared'], axis = 1)




#kaggle model: 
#cross validation to get alpha best for ridge
dependentV=train_data["SalePrice"]
train_data.drop("SalePrice", axis=1, inplace=True)

train_data.shape
test_data.shape
X_train, X_test, y_train, y_test = train_test_split(train_data, dependentV, test_size = 0.3, random_state = 123)



#use ridge model since it performs better than lasso in our test and train 
alphaList = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 20, 30, 40, 50, 60]
trainRmseList=[]
testRmseList=[]

ridgeModel = RidgeCV(alphas = alphaList, cv=5)
ridgeModel.fit(train_data, dependentV)
#get the best alpha
alphaBest = ridgeModel.alpha_
print("Best alpha for ridge model is ", alphaBest)
#try to find a more precised alpha
print("Try again for more precision around " + str(alphaBest))
temp=[0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3]
ridgeModel = RidgeCV(alphas = [alphaBest*i for i in temp], cv = 5)
ridgeModel.fit(train_data, dependentV)
#find the more precise alpha
alphaBest = ridgeModel.alpha_
#look at the result 
print("Best alpha for ridge model is ", alphaBest)
y_pred=ridgeModel.predict(train_data)
print("RMSE for ridge model is ", RMSE(y_pred,dependentV))
y_test_pred = ridgeModel.predict(X_test)
print("RMSE for ridge model is out-of-sample ", RMSE(y_test_pred, y_test))
# Plot important coefficients
coefs = pd.Series(ridgeModel.coef_, index = train_data.columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()








train_data.shape


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
adaBoostModel = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=400, learning_rate=0.5)
adaBoostModel.fit(X_train, y_train)
y_trainPred2=adaBoostModel.predict(X_train)
#in sample
print("adaboost in-sample r-squared is")
print(r2_score(y_train, y_trainPred2))
print ("adaboost RMSE is of in-sample")
print(RMSE(y_trainPred2,y_train))
#out of sample
y_testPred2=adaBoostModel.predict(X_test)
print("adaboost out-of-sample r-squared is")
print(r2_score(y_test, y_testPred2))
print ("adaboost RMSE is out-of-sample")
print(RMSE(y_testPred2,y_test))


lassoModel = LassoCV(alphas = alphaList, cv=5)
lassoModel.fit(train_data, dependentV)
#get the best alpha
alphaBest = lassoModel.alpha_
print("Best alpha for lasso model is ", alphaBest)
#try to find a more precised alpha
print("Try again for more precision around " + str(alphaBest))
temp=[0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3]
lassoModel = LassoCV(alphas = [alphaBest*i for i in temp], cv = 5)
lassoModel.fit(train_data, dependentV)
#find the more precise alpha
alphaBest = lassoModel.alpha_
#look at the result 
print("Best alpha for lasso model is ", alphaBest)
y_pred=lassoModel.predict(train_data)
print("RMSE for lasso model is ", RMSE(y_pred,dependentV))
# Plot important coefficients
coefs = pd.Series(lassoModel.coef_, index = train_data.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()



submit = pd.DataFrame()
submit['Id'] = test_data['Id']
array_sale_ada = adaBoostModel.predict(test_data)
array_sale_ridge = ridgeModel.predict(test_data)
array_sale_lasso = lassoModel.predict(test_data)
submit['SalePrice'] = array_sale_ridge*0.92 + array_sale_ada*0.06 + array_sale_lasso*0.02
submit['SalePrice'] = np.exp(submit['SalePrice'])
submit.head(20)
submit.to_csv("/Users/wufei/Desktop/kaggle/submission_3h.csv", index=False)








#drop outliers: 
#train.to_csv("/Users/wufei/Desktop/kaggle/train_01.csv")
#test.to_csv("/Users/wufei/Desktop/kaggle/test_.csv")

