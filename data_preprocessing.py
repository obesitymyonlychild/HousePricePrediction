#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:39:35 2020

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

#train_data=pd.read_csv("/Users/wufei/Desktop/kaggle/train.csv")
train_data=pd.read_csv("Users/postgres/Desktop/data/train.csv")


#train_data['SalePrice'] = np.log(train_data['SalePrice'])
train_data.index=train_data["Id"]
train_data.drop("Id", axis=1, inplace=True)


train_data=train_data.replace({"Alley":{"Grvl" : 1, "Pave" : 2}})
train_data=train_data.replace({"LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3}})
train_data=train_data.replace({"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}})
train_data['ExterCond'] = train_data['ExterCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0])
train_data['ExterQual'] = train_data['ExterQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0])
train_data['HeatingQC'] = train_data['HeatingQC'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0])
train_data['KitchenQual'] = train_data['KitchenQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [4, 3, 2, 1, 0])
train_data['BsmtQual'] = train_data['BsmtQual'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], [5, 4, 3, 2, 1, 0])
train_data['BsmtCond'] = train_data['BsmtCond'].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], [5, 4, 3, 2, 1, 0])
train_data['BsmtExposure'] = train_data['BsmtExposure'].replace(['Gd', 'Av', 'Mn', 'No', 'NA'], [4, 3, 2, 1 ,0])
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].replace(['GLQ','ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], [6, 5, 4 ,3, 2, 1, 0])
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].replace(['GLQ','ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], [6, 5, 4 ,3, 2, 1, 0])
train_data['Functional'] = train_data['Functional'].replace(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'], [7, 6, 5, 4, 3, 2, 1 ,0])
train_data['PavedDrive'] = train_data['PavedDrive'].replace(['Y', 'P', 'N'], [2, 1, 0])
train_data["Condition1"] = train_data.Condition1.replace({"RRNe" : "Other", 
                                                  "RRNn" : "Other","PosA" : "Other", 
                                                   "RRAe" : "Other"
                                                  })
train_data["Electrical"] = train_data.Electrical.replace({"Mix" : "Other", 
                                                  "FuseP" : "Other"
                                                  })
train_data["Exterior1st"] = train_data.Exterior1st.replace({"AsphShn" : "Other", 
                                                  "CBlock" : "Other","ImStucc" : "Other", 
                                                   "BrkComm" : "Other","Stone" : "Other"
                                                  })
train_data["Exterior2nd"] = train_data.Exterior2nd.replace({"CBlock" : "Other", 
                                                  "AsphShn" : "Other","Stone" : "Other", 
                                                   "Brk Cmn" : "Other","ImStucc" : "Other"
                                                  })
train_data["Foundation"] = train_data.Foundation.replace({"Wood" : "Other", 
                                                  "Stone" : "Other"
                                                  })

train_data["HouseStyle"] = train_data.HouseStyle.replace({"2.5Fin" : "Other", 
                                                  "2.5Unf" : "Other",
                                                  "1.5Unf" : "Other"
                                                  })
train_data["LotConfig"] = train_data.LotConfig.replace({"FR3" : "FR2"
                                                  })

train_data["MSZoning"] = train_data.MSZoning.replace({"C (all)" : "Other", 
                                                  "RH" : "Other"
                                                  })
train_data["Neighborhood"] = train_data.Neighborhood.replace({"Blueste" : "Other", 
                                                  "NPkVill" : "Other",
                                                  "Veenker" : "Other"
                                                  })
train_data["RoofStyle"] = train_data.RoofStyle.replace({"Shed" : "Other", 
                                                  "Mansard" : "Other",
                                                  "Gambrel" : "Other",
                                                  "Flat" : "Other"
                                                  })
train_data["SaleCondition"] = train_data.SaleCondition.replace({"AdjLand" : "Other", 
                                                  "Alloca" : "Other"
                                                  })
train_data["SaleType"] = train_data.SaleType.replace({"Con" : "Other", 
                                                  "Oth" : "Other",
                                                  "CWD" : "Other", "ConLI" : "Other",
                                                  "ConLw" : "Other","ConLD" : "Other"
                                                  })


train_data['Total_Square_Feet'] = (train_data['BsmtFinSF1'] + train_data['BsmtFinSF2'] + train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF'])
train_data['Total_Bath'] = (train_data['FullBath'] + (0.5 * train_data['HalfBath']) + train_data['BsmtFullBath'] + (0.5 * train_data['BsmtHalfBath']))
train_data['Total_Porch_Area'] = (train_data['OpenPorchSF'] + train_data['3SsnPorch'] + train_data['EnclosedPorch'] + train_data['ScreenPorch'] + train_data['WoodDeckSF'])
train_data["KitchenScore"] = train_data["KitchenAbvGr"] * train_data["KitchenQual"]


#fill na

for column in train_data:
    perc = train_data[column].isnull().sum()/train_data.shape[1]
    
    if perc >= 0.80:
        train_data = train_data.drop([column], axis = 1)
    else:
        if(pd.isna(train_data[column].mode()[0])):
            train_data = train_data.drop([column], axis = 1)
        else:
            if is_numeric_dtype(train_data[column]):
                if (column == 'SalePrice' or column == 'MSZoing' or column == 'OverallCond' or
            column == 'OverallQual'):
                    #replacing OverallCond and OverallQual missing data with mode.
                    #replacing remaining numeric variables' (with low median) missing data with 0.
                    #variables similar to condition and quality with missing value filled with 0.
                    train_data[column] = train_data[column].fillna(train_data[column].mode())
                else:  
                    #replacing other numeric variables' missing data with median.
                    train_data[column] = train_data[column].fillna(train_data[column].median())
            elif(is_string_dtype([column])):
                #replacing numeric variables with New Category None if there are less than 4 categories
                #else using the most frequent category.
                #also if there is one category that appears more frequent than 80%, remove the column.
                if (train_data[column].value_counts()[train_data[column].mode()[0]]/train_data.shape[0] > 0.8):
                    train_data = train_data.drop([column], axis = 1)
                else:
                    if (train_data[column].unique() < 4):
                        train_data[column] = train_data[column].fillna("None")
                    else:
                        train_data[column] = train_data[column].fillna(train_data[column].mode()[0])
            else:
                train_data[column] = train_data[column].fillna(train_data[column].mode()[0])
 

#check whether there are still missing cells 
train_data.isnull().sum().max()
train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]
train_data.shape



#delete skew categorical before transform to dummy
categorical_features = train_data.select_dtypes(include = ["object"]).columns
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
#numerical_features = numerical_features.drop("SalePrice")
train_cat = train_data[categorical_features]
pct=[]
for ix in train_cat.columns:
    temp=train_cat[ix].describe()
    pct.append(temp["freq"]/temp["count"])
skewData=pd.DataFrame(pct,index=train_cat.columns,columns=["skewness"])
skewData=skewData.sort_values(by="skewness",ascending=False)
train_data = train_data.drop((skewData[skewData['skewness'] >= 0.9]).index,1) 



for column in numerical_features:
    if column == 'SalePrice':
        pass
    if train_data[column].skew() > 3:
        train_data[column] = np.log(train_data[column])



#feature engineering to get some variables more useful
train_data["OverallGrade"] = train_data["OverallQual"] * train_data["OverallCond"]
train_data["AllSF"] = train_data["GrLivArea"] + train_data["TotalBsmtSF"]
train_data.drop(["GrLivArea","TotalBsmtSF"],axis=1)
train_data["AllFlrsSF"] = train_data["1stFlrSF"] + train_data["2ndFlrSF"]
train_data.drop(["1stFlrSF","2ndFlrSF"],axis=1)
train_data["BoughtOffPlan"] = train_data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})





numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
numerical_features = numerical_features.drop("Id")

for column in numerical_features:
    new_col =  column + "squared"
    train_data[new_col] = train_data[column] ** 2
    corr = train_data.corr()
    if corr.SalePrice[new_col] > corr.SalePrice[column]:
        train_data.drop([column], axis = 1)
    else:
        train_data.drop([new_col], axis = 1)

numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
numerical_features = numerical_features.drop("Id")

        
for column in numerical_features:
    new_col= column + "log"
    train_data[new_col] = np.log(train_data[column] + 1e-28)
    corr2 = train_data.corr()
    if corr2.SalePrice[new_col] > corr2.SalePrice[column]:
        train_data.drop([column], axis = 1)
    else:
        train_data.drop([new_col], axis = 1)


#get dummies:
categorical_features = train_data.select_dtypes(include = ["object"]).columns
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")

train_num = train_data[numerical_features]
train_num.shape
train_cat = train_data[categorical_features]
train_num = train_num.fillna(train_num.median())


#2.6 turn category features to dummy
train_cat = pd.get_dummies(train_cat, drop_first=True)

#2.7 join categorical and numerical features 
train_data_new = pd.concat([train_num, train_cat], axis = 1)

#remove collinear columns: 
corr_matrix = train_data_new.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
train_data_new.drop(to_drop, axis=1, inplace=True)
train_data_new['SalePrice'] = train_data['SalePrice']


train_data = train_data_new

train_data.columns.difference(test_data.columns)


train_data = train_data.drop(['YrSold'], axis = 1)
#train_data = train_data.drop(['1stFlrSFsquared'], axis = 1)



#train_data.drop([197,810,1170,1182,1298,1386,1423],inplace=True)

#train_data.to_csv("/Users/wufei/Desktop/kaggle/train_2.csv.csv")
train_data.to_csv("/Users/wufei/Desktop/kaggle/test_2.csv")










