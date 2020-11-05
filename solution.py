#Initialy necessery modules imported.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

#Firstly, train and test sets uploaded.
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Id hasn't any affect on the "SalePrice". So it can be dropped.
train=train.drop(["Id"],axis=1)
test=test.drop(["Id"],axis=1)

#Figure 1 and 2 show the null value disturbution respectively.
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())

#The null values handled in the for loop. 
column_name=test.columns
for i in column_name:
    if test[i].isnull().sum() !=0:
        if test[i].dtype==object:                     #For object type column fill null values by using most frequent values in the same column.
            test[i]=test[i].fillna(test[i].mode()[0])
        else:
            test[i]=test[i].fillna(test[i].mean())    #For numeric type column fill null values by using mean in the same column.
    if train[i].isnull().sum() !=0:
        if train[i].dtype==object:
            train[i]=train[i].fillna(train[i].mode()[0])
        else:
            train[i]=train[i].fillna(train[i].mean())

# Target variable seperated with training set and assign y values.
y=train["SalePrice"]
train=train.drop("SalePrice",axis=1)

## All converts implement both test data and train data.
#"MSSubClass" Column contain numeric values. However this values indicates the categories. So it must be convert into object column type.
train["MSSubClass"]=train.MSSubClass.astype(object)
test["MSSubClass"]=test.MSSubClass.astype(object)

#"YrSold","YearBuilt","YearRemodAdd","GarageYrBlt" column contain date column year part. To detect linear relationship with this column,
#substracting this column from todays date. And, dropped old values.
train["SoldBefore"]=2020-train["YrSold"]
test["SoldBefore"]=2020-test["YrSold"]
train["BuiltBefore"]=2020-train["YearBuilt"]
test["BuiltBefore"]=2020-test["YearBuilt"]
train["RomedBefore"]=2020-train["YearRemodAdd"]
test["RomedBefore"]=2020-test["YearRemodAdd"]
train["GarageYrBltBefore"]=2020-train["GarageYrBlt"]
test["GarageYrBltBefore"]=2020-test["GarageYrBlt"]
train=train.drop(["YearBuilt","YrSold","MoSold","YearRemodAdd","GarageYrBlt"],axis=1)
test=test.drop(["YearBuilt","YrSold","MoSold","YearRemodAdd","GarageYrBlt"],axis=1)

#The main area from house given seperate. To find total area of the house can provide more insightful relation with the target variable.
#So, first floor area, second floor area and basement area  are summed. Using Total Area in the model demolish RMSE nearly %15.
train['TotalArea'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalArea'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# "GardenArea" exatcly before provide more meaningful relation with target variable. It can be find to substract first floor area from lot area.
# This provide to demolish RMSE %5.
train['GardenArea'] = train['LotArea'] - train['1stFlrSF']
test['GardenArea'] = test['LotArea'] - test['1stFlrSF']

#Using total porch area also decrease RMSE.
train["TotalPorch"]=train["OpenPorchSF"]+train["WoodDeckSF"]+train["3SsnPorch"]+train["ScreenPorch"]+train["EnclosedPorch"]
test["TotalPorch"]=test["OpenPorchSF"]+test["WoodDeckSF"]+test["3SsnPorch"]+test["ScreenPorch"]+test["EnclosedPorch"]



#Some subclass have less observation in datasets. So it harden to use this data more informative way. I apply to tighten two or three subclass in one subclass
#due to their descriptive statistics such as mean, standard deviation, 0.25 percentile, 0.75 percentile. I am not sure this must be done or not however,
#in this model, it can improved the model somehow.
di_exterior={"AsphShn":"AsbShng","Brk Cmn":"AsbShng","CBlock":"AsbShng","Wd Sdng":"MetalSD","WdShing":"MetalSD","Stucco":"HdBoard","Plywood":"BrkFace","ImStucc":"CemntBd","Stone":"CemntBd"}
train["Exterior1st"].replace(di_exterior,inplace=True)
test["Exterior1st"].replace(di_exterior,inplace=True)
train["Exterior2nd"].replace(di_exterior,inplace=True)
test["Exterior2nd"].replace(di_exterior,inplace=True)

#The roof material subclass slightly differ from each other. It can easily two group. 
di_roofmatl={"ClyTile":0,"CompShg":0,"Membran":1,"Metal":0,"Roll":0,"Tar&Grv":0,"WdShake":1,"WdShngl":1}
train["RoofMatl"].replace(di_roofmatl,inplace=True)
test["RoofMatl"].replace(di_roofmatl,inplace=True)

#According the describe statistics it can easily convert to int column.For example the price range or distirbution is higher for "1Fam" category than 
#"TwnhsE" category go on like that.
di_bldgtype={"2fmCon":1,"Duplex":2,"Twnhs":3,"TwnhsE":4,"1Fam":5}
train["BldgType"].replace(di_bldgtype,inplace=True)
test["BldgType"].replace(di_bldgtype,inplace=True)

#The LotShape and LandContour completely same way distirbution like BldgType. So the same process implement for this category. 
di_lotshape={"Reg":1,"IR1":4,"IR2":3,"IR3":2}
train["LotShape"].replace(di_lotshape,inplace=True)
test["LotShape"].replace(di_lotshape,inplace=True)

di_landcontor={"Bnk":1,"Lvl":2,"Low":3,"HLS":4}
train["LandContour"].replace(di_landcontor,inplace=True)
test["LandContour"].replace(di_landcontor,inplace=True)

#"ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","GarageQual", and "GarageCond" values indicates relatively situation with each other.
#So converting this column as numeric value give model less complexity.
di_exterqual={"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0}
train["ExterQual"].replace(di_exterqual,inplace=True)
test["ExterQual"].replace(di_exterqual,inplace=True)
train["ExterCond"].replace(di_exterqual,inplace=True)
test["ExterCond"].replace(di_exterqual,inplace=True)
train["BsmtQual"].replace(di_exterqual,inplace=True)
test["BsmtQual"].replace(di_exterqual,inplace=True)
train["BsmtCond"].replace(di_exterqual,inplace=True)
test["BsmtCond"].replace(di_exterqual,inplace=True)
train["HeatingQC"].replace(di_exterqual,inplace=True)
test["HeatingQC"].replace(di_exterqual,inplace=True)
train["KitchenQual"].replace(di_exterqual,inplace=True)
test["KitchenQual"].replace(di_exterqual,inplace=True)
train["GarageQual"].replace(di_exterqual,inplace=True)
test["GarageQual"].replace(di_exterqual,inplace=True)
train["GarageCond"].replace(di_exterqual,inplace=True)
test["GarageCond"].replace(di_exterqual,inplace=True)

#Due to using different combined values and less of efect on target variables some columns drops.
unused_feature=["OpenPorchSF","WoodDeckSF","TotalBsmtSF","3SsnPorch","ScreenPorch","EnclosedPorch","1stFlrSF","2ndFlrSF","LotArea","Alley","MiscFeature","PoolQC","Fence","FireplaceQu","Street","Utilities","Condition2"]
test=test.drop(unused_feature,axis=1)
train=train.drop(unused_feature,axis=1)

#Firstly to demostrate relation with target variable object and integer column name seperated.
def object_or_integer(data,target):
    column_name=data.columns
    object_type=[]
    numeric_type=[]
    for i in column_name:
        if train[i].dtype==object:
            object_type.append(i)
        else:
            numeric_type.append(i)
    return object_type,numeric_type
  
obj_column, int_column = object_or_integer(train,y)

#After seperated column firstly, numeric column correlation with target variable investigated and if correlation values are less than 0.05, 
#the column dropped. Also, p-values are greater than 0.05 again the column is dropped. P-values indicates two features are related in significant important way.
#I am not sure about using p-values like that however it can provide improvement on model performance.
def correlation_with_target(int_column,y):
    corre={}
    p_value={}
    dropcolumn=[]
    for i in int_column:
        corr = pearsonr(train[i],y)
        if (abs(corr[0]) <0.05) or abs(corr[1])>0.05 :
            dropcolumn.append(i)
        corre[i]=corr[0]
        p_value[i]=corr[1]
    # plt.scatter(corre.keys(),corre.values())
    # plt.xticks(rotation=90)
    return corre,dropcolumn,p_value

corr,dropcolumnfeature,p_values=correlation_with_target(int_column,y)
dropcolumnfeature=["ExterCond","BsmtFinSF2","LowQualFinSF","BsmtHalfBath","MiscVal","SoldBefore"]
test=test.drop(dropcolumnfeature,axis=1)
train=train.drop(dropcolumnfeature,axis=1)

#Categorical feature various with target value can serve good relationship. So, I create a list with describe statistics with categorical feature dataframe.
def distirbution_with_target_mean(train,obj_column,y):
    distirbution=[]
    a=train.copy()
    a["y"]=y
    for i in obj_column:
        dist=a.groupby(i).y.describe()
        distirbution.append(dist)
    return distirbution
  
distirbutions= distirbution_with_target_mean(train, obj_column,y)

#Outliers can be the other effect feature on the model performance.Changing the bigger value than 0.75 percentile with the 0.75 percentile and
#add new column actually indicating the value is bigger than 0.75 percentile can handle with the outliers.
big_data_handle=["LotFrontage","MasVnrArea","BsmtFinSF1","BsmtUnfSF","GrLivArea","GarageArea","TotalPorch","TotalArea","GardenArea"]
for ii in big_data_handle:
    train[str(ii+"_big")] = (train[ii]>np.percentile(train[ii],75)).astype(float)
    test[str(ii+"_big")] = (test[ii]>np.percentile(test[ii],75)).astype(float)
    train.loc[train[ii]>np.percentile(train[ii],75),ii]=np.percentile(train[ii],75)
    test.loc[test[ii]>np.percentile(test[ii],75),ii]=np.percentile(test[ii],75)
    
 #To handle with the dummy variables and the same corelated feature each other train and test datasets merge together.
 finalset = train.append(test,sort=False)

 def to_drop_same_correlated_feature(data):
    covarianceMatrix = data.corr()
    listOfFeatures = [i for i in covarianceMatrix]
    Dropped_Features = set() 
    for i in range(len(listOfFeatures)) :
        for j in range(i+1,len(listOfFeatures)): #Avoid repetitions 
            feature1=listOfFeatures[i]
            feature2=listOfFeatures[j]
            if abs(covarianceMatrix[feature1][feature2]) > 0.75: #If the correlation between the features is > 0.8
                Dropped_Features.add(feature1) 

    data = data.drop(Dropped_Features, axis=1)
    return data
  
finalset=to_drop_same_correlated_feature(finalset)

#To get dummy variables:
def category_onehot_other(colums,finalset):
    setfinal=finalset
    i=0
    for field in colums:
        df1=pd.get_dummies(finalset[field],drop_first=True,prefix=field)
        finalset.drop([field],axis=1,inplace=True)
        if i ==0:
            setfinal=df1.copy()
        else:
            setfinal=pd.concat([setfinal,df1],axis=1)
        i=i+1
    setfinal=pd.concat([finalset,setfinal],axis=1)
    return setfinal
  
df=category_onehot_other(obj_column,finalset)

#To seperate df_test and df_train:
df_train=df.iloc[:1460,:].values
df_test=df.iloc[1460:,:].values
y=y.values

###Firstly a hibrid model system applied for the data. So to detect coeffient of model effect on target variables firstly the uncomment codes are applied.
###Also, I dont directly get the score, I tried to minimize RMSE. 
# X_train,X_test,y_train,y_test=train_test_split(df_train,y,test_size=0.2,random_state=21)

# gb=GradientBoostingRegressor()
# gb.fit(X_train,y_train)
# y_pred_gb=gb.predict(X_test)

# rf=RandomForestRegressor()
# rf.fit(X_train,y_train)
# y_pred_rf=rf.predict(X_test)

# ls=Lasso()
# ls.fit(X_train,y_train)
# y_pred_ls=ls.predict(X_test)

# rd=Ridge()
# rd.fit(X_train,y_train)
# y_pred_rd=rd.predict(X_test)

# ln=LinearRegression()
# ln.fit(X_train,y_train)
# y_pred_ln=ln.predict(X_test)


# plt.plot(y_pred_rf,label="RF")
# plt.plot(y_pred_ls,label="LS")
# plt.plot(y_pred_rd,label="RD")
# plt.plot(y_pred_ln,label="LN")
# plt.plot(y_pred_gb,label="GB")
# plt.plot(y_test,alpha=0.5,label="True")
# plt.legend()
# plt.show()

# print("GB: ",MSE(y_test,y_pred_gb)**0.5)
# print("RF: ",MSE(y_test,y_pred_rf)**0.5)
# print("LS: ",MSE(y_test,y_pred_ls)**0.5)
# print("RD: ",MSE(y_test,y_pred_rd)**0.5)
# print("LN: ",MSE(y_test,y_pred_ln)**0.5)

# mean_squ={}
# for i in range(11):
#     for j in range(11-i):
#         for k in range(11-(i+j)):
#             for l in range(11-(i+j+k)):
#                 for ii in range(11-(i+j+k+l)):
#                     y_pred=(i*y_pred_rf+j*y_pred_ls+k*y_pred_rd+l*y_pred_ln+ii*y_pred_gb)/10
#                     means=MSE(y_test,y_pred)**0.5
#                     mean_squ[str(i)+"*rf+"+str(j)+"*ls+"+str(k)+"*rd+"+str(l)+"*ln+"+str(ii)+"*gb"]=means
# min_mean=min(mean_squ.values())
# print(min_mean)
# a=[key for key in mean_squ if mean_squ[key]==min_mean]
###After some tried 0.3*y_pred_rf, 0.3*y_pred_rd, 0.4*y_pred_gb give the optimum result.

##To find optimum result three model result combination is used.
gb=GradientBoostingRegressor()
gb.fit(df_train,y)
y_pred_gb=gb.predict(df_test)

rf=RandomForestRegressor()
rf.fit(df_train,y)
y_pred_rf=rf.predict(df_test)

rd=Ridge()
rd.fit(df_train,y)
y_pred_rd=rd.predict(df_test)

y_pred=0.3*y_pred_rf+0.3*y_pred_rd+0.4*y_pred_gb

#To sumbit the score:
def submission(csv_file,y_pred):
    sub_df=pd.read_csv(csv_file)
    datasets=pd.DataFrame({"Id":sub_df["Id"],"SalePrice":y_pred})
    datasets.to_csv("sample_submission1.csv",index=False)
    
submission("sample_submission.csv", y_pred)

###After submitting I get  0.13 score. I have tried to improve model performance.
