# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:34:14 2021

@author: khare
"""

## ASSIGNMENT-"Multiple linear Regression"
## Name- SATYAM RAJ KHARE



import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

#1.)An analytics company has been tasked with the crucial job of finding out what 
#factors affect a startup company and if it will be profitable or not. For this, 
#they have collected some historical data and would like to apply multilinear 
#regression to derive brief insights into their data. Predict profit, given different
#attributes for various startup companies.
# loading the data
df= pd.read_csv(r"C:/Users/khare/Downloads/MLR/assign/Datasets_MLR\50_Startups.csv")

# Preprocessing & Exploratory data analysis:

df.isna().sum() # no missing values
df.duplicated().sum() # no duplicates 

D=df.describe()
Skewness=df.skew()
Kurtosis=df.kurt()
df.columns = ['research', 'Admin', 'Marketing_Spend', 'State', 'Profit']

#Convert State feature from categorical to numerical
lb = LabelEncoder() # label Encoder 
df["State"] = lb.fit_transform(df["State"])

#Graphical Representation

#R&D_Spend
plt.bar(height = df['research'], x = np.arange(1,51,1))
plt.hist(df['research']) #histogram
plt.boxplot(df['research']) #boxplot

# Admin
plt.bar(height =  df['Admin'], x = np.arange(1,51, 1))
plt.hist( df['Admin']) #histogram
plt.boxplot( df['Admin']) #boxplot

# Marketing_Spend
plt.bar(height =  df['Marketing_Spend'], x = np.arange(1,51, 1))
plt.hist( df['Marketing_Spend']) #histogram
plt.boxplot( df['Marketing_Spend']) #boxplot

# State
plt.bar(height =  df['State'], x = np.arange(1,51, 1))
plt.hist( df['State']) #histogram
plt.boxplot( df['State']) #boxplot

# Profit
plt.bar(height =  df['Profit'], x = np.arange(1,51, 1))
plt.hist( df['Profit']) #histogram
plt.boxplot( df['Profit']) #boxplot

# Jointplot

sns.jointplot(x=df['research'], y=df['Profit'])
sns.jointplot(x=df['Admin'], y=df['Profit'])
sns.jointplot(x=df['Marketing_Spend'], y=df['Profit'])
sns.jointplot(x=df['State'], y=df['Profit'])

# Correlation matrix 
correlation=df.corr()

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(df["research"])
sns.countplot(df["Admin"])
sns.countplot(df["Marketing_Spend"])
sns.countplot(df["State"])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(df['Profit'], dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
                             
# we see there exists collinearity between input variables 
# [research & Market_Spending]so there exists collinearity problem

############# preparing model considering all the variables########################
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit~ research + Admin + Marketing_Spend + State', data = df).fit() # regression model

# Summary
ml1.summary()
# p-values for Admin,Marketing_Spend,State are more than 0.05

pred1 = ml1.predict(pd.DataFrame(df))

###Error calculation
# residual values 
resid1 = pred1 - df.Profit
# RMSE value for test data 
rmse1 = np.sqrt(np.mean(resid1 * resid1))
rmse1 

##R-squared value
Rsquared1 =ml1.rsquared 
Rsquared1 
####Added Vraiable Plot##

sm.graphics.plot_partregress_grid(ml1)
# State is near to zero has influence on other features

###Influence Index Plots##
# Checking whether data has any influential values 
sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 45,46,48,49 is showing high influence so we can exclude that entire row

df = df.drop(df.index[[45,46,48,49]])

##################Preparing model after removing Influencial Index#############               
ml2 = smf.ols('Profit~ research + Admin + Marketing_Spend + State', data = df).fit()    

# Summary
ml2.summary()
# Droping index 45,48,49 lower down p value of Marketing_Spend,State,Admin

pred2 = ml1.predict(pd.DataFrame(df))

###Error calculation
# residual values 
resid2 = pred2 - df.Profit
# RMSE value for test data 
rmse2 = np.sqrt(np.mean(resid2 * resid2))
rmse2 

##R-squared value
Rsquared2 =ml2.rsquared 
Rsquared2 



######## VIF ######
# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_research = smf.ols('research ~  Admin + Marketing_Spend + State', data = df).fit().rsquared  
vif_research = 1/(1 - rsq_research) 

rsq_Admin = smf.ols('Admin ~  research + Marketing_Spend + State', data = df).fit().rsquared  
vif_Admin = 1/(1 - rsq_Admin)

rsq_MS = smf.ols(' Marketing_Spend ~  research + Admin + State', data = df).fit().rsquared  
vif_MS = 1/(1 - rsq_MS) 

rsq_State = smf.ols(' State ~ research + Admin + Marketing_Spend', data = df).fit().rsquared  
vif_State = 1/(1 - rsq_State ) 

# Storing vif values in a data frame
d1 = {'Variables':['research', 'Admin', 'Marketing_Spend', 'State'], 'VIF':[vif_research, vif_Admin, vif_MS, vif_State]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# VIF of all values is lower than 10 


########################### Final-Test Model ###############################
# In Final model we drop State feature,as in Partial Regression Plot it is near to Zero.
# Apply transformation on Admin feature to make P-value less than 0.05
df["Admin"] = df.Admin**-5

# Final-Test model
final_Test_ml = smf.ols('Profit~ research + Marketing_Spend + Admin',data=df).fit()
final_Test_ml.summary() 

# Prediction
pred = final_Test_ml.predict(df)

###Error calculation
# residual values 
resid3 = pred - df.Profit
# RMSE value for test data 
rmse3 = np.sqrt(np.mean(resid3 * resid3))
rmse3 

##R-squared value
Rsquared3 =final_Test_ml.rsquared 
Rsquared3 

# Q-Q plot
res = final_Test_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q llot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = df.Profit, lowess = True)
plt.xlaoel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_Test_ml)

sm.graphics.plot_partregress_grid(final_Test_ml)

##################### Final Model ###########################################
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.3,random_state=0) # 30% test data

# preparing the model on train data 
model = smf.ols("Profit ~ research  + Marketing_Spend + Admin", data = df_train).fit()

# prediction on test data set 
test_pred = model.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

##R-squared value
Rsquared =model.rsquared 
Rsquared

# RMSE Table
data1 = {"MODEL":pd.Series(["ml1", "ml2", "Final_Test_ml","model(test)" ,"model(train)"]), "RMSE":pd.Series([rmse1, rmse2, rmse3,test_rmse,train_rmse])}
table_rmse = pd.DataFrame(data1)
table_rmse

#R-squared Table
data2 = {"MODEL":pd.Series(["ml1", "ml2", "Final_Test_ml","model"]), "R-squared":pd.Series([Rsquared1, Rsquared2, Rsquared3,Rsquared])}
table_rsqr = pd.DataFrame(data2)
table_rsqr
