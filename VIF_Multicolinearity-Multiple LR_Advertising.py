#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


adv=pd.read_csv("Multiple_LR_Advertising.csv")


# In[3]:


adv.head()


# In[4]:


adv.tail()


# In[5]:


adv.info()


# In[6]:


adv.describe()


# In[7]:


adv.corr()


# In[8]:


adv.boxplot('TV')


# In[9]:


adv.boxplot('radio')


# In[10]:


adv.boxplot('newspaper')


# In[11]:


adv.boxplot('sales')


# In[12]:


plt.scatter(x=adv['TV'],y=adv['radio'])
plt.title("Multiple_LR_Advertising.csv")
plt.xlabel("TV")
plt.ylabel("radio")
plt.figure(figsize=(15,8))


# In[13]:


plt.scatter(x=adv['newspaper'],y=adv['radio'])
plt.title("Multiple_LR_Advertising.csv")
plt.xlabel("newspaper")
plt.ylabel("radio")
plt.figure(figsize=(15,8))


# # all the dependent variable should be linearly related to the independent variable

# # there should be no Multi-colinearity. Independent variable should not be having any correlation against themselves

# # Model 1 

# In[14]:


X=adv[['TV','newspaper','radio']]
X.head()


# In[15]:


Y=adv[['sales']]
Y.head()


# In[16]:


from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[17]:


X_train.head()


# In[18]:


X_train.shape


# In[19]:


Y_train.head()


# In[20]:


Y_train.shape


# In[21]:


X_test.shape


# In[22]:


Y_test.shape


# # Evaluate the Model

# In[23]:


from sklearn.linear_model import LinearRegression


# In[24]:


lr=LinearRegression()


# In[25]:


lr.fit(X,Y)


# In[26]:


Y_predict=lr.predict(X_test)


# In[27]:


Y_predict


# In[28]:


Y_predict=lr.predict(X_test)


# In[29]:


lr.intercept_


# In[30]:


lr.coef_


# #coeff_adv=pd.DataFrame(lr.coef_,X_test.columns,columns=['Co])
# coeff_adv
# 

# In[31]:


from sklearn import metrics


# In[32]:


r_squared=metrics.r2_score(Y_test,Y_predict)


# In[33]:


r_squared


# In[34]:


mse=metrics.mean_squared_error(Y_test,Y_predict)


# In[35]:


mse


# In[36]:


rmse=np.sqrt(mse)


# In[37]:


AdjustedR= (1 - ((1-r_squared)*199)/(200-3-1))
AdjustedR


# # 1-[(1-r_squared)*(n-1)]/(n-k-1) where n= observations and k is the number of variables

# # Model II

# In[38]:


X=adv[['TV','radio']]


# In[39]:


X


# In[40]:


Y=adv['sales']


# In[41]:


Y


# In[42]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[43]:


lr=LinearRegression()


# In[44]:


lr.fit(X_train,Y_train)
Y_predict=lr.predict(X_test)


# In[45]:


lr.intercept_


# In[46]:


lr.coef_


# In[47]:


coeff_adv = pd.DataFrame(lr.coef_,index=X_test.columns,columns=['Coefficient'])
coeff_adv


# In[48]:


from sklearn import metrics


# In[49]:


r_squared=metrics.r2_score(Y_test,Y_predict)


# In[50]:


r_squared


# In[51]:


mse=metrics.mean_squared_error(Y_test,Y_predict)


# In[52]:


mse


# In[53]:


rmse=np.sqrt(mse)


# In[54]:


rmse


# In[55]:


adjusted_r=1-((1-r_squared)*199)/(200-2-1)


# In[56]:


adjusted_r


# In[57]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["features"] = X_test.columns
vif["VIF Factor"] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]
vif


# vif should not be more than 10 because it affects other variables as well in the overall equation
# 

# In[58]:


newdf= pd.DataFrame(Y_test.values, columns=['Original Y-Test'])
newdf.head()


# In[59]:


newdf = pd.DataFrame(Y_test.values, columns=['Original Y-Test'])
newdf.head()
newdf['Predicted Y-Test'] = pd.Series(Y_predict)
newdf.head()
newdf['Residual'] = newdf['Original Y-Test'] - newdf['Predicted Y-Test']


# In[60]:


newdf.head()


# # Assumption-3 No correlation between the residuals
# One way to determine if this assumption is met is to perform a Durbin-Watson test, which is used to detect the presence of autocorrelation in
# the residuals of a regression. This test uses the following hypotheses:
# H0 (null hypothesis): There is no correlation among the residuals.
# HA (alternative hypothesis): The residuals are autocorrelated.
# The test statistic is approximately equal to 2*(1-r) where r is the sample autocorrelation of the residuals. Thus, the test statistic will always be
# between 0 and 4 with the following interpretation:
# A test statistic of 2 indicates no serial correlation.
# The closer the test statistics is to 0, the more evidence of positive serial correlation.
# The closer the test statistics is to 4, the more evidence of negative serial correlation.
# As a rule of thumb, test statistic values between the range of 1.5 and 2.5 are considered normal.
# 

# In[61]:


from statsmodels.stats.stattools import durbin_watson


# In[62]:


durbin_watson(newdf['Residual']) ## no correlation between residuals


# In[63]:


Check_X = X_test.iloc[1:2]


# In[64]:


lr.predict(Check_X)


# In[65]:


lr.predict([[25,40]])


# In[66]:


Check_X


# In[67]:


lr.predict([[20,30]])


# In[68]:


lr.predict([[55,65]])


# # Model III

# In[69]:


np.random.seed(12345)
# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(adv))
mask_large = nums > 0.5
# initially set Size to small, then change roughly half to be large
adv['Size'] = 'small'
# Series.loc is a purely label-location based indexer for selection by label
adv.loc[mask_large, 'Size'] = 'large'
adv.head()
# set a seed for reproducibility
np.random.seed(123456)
# assign roughly one third of observations to each group
nums = np.random.rand(len(adv))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
adv['Area'] = 'rural'
# Series.loc is a purely label-location based indexer for selection by label
adv.loc[mask_suburban, 'Area'] = 'suburban'
adv.loc[mask_urban, 'Area'] = 'urban'
adv.head()


# In[70]:


adv.drop('newspaper',axis=1,inplace=True)


# In[71]:


adv


# In[72]:


adv.head()


# In[73]:


adv_dummies=pd.get_dummies(adv,drop_first=True)


# In[74]:


adv_dummies.head()


# In[75]:


X=adv_dummies[['TV','radio','Size_small','Area_suburban','Area_urban']]


# In[76]:


Y=adv_dummies['sales']


# In[77]:


X_train,Y_test,Y_train,Y_test,train_test_split(X,Y,test_size=0.2,random_state=0)


# In[78]:


lr=LinearRegression()


# In[79]:


lr.fit(X_train,Y_train)


# In[80]:


Y_predict=lr.predict(X_test)


# In[81]:


r_squared=metrics.r2_score(Y_test,Y_predict)
r_squared


# In[82]:


mse=metrics.mean_squared_error(Y_test,Y_predict)
mse


# rmse=np.sqrt(mse)
# rmse

# In[83]:


rmse=np.sqrt(mse)
rmse


# In[84]:


adv_dummies.count()


# In[85]:


adjusted_r=1-((1-r_squared)*199)/(200-5-1)
adjusted_r


# In[86]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["features"] = X_test.columns
vif["VIF Factor"] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]
vif


# In[87]:


newdf = pd.DataFrame(Y_test.values, columns=['Original Y-Test'])
newdf.head()
newdf['Predicted Y-Test'] = pd.Series(Y_predict)
newdf.head()
newdf['Residual'] = newdf['Original Y-Test'] - newdf['Predicted Y-Test']


# In[88]:


newdf


# In[89]:


from statsmodels.stats.stattools import durbin_watson
durbin_watson(newdf['Residual'])


# # Assumption 4: This states that the residuals will be normaly distributed

# In[90]:


sns.distplot(newdf['Residual'])

