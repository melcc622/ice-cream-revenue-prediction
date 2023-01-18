#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# ### Import dataset

# In[7]:


df = pd.read_csv('IceCreamData.csv')
df.head()


# In[8]:


df.info()


# In[9]:


df.describe()
# We have a dataset of two columns: `Temperature` and `Revenue`

# Some questions I want to explore and test:

# 1. I hypothesize that temperature and revenue have a strong positive association
# 2. What 


# ## Data Visualization

# In[10]:


plt.hist(df['Revenue'],color='green')


# In[11]:


plt.hist(df['Temperature'],color='orange')


# In[12]:


# The variables are roughly normal according to the histograms
# We can verify according to the Shapiro Normality Test

temp_shapiro = stats.shapiro(df['Temperature'])
rev_shapiro = stats.shapiro(df['Revenue'])

print(f"The Results are {temp_shapiro} & {rev_shapiro}")


# We can conclude that the variables are normally distributed
# based on the Shapiro Wilk's normality tests

# 


# In[13]:


plt.figure(dpi=(500))
sns.pairplot(df)


# In[14]:


sns.pairplot(df, kind="kde")


# ## Checking Linear Regression Assumptions
# 

# ## Fitting the Model

# In[17]:


df.iloc[:,:-1]


# ## Splitting the dataset using `train_test_split`

# In[21]:


# Splitting the dataset

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

TempTrain, TempTest, RevTrain, RevTest = train_test_split(X,Y,test_size=0.3,random_state=0)


# In[24]:


# Brief look at the splits

print('Training Temperature: ', TempTrain.shape, 'Training Revenue: ', RevTrain.shape,
     'Testing Temperature: ', TempTest.shape, 'Testing Revenue: ', RevTest.shape)


# In[27]:


lnr = LinearRegression()
lnr.fit(TempTrain,RevTrain)


# ### Regression Formula: 
# $$ \hat{Revenue} = slope \times \hat{Temperature} + intercept $$

# In[31]:


print('Linear Model Coefficient (slope): ', lnr.coef_)
print('Linear Model Coefficient (intercept): ', lnr.intercept_)


# In[32]:


# Prediction

RevPred = lnr.predict(TempTest)
np.set_printoptions(precision=2)
print('Revenue Predictions Preview')
print((np.concatenate((RevPred.reshape(len(RevPred),1),RevTest.reshape(len(RevTest),1)),1))[:11,])


# ### Assessing the Predictions

# In[41]:


# plotting the Training set predictions 

plt.scatter(TempTrain, RevTrain, color='pink')
plt.plot(TempTrain, lnr.predict(TempTrain), color = 'blue')
plt.ylabel('{} ($)'.format(df.columns[1]))
plt.xlabel('{} (Celsius)'.format(df.columns[0]))
plt.title('Ice Cream Revenue (Training dataset)')


# In[43]:


# plotting the Testing set predictions 

plt.scatter(TempTest, RevTest, color='pink')
plt.plot(TempTest, lnr.predict(TempTest), color = 'green')
plt.ylabel('{} ($)'.format(df.columns[1]))
plt.xlabel('{} (Celsius)'.format(df.columns[0]))
plt.title('Ice Cream Revenue (Testing dataset)')


# $$R^2, RMSE, MSE, MAE $$

# In[36]:


from sklearn import metrics
r2_score = metrics.r2_score(RevTest, RevPred)
mae = metrics.mean_absolute_error(RevTest, RevPred)
mse = metrics.mean_squared_error(RevTest, RevPred)
rmse = np.sqrt(mean_squared_error(RevTest, RevPred))


# In[67]:


# print(f'The R2 score for Linear Regression Predictions are {r2_score}')
# print(f'The MAE for Linear Regression Predictions are {mae}')
# print(f'The MSE for Linear Regression Predictions are {mse}')
# print(f'The RMSE for Linear Regression Predictions are {rmse}')
ols_results = pd.DataFrame([['OLS Linear Regression', mae, mse, rmse, r2_score]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

ols_results


# ## Regression Tree

# In[45]:


from sklearn.tree import DecisionTreeRegressor

regtree = DecisionTreeRegressor(random_state = 0)

regtree.fit(TempTrain,RevTrain)


# In[53]:


# Testing Set predictions
RevPredTree = regtree.predict(TempTest)
print('Some Revenue Predictions:',(RevPredTree)[:11])


# ### Evaluation metrics for the regression tree

# In[57]:


tree_mae = metrics.mean_absolute_error(RevTest, RevPredTree)
tree_mse = metrics.mean_squared_error(RevTest, RevPredTree)
tree_rmse = np.sqrt(metrics.mean_squared_error(RevTest, RevPredTree))
tree_r2 = metrics.r2_score(RevTest, RevPredTree)

tree_results = pd.DataFrame([['Decision Tree Regression', tree_mae, tree_mse, tree_rmse, tree_r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

tree_results


# In[48]:


from sklearn.tree import plot_tree
plt.figure(figsize=(10,8), dpi=200)
plot_tree(regtree)


# ## Random Forest

# In[63]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)

rf.fit(TempTrain,RevTrain)


# In[68]:


# make predictions

rf_pred = rf.predict(TempTest)

forest_mae = metrics.mean_absolute_error(RevTest, rf_pred)
forest_mse = metrics.mean_squared_error(RevTest, rf_pred)
forest_rmse = np.sqrt(metrics.mean_squared_error(RevTest, rf_pred))
forest_r2 = metrics.r2_score(RevTest, rf_pred)

forest_results = pd.DataFrame([['Random Forest', forest_mae, forest_mse, forest_rmse, forest_r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

forest_results

# random forest algorithm is relatively better than decision tree


# ## Boosting Algorithms

# ## AdaBoost

# In[70]:


from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor()
ada.fit(TempTrain, RevTrain)


# In[73]:


# Testing Set predictions 
AdaRevPred = ada.predict(TempTest)

ada_mae = metrics.mean_absolute_error(RevTest, AdaRevPred)
ada_mse = metrics.mean_squared_error(RevTest, AdaRevPred)
ada_rmse = np.sqrt(metrics.mean_squared_error(RevTest, AdaRevPred))
ada_r2 = metrics.r2_score(RevTest, AdaRevPred)

ada_results = pd.DataFrame([['AdaBoost Regressor', ada_mae, ada_mse, ada_rmse, ada_r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

ada_results


# ## Gradient Boost

# In[74]:


from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb.fit(TempTrain, RevTrain)


# In[75]:


#  Testing Set predictions

GBRevPred = gb.predict(TempTest)

gb_mae = metrics.mean_absolute_error(RevTest, GBRevPred)
gb_mse = metrics.mean_squared_error(RevTest, GBRevPred)
gb_rmse = np.sqrt(metrics.mean_squared_error(RevTest, GBRevPred))
gb_r2 = metrics.r2_score(RevTest, GBRevPred)

gb_results = pd.DataFrame([['GradientBoosting Regressor', gb_mae, gb_mse, gb_rmse, gb_r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

gb_results


# ## Evaluation Summary of Algorithms

# In[81]:


ols_results, tree_results, forest_results ,ada_results,gb_results


# In[ ]:


# In general, the OLS Simple linear regression has the best predictive probability for the ice cream revenue dataset 

# It has the lowest evaluation metrics for MAE, MSE, RMSE, and highest R-squared value

