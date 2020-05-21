#!/usr/bin/env python
# coding: utf-8

# # Task 1 - Regression (with outliers)

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import janitor
import pandas_flavor as pf
import statsmodels.api as sm
import impyute as impy
from IPython.display import display
from scipy import stats
from statsmodels.compat import lzip
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms


# Import cleaned dataset with calculated variables

# In[ ]:


df = pd.read_csv('task1_data_clean_ready.csv')
df


# Before running the regression, let's test the assumptions first. Use scatterplot to see if the outcome variable (GovSupport) is linearly related to the predictor (NegEmot)

# In[ ]:


plt.scatter(x=df['NegEmot'], y=df['GovSupport'])
plt.xlabel('NegEmot')
plt.ylabel('GovSupport')
plt.show()


# Does not appear to have any relationship.

# In[ ]:


plt.scatter(x=df['Egalitarianism'], y=df['GovSupport'])
plt.xlabel('Egalitarianism')
plt.ylabel('GovSupport')
plt.show()


# In[ ]:


plt.scatter(x=df['Individualism'], y=df['GovSupport'])
plt.xlabel('Individualism')
plt.ylabel('GovSupport')
plt.show()


# In[ ]:


plt.scatter(x=df['AGE'], y=df['GovSupport'])
plt.xlabel('AGE')
plt.ylabel('GovSupport')
plt.show()


# Let's see what happens when we put a line of best fit for all the predictors

# In[ ]:


x = df['NegEmot']
y = df['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('NegEmot')
plt.ylabel('GovSupport')
plt.show()


x = df['Egalitarianism']
y = df['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('Egalitarianism')
plt.ylabel('GovSupport')
plt.show()

x = df['Individualism']
y = df['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('Individualism')
plt.ylabel('GovSupport')
plt.show()


x = df['AGE']
y = df['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('AGE')
plt.ylabel('GovSupport')
plt.show()

x = df['GENDER']
y = df['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('GENDER')
plt.ylabel('GovSupport')
plt.show()


# Check Pearson's correlation coefficient to see if it confirms weak correlation

# In[ ]:


stats.pearsonr(df['NegEmot'], df['GovSupport'])


# Check normality of data - skewness (<2) & kurtosis (<7) for variable distribution

# In[ ]:


df.skew(axis = 0) 


# In[ ]:


df.kurtosis(axis=0)


# Other assumptions tests done in R (homoscedasticity and normality of residuals)
# 
# Now let's run the regression

# In[ ]:


results = smf.ols('GovSupport ~ NegEmot', data=df).fit()
print(results.summary())


# ## Multiple regression

# 1. NegEmot controlling for Egalitarianism, Individualism
# 2. Add age and gender to above
# 
# Multicollinearity diagnostics: check if the three independent variables are correlated

# In[ ]:


df.corr()


# Check VIF factors

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


X = df[['NegEmot', 'Egalitarianism', 'Individualism']]
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# VIF factors look OK, all <10

# Let's run the regression, controlling for Egal and Ind

# In[ ]:


results2 = smf.ols('GovSupport ~ NegEmot+Egalitarianism+Individualism', data=df).fit()
print(results2.summary())


# Run the regression adding AGE and GENDER

# In[ ]:


results3 = smf.ols('GovSupport ~ NegEmot+Egalitarianism+Individualism+AGE+GENDER', data=df).fit()
print(results3.summary())


# Check VIF values - including age and gender

# In[ ]:


X = df[['NegEmot', 'Egalitarianism', 'Individualism', 'AGE', 'GENDER']]
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# # End

# In[ ]:




