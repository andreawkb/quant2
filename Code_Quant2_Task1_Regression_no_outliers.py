#!/usr/bin/env python
# coding: utf-8

# ## Task 1: Regression - Analysis without outliers

# Import libraries needed

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


# Import cleaned dataset (no missing data but currently with outliers)

# In[ ]:


df = pd.read_csv('task1_clean_replaced_NaNs.csv')


# In[ ]:


df


# In[ ]:


df.describe()


# Check data distribution of items using histograms

# In[ ]:


plt.style.use('ggplot')
plt.hist(df['NegEmot1'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['NegEmot2'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['NegEmot3'], bins=7)
plt.show()


# In[ ]:


plt.style.use('ggplot')
plt.hist(df['Egal1'], bins=6)
plt.show()

plt.style.use('ggplot')
plt.hist(df['Egal2'], bins=5)
plt.show()

plt.style.use('ggplot')
plt.hist(df['Egal3'], bins=6)
plt.show()


# In[ ]:


plt.style.use('ggplot')
plt.hist(df['Ind1'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['Ind2'], bins=6)
plt.show()

plt.style.use('ggplot')
plt.hist(df['Ind3'], bins=7)
plt.show()


# In[ ]:


plt.style.use('ggplot')
plt.hist(df['GovSupport1'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['GovSupport2'], bins=6)
plt.show()

plt.style.use('ggplot')
plt.hist(df['GovSupport3'], bins=7)
plt.show()


# Data not normally distributed, but expected given data based on Likert scale...

# Use IQR/boxplots rather than z-score to visualise outliers in dataset because data not normally distributed

# In[ ]:


boxplot = df.boxplot(column=['NegEmot1', 'NegEmot2', 'NegEmot3'])
plt.grid(b=None)
plt.show()

boxplot = df.boxplot(column=['Egal1', 'Egal2', 'Egal3'])
plt.grid(b=None)
plt.show()

boxplot = df.boxplot(column=['Ind1', 'Ind2', 'Ind3'])
plt.grid(b=None)
plt.show()

boxplot = df.boxplot(column=['GovSupport1', 'GovSupport2', 'GovSupport3'])
plt.grid(b=None)
plt.show()


# Drop all rows with outliers (data points outside of IQR)

# In[ ]:


df = df.drop(df[df.NegEmot3 < 2].index)
df = df.drop(df[df.Egal1 < 4].index)
df = df.drop(df[df.Egal2 < 4].index)
df = df.drop(df[df.Egal3 < 4].index)
df = df.drop(df[df.Ind1 > 4].index)
df = df.drop(df[df.GovSupport1 < 2].index)


# In[ ]:


df


# 172 cases remaining (out of the original 212).
# 
# Let's look at the summary.

# In[ ]:


df.describe()


# Combine subscales before primary analysis

# In[ ]:


df['NegEmot'] = df[['NegEmot1', 'NegEmot2', 'NegEmot3']].mean(axis=1)
df['Egalitarianism'] = df[['Egal1', 'Egal2', 'Egal3']].mean(axis=1)
df['Individualism'] = df[['Ind1', 'Ind2', 'Ind3']].mean(axis=1)
df['GovSupport'] = df[['GovSupport1', 'GovSupport2', 'GovSupport3']].mean(axis=1)

df


# Variables now calculated and combined, drop original subscales so that new dataframe only contains calculated variables

# In[ ]:


df1 = df.iloc[:,[1, 2, 3, 16, 17, 18, 19]]
df1


# Save a csv copy

# In[ ]:


#df1.to_csv('task1_data_clean_no_outliers.ready.csv')


# ## Primary Analysis (without outliers)

# Plot a scatterplot to visualise relationship between NegEmot and GovSupport

# In[ ]:


plt.scatter(x=df1['NegEmot'], y=df1['GovSupport'])
plt.xlabel('NegEmot')
plt.ylabel('GovSupport')
plt.show()


# Let's plot a line of best fit and see

# In[ ]:


x = df1['NegEmot']
y = df1['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('NegEmot')
plt.ylabel('GovSupport')
plt.show()


# Run correlation to see if the two are correlated

# In[ ]:


stats.pearsonr(df1['NegEmot'], df1['GovSupport'])


# Check if x variables are correlated (multicollinearity)

# In[ ]:


df1.corr()


# Not really. Let's test the assumptions in R and then run the regression models

# ## Model 1 - NegEmot and GovSupport (no outliers)

# In[ ]:


results = smf.ols('GovSupport ~ NegEmot', data=df1).fit()
print(results.summary())


# ## Model 2 - Control for Egal and Ind (no outliers)

# Check VIF first

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


X = df1[['NegEmot', 'Egalitarianism', 'Individualism']]
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# Run regression model 2

# In[ ]:


results2 = smf.ols('GovSupport ~ NegEmot+Egalitarianism+Individualism', data=df1).fit()
print(results2.summary())


# ## Model 3 - all x variables included (no outliers)

# Check VIF

# In[ ]:


X = df1[['NegEmot', 'Egalitarianism', 'Individualism', 'AGE', 'GENDER']]
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# In[ ]:


results3 = smf.ols('GovSupport ~ NegEmot+Egalitarianism+Individualism+AGE+GENDER', data=df1).fit()
print(results3.summary())


# Out of interest let's plot GovSupport against Egal and see what happens

# In[ ]:


x = df1['Egalitarianism']
y = df1['GovSupport']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('Egalitarianism')
plt.ylabel('GovSupport')
plt.show()


# In[ ]:




