#!/usr/bin/env python
# coding: utf-8

# ## Task 2 - Analyses without outliers

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


# In[ ]:


df = pd.read_csv("Task2_data_clean.csv")


# In[ ]:


df


# In[ ]:


plt.style.use('ggplot')
plt.hist(df['AutSup1'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['AutSup2'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['AutSup3'], bins=7)
plt.show()


# In[ ]:


plt.style.use('ggplot')
plt.hist(df['IM1'], bins=6)
plt.show()

plt.style.use('ggplot')
plt.hist(df['IM2'], bins=5)
plt.show()

plt.style.use('ggplot')
plt.hist(df['IM3'], bins=6)
plt.show()


# In[ ]:


plt.style.use('ggplot')
plt.hist(df['Energy1'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['Energy2'], bins=7)
plt.show()

plt.style.use('ggplot')
plt.hist(df['Energy3'], bins=7)
plt.show()


# Data not normally distributed. Use boxplots to find outliers

# In[ ]:


boxplot = df.boxplot(column=['AutSup1', 'AutSup2', 'AutSup3'])
plt.grid(b=None)
plt.show()

boxplot = df.boxplot(column=['IM1', 'IM2', 'IM3'])
plt.grid(b=None)
plt.show()

boxplot = df.boxplot(column=['Energy1', 'Energy2', 'Energy3'])
plt.grid(b=None)
plt.show()


# Drop rows with outliers

# In[ ]:


df = df.drop(df[df.AutSup3 < 2].index)
df = df.drop(df[df.IM1 < 4].index)
df = df.drop(df[df.IM2 < 4].index)
df = df.drop(df[df.IM3 < 4].index)
df = df.drop(df[df.Energy1 < 2].index)
df = df.drop(df[df.Energy2 < 4.5].index)


# In[ ]:


df


# 167 cases remaining (out of original 213). Let's look at the summary

# In[ ]:


df.describe()


# Combine subscales before primary analyses

# In[ ]:


df['AutSup'] = df[['AutSup1', 'AutSup2', 'AutSup3']].mean(axis=1)
df['IntMot'] = df[['IM1', 'IM2', 'IM3']].mean(axis=1)
df['Energy'] = df[['Energy1', 'Energy2', 'Energy3']].mean(axis=1)

df


# Drop original subscales to leave only combined variables

# In[ ]:


df1 = df.iloc[:,[1, 2, 3, 13, 14, 15]]
df1


# Save as csv

# In[ ]:


#df1.to_csv("Task2_data_clean_no_outliers_ready.csv")


# ## Regressios (without outliers) - Mediation model in R

# In[ ]:


plt.scatter(x=df1['AutSup'], y=df1['Energy'])
plt.xlabel('Autonomy Support')
plt.ylabel('Energy')
plt.show()


# C path, total effect

# In[ ]:


results = smf.ols('Energy ~ AutSup', data=df1).fit()
print(results.summary())


# C' path (indirect effect)

# In[ ]:


results2 = smf.ols('Energy ~ AutSup+IntMot', data=df1).fit()
print(results2.summary())


# a path

# In[ ]:


results3 = smf.ols('IntMot ~ AutSup', data=df1).fit()
print(results3.summary())


# b path

# In[ ]:


results4 = smf.ols('Energy ~ IntMot', data=df1).fit()
print(results4.summary())


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


X = df1[['AutSup', 'IntMot', 'Energy']]
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# In[ ]:




