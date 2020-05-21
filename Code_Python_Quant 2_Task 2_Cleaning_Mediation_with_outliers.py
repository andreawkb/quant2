#!/usr/bin/env python
# coding: utf-8

# ## Task 2 - Preliminary and Primary Analyses (Mediation) with outliers

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


# Import data from csv file

# In[ ]:


df = pd.read_csv('Task2_raw_data.csv')


# See all rows of dataset

# In[ ]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None
display(df)


# Check for duplication

# In[ ]:


df.duplicated()


# Replace 999 with NaN so that Python recognises it is a missing value

# In[ ]:


df2 = df.replace (999, np.nan)
df2


# Now let's take a look at the summary of the data

# In[ ]:


df2.describe()


# Missing values found in 'IM1' 'IM2' and all 3 Energy sub-scales. Why are there so many missing for Energy?
# 
# Just to confirm number of missing values per sub-scale

# In[ ]:


df2.isnull().sum(axis=0)


# In[ ]:


is_NaN = df2.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df2[row_has_NaN]
print(rows_with_NaN)


# Check total number of missing values

# In[ ]:


df2.isnull().sum().sum()


# In[ ]:


df2.count().sum()


# 41 values missing out of 2568 datapoints.
# (41/2568)*100 = 1.6% of overall missingness.
# 
# Let's find out rates of missingness per variable and per cases.

# In[ ]:


df2_num_missing = df2.isnull().sum(axis=0)
df2_num_missing


# Double check number of cases in total so that we can calculate rate of missingness.

# In[ ]:


len(df2)


# Divide each value by number of cases to calculate percentage of missingness for each variable.

# In[ ]:


df2_num_missing_variable = (df2_num_missing / len(df2))*100
df2_num_missing_variable


# Calculate mean (rate of missingness per variable)

# In[ ]:


df2_num_missing_variable.sum()/5


# Rows with missing values

# In[ ]:


df2.isnull().sum(axis=1)


# Divide number of missing values by number of variables for rate of missingness

# In[ ]:


df2_num_missing_case = df2.isnull().sum(axis=1)
df2_num_missing_case_all = (df2_num_missing_case/len(df2.columns))*100
df2_num_missing_case_all


# Most cases only have one missing value, except case 11. Let's find out how many rows have missing values.

# In[ ]:


sum([True for idx,row in df2.iterrows() if any(row.isnull())])


# Let's calculate the mean (rate of missingess per cases)

# In[ ]:


df2_num_missing_case_all.sum()/39


# Need to do MCAR in R.

# In[ ]:


#df2.to_csv('Task2_data_NaN.csv')


# MCAR p=0.224. MCAR inferred.
# 
# Use mean substitution (with mean of completed relevant sub-scales). 
# Except case 30 - list-wise delete as all Energy subscales missing, cannot calculate mean.

# In[ ]:


df3 = df2.drop([11])
df3


# Now replace missing values with subscale mean in csv file

# In[ ]:


#df3.to_csv('Task2_data_NaN_2.csv')


# In[ ]:


df4 = pd.read_csv('Task2_data_clean.csv')


# In[ ]:


df4


# Check there are no missing values in cleaned dataset

# In[ ]:


df4.isnull().sum(axis=1)


# Check outliers using boxplots

# In[ ]:


boxplot = df4.boxplot(column=['AutSup1', 'AutSup2', 'AutSup3'])
plt.grid(b=None)
plt.show()

boxplot = df4.boxplot(column=['IM1', 'IM2', 'IM3'])
plt.grid(b=None)
plt.show()

boxplot = df4.boxplot(column=['Energy1', 'Energy2', 'Energy3'])
plt.grid(b=None)
plt.show()


# Drop rows with outliers and run analyses in a separate notebook. For now carry on with outliers.
# 
# Calculate scales to combine

# In[ ]:


df4['AutSup'] = df4[['AutSup1', 'AutSup2', 'AutSup3']].mean(axis=1)
df4['IntMot'] = df4[['IM1', 'IM2', 'IM3']].mean(axis=1)
df4['Energy'] = df4[['Energy1', 'Energy2', 'Energy3']].mean(axis=1)

df4


# Drop original subscales so that only combined and calculated scales remained

# In[ ]:


df5 = df4.iloc[:,[1, 2, 3, 13, 14, 15]]
df5


# Export clean and combined data as csv file ready for primary analysis

# In[ ]:


#df5.to_csv('Task2_data_clean_withOutliers_ready.csv')


# ## Check regressions (not primary analyses - mediation model in R)

# In[ ]:


df5.describe()


# Let's take a look at the relationship between AutSup and Energy using a simple scatterplot

# In[ ]:


x = df5['AutSup']
y = df5['Energy']
m, b = np.polyfit(x,y,1)
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.xlabel('Autonomy Support')
plt.ylabel('Energy')
plt.show()


# Check Pearson's correlation coefficient

# In[ ]:


stats.pearsonr(df5['AutSup'], df5['Energy'])


# Check normality of data: skewness (<2) and kurtosis (<7)

# In[ ]:


df5.skew(axis = 0) 


# In[ ]:


df5.kurtosis(axis=0)


# Now test other assumptions in R before running regressions here. Mediation model will be tested in R as well!

# Now let's check total effect (c path)

# In[ ]:


results = smf.ols('Energy ~ AutSup', data=df5).fit()
print(results.summary())


# Control for intrinsic motivation (c' path)

# In[ ]:


results2 = smf.ols('Energy ~ AutSup+IntMot', data=df5).fit()
print(results2.summary())


# a path

# In[ ]:


results3 = smf.ols('IntMot ~ AutSup', data=df5).fit()
print(results3.summary())


# b path

# In[ ]:


results4 = smf.ols('Energy ~ IntMot', data=df5).fit()
print(results4.summary())


# Check VIF values

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


X = df5[['AutSup', 'IntMot', 'Energy']]
X['Intercept'] = 1

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)


# In[ ]:




