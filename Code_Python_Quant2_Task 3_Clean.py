#!/usr/bin/env python
# coding: utf-8

# ## Task 3 - CFA (Data screening and cleaning)

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


# Import raw data from csv file and create dataframe

# In[ ]:


df = pd.read_csv('0.Raw data.csv')


# In[ ]:


df


# ## Data screening and cleaning

# Let's visually inspect the data
# I want to see all rows

# In[ ]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None
display(df)


# Check for duplications

# In[ ]:


df.duplicated()


# Replace missing data with NaN

# In[ ]:


df2 = df.replace (999, np.nan)
df2


# Let's see a summary of the data

# In[ ]:


df2.describe()


# Let's check which columns contain missing values

# In[ ]:


df2.isnull().sum(axis=0)


# Let's see which rows contain missing values

# In[ ]:


is_NaN = df2.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df2[row_has_NaN]
print(rows_with_NaN)


# Check total number of missing valuesn in dataset

# In[ ]:


df2.isnull().sum().sum()


# Check total number of values in dataset

# In[ ]:


df2.count().sum()


# Total number of values in dataset should have been: 2773+9 = 2782
# (9/2782)*100

# In[ ]:


(9/2782)*100


# 0.32% of overall missingness. 
# Let's find out rates of missingness per variable and cases.

# Rates of missingness per VARIABLE

# In[ ]:


df2_num_missing = df2.isnull().sum(axis=0)
df2_num_missing


# In[ ]:


len(df2)


# Divide each value by number of cases to calculate percentage of missingness for each variable.

# In[ ]:


df2_num_missing_variable = (df2_num_missing / len(df2))*100
df2_num_missing_variable


# Calculate mean rate of missingness per variable

# In[ ]:


df2_num_missing_variable.sum()/7


# Calculate rates of missingness for CASES

# In[ ]:


df2.isnull().sum(axis=1)


# Divide number of missing values by number of variables for rate of missingness

# In[ ]:


df2_num_missing_case = df2.isnull().sum(axis=1)
df2_num_missing_case_all = (df2_num_missing_case/len(df2.columns))*100
df2_num_missing_case_all


# Most cases only have one missing value, except row 11 (number/case 30).
# 
# Let's find out how many rows contain missing values

# In[ ]:


sum([True for idx,row in df2.iterrows() if any(row.isnull())])


# Calculate the mean rate of missingness per cases

# In[ ]:


df2_num_missing_case_all.sum()/6


# A bit high, but that's down to case 30 (which will be deleted list-wise)

# Now check MCAR in R

# In[ ]:


#df2.to_csv('Task3_data_NaN.csv')


# MCAR p=0.659 MCAR inferred.
# 
# Use person mean substitution (from subscale) to replace NaNs, except case 30 (list-wise delete)

# In[ ]:


df3 = df2.drop([11])
df3


# Case 30 removed from dataset.
# 
# Now manually replace missing values with subscale mean in csv file (don't know how to do this in python yet)

# In[ ]:


#df3.to_csv('Task3_data_NaN_2.csv')


# Import new csv file with missing data replaced

# In[ ]:


df4 = pd.read_csv('Task3_data_clean.csv')


# In[ ]:


df4


# Check for missing values, shouldn't be any

# In[ ]:


df4.isnull().sum(axis=1)


# Check summary

# In[ ]:


df4.describe()


# N=213 (one case removed)

# ## Outliers

# Use boxplots to check for outliers

# In[ ]:


boxplot = df4.boxplot(column=['PER1', 'PER2', 'PER3'])
plt.grid(b=None)
plt.show()

boxplot = df4.boxplot(column=['RUM1', 'RUM2', 'RUM3'])
plt.grid(b=None)
plt.show()

boxplot = df4.boxplot(column=['EX1', 'EX2', 'EX3', 'EX4'])
plt.grid(b=None)
plt.show()


# Carry on with analysis and run analysis without outliers later to compare results

# Move over to R to do SEM

# In[ ]:


df4.dtypes


# In[ ]:


df4


# In[ ]:




