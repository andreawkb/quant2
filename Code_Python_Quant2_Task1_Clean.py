#!/usr/bin/env python
# coding: utf-8

# # Task 1 - Data screening and cleaning

# Install libraries needed

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


# Import data by importing the csv file as pandas dataframe.
# Raw data has already been visually inspected and any coding errors rectified.
# Incorrectly coded values (cases 134, 216 and 222) have been updated (decimals removed and duplication removed).
# Import 'T1.Raw data_v2.csv'

# In[ ]:


df = pd.read_csv('T1.Raw data_v2.csv')


# I want to see all rows and columns displayed to visually inspect the data

# In[ ]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None
display(df)


# Visually inspect data. N=212. Some age values have been replaced by the mean. Some cases seem to have been removed? 
# 999 denotes missing value.
# Let's look at a summary of the dataset.

# In[ ]:


df.describe()


# Python not recognising 999 as missing values. This is problematic - descriptive data is skewed as a result (i.e. mean, max, etc.).
# Need to replace 999 with NaN.
# Let's check if there are any duplicates first.

# In[ ]:


df.duplicated()


# No duplication. Replace 999 with NaN.
# 
# df = missing data still displayed as 999
# df2 = missing data replaced with NaN

# In[ ]:


df2 = df.replace (999, np.nan)
df2


# Missing data found in variables Egal1, Egal2, Ind2, Ind3, GovSupport2.
# 
# Hmm, variables with missing data have been converted to floats. Let's check the data types.

# In[ ]:


print(df2.dtypes)


# Let's check summary of data again. Counts should exclude missing data. Mean, max etc. should also be fixed.

# In[ ]:


df2.describe()


# Let's find out how many missing values there are per variable/column.

# In[ ]:


df2.isnull().sum(axis=0)


# Let's find out where the missing data are exactly

# In[ ]:


is_NaN = df2.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df2[row_has_NaN]
print(rows_with_NaN)


# Ok, rows with missing data = 3,6,14,17,24,31,38,168.
# Total number of missing values is quite low = 8.
# Let's double check this.

# In[ ]:


df2.isnull().sum().sum()


# 8 missing values; but out of how many?

# In[ ]:


df2.count().sum()


# 3172+8=3180 (total values in dataset). 8/3180*100=0.25%.
# 
# Overall missingness is 0.25%, so very low (<1%)
# 
# Need to find out rates of missingness. Let's find out the rates of missingness per variable/column first.
# 

# In[ ]:


#df2_missing.dtypes

df2.isnull().dtypes


# Find out total number of missing values per variable using sum

# In[ ]:


df2_num_missing = df2.isnull().sum(axis=0)
df2_num_missing


# I can now see total number of missing values per variable. Double check number of cases in total so that we can calculate rate of missingness.

# In[ ]:


len(df2)


# Divide each value by number of cases to calculate percentage of missingness for each variable.

# In[ ]:


df2_num_missing_variable = (df2_num_missing / len(df2))*100
df2_num_missing_variable


# Range of missingness for variables = 0.47% to 1.41%. 
# 
# Now calculate mean (divided by total number of variables)
# 

# Sum divided by number of variables with missing values = mean

# In[ ]:


df2_num_missing_variable.sum()/5


# Now calculate rates of missingness for cases. First bring up number of missing values for rows (cases).

# In[ ]:


df2.isnull().sum(axis=1)


# Divide number of missing values by number of variables for rate of missingness

# In[ ]:


df2_num_missing_case = df2.isnull().sum(axis=1)
df2_num_missing_case_all = (df2_num_missing_case/len(df2.columns))*100
df2_num_missing_case_all


# Looks like there is only 1 missing value per case max.
# 
# Calculate mean of rates of missingness for cases.
# 

# In[ ]:


df2_num_missing_case_all.sum()/8


# Need to do Little's MCAR test in R as not available in Python.
# 
# In R, littleMCAR test result: p value = 0.057, borderline significant. Rounded to 0.06, so > 0.05; null hypothesis cannot be rejected. Missing data therefore missing completely at random.
# 
# Replace missing values with mean of participant's relevant subscales in dataset (v2).
# Export dataset to csv to replace NaNs with participant's mean.
# 
# 

# In[ ]:


#df2.to_csv('task1_clean_without_NaNs.csv')


# NaNs substituted with participants' mean calculated from relevant subscale

# In[ ]:


df3 = pd.read_csv('task1_clean_replaced_NaNs.csv')
df3


# Double check there are definitely no more missing values.

# In[ ]:


df3.isnull().sum(axis=1)


# Check for outliers using boxplots

# In[ ]:


boxplot = df3.boxplot(column=['NegEmot1', 'NegEmot2', 'NegEmot3'])
plt.grid(b=None)
plt.show()

boxplot = df3.boxplot(column=['Egal1', 'Egal2', 'Egal3'])
plt.grid(b=None)
plt.show()

boxplot = df3.boxplot(column=['Ind1', 'Ind2', 'Ind3'])
plt.grid(b=None)
plt.show()

boxplot = df3.boxplot(column=['GovSupport1', 'GovSupport2', 'GovSupport3'])
plt.grid(b=None)
plt.show()


# Some outliers are present, none for NegEmot1,2,3; few for GovSupport1. 
# 
# Let's run the analysis with and without outliers and see what happens.

# Data cleaning and prep now complete.
# 
# Now combine subscales before analysis.

# In[ ]:


df3['NegEmot'] = df3[['NegEmot1', 'NegEmot2', 'NegEmot3']].mean(axis=1)
df3['Egalitarianism'] = df3[['Egal1', 'Egal2', 'Egal3']].mean(axis=1)
df3['Individualism'] = df3[['Ind1', 'Ind2', 'Ind3']].mean(axis=1)
df3['GovSupport'] = df3[['GovSupport1', 'GovSupport2', 'GovSupport3']].mean(axis=1)


df3


# Variables now combined. Drop subscales so that new dataframe only contains calculated variables.
# 
# Calculated variables [NegEmot, Egalitarianism, Individualism, GovSupport].

# In[ ]:


df4 = df3.iloc[:,[1, 2, 3, 16, 17, 18, 19]]
df4


# Export clean and combined data as csv file ready for primary analysis (with outliers)

# In[ ]:


#df4.to_csv('task1_data_clean_ready.csv')


# Data cleaning and prep complete. 

# In[ ]:




