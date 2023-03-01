#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt

# We will use the Seaborn library
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['image.cmap'] = 'viridis'


# In[2]:


df = pd.read_csv('rfm_lesson.csv', parse_dates = ['InvoiceDate'])


# In[14]:


df


# In[20]:


df.CustomerCode.nunique()


# In[15]:


df.dtypes


# In[5]:


df.describe()


# In[12]:


# try to convert types into string
df['InvoiceNo'] = df['InvoiceNo'].apply(str)
df['CustomerCode'] = df['CustomerCode'].apply(str)


# In[16]:


df.dtypes


# In[6]:


last_date = df['InvoiceDate'].max()


# In[17]:


last_date


# In[18]:


# groupping
rfmTable = df.groupby('CustomerCode').agg({'InvoiceDate': lambda x: (last_date - x.max()).days, # Recency
                                        'InvoiceNo': lambda x: len(x), # Frequency
                                        'Amount': lambda x: x.sum()}) # Monetary Value

# переименовываем колонки так, как нам необходимо
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'Amount': 'monetary_value'}, inplace=True)


# In[19]:


rfmTable


# In[21]:


quantiles = rfmTable.quantile(q=[0.25, 0.5, 0.75])


# In[22]:


quantiles


# In[24]:


rfmSegmentation = rfmTable


# In[25]:


#For Recency
def RClass(value,parameter_name,quantiles_table):
    if value <= quantiles_table[parameter_name][0.25]:
        return 1
    elif value <= quantiles_table[parameter_name][0.50]:
        return 2
    elif value <= quantiles_table[parameter_name][0.75]: 
        return 3
    else:
        return 4

# For Frequency and Moneraty_value
def FMClass(value, parameter_name,quantiles_table):
    if value <= quantiles_table[parameter_name][0.25]:
        return 4
    elif value <= quantiles_table[parameter_name][0.50]:
        return 3
    elif value <= quantiles_table[parameter_name][0.75]: 
        return 2
    else:
        return 1
    
    

rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency',quantiles))

rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency',quantiles))

rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value',quantiles))

rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str) + rfmSegmentation.F_Quartile.map(str) + rfmSegmentation.M_Quartile.map(str)


# In[26]:


rfmSegmentation


# In[27]:


pd.crosstab(index = rfmSegmentation.R_Quartile, columns = rfmSegmentation.F_Quartile)


# In[28]:


# visualisation
rfm_table = rfmSegmentation.pivot_table(
                        index='R_Quartile', 
                        columns='F_Quartile', 
                        values='monetary_value', 
                        aggfunc=np.median).applymap(int)
sns.heatmap(rfm_table, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=4.15, annot_kws={"size": 10},yticklabels=4);


# In[53]:


# looks, how much customers in each segment
rfm_count = rfmSegmentation.groupby('RFMClass', as_index = False)                .agg({'monetary_value': 'count'})                .sort_values(by = 'monetary_value', ascending = False)


# In[54]:


rfm_count


# In[ ]:




