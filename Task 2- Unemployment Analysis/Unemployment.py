#!/usr/bin/env python
# coding: utf-8

# # Prajjwal Vishwakarma
# ## Task 2: Unemployment Analysis 

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[28]:


df = pd.read_csv("Unemployment in India.csv")


# In[29]:


df.head()


# In[30]:


df = df.dropna()


# In[31]:


df.info


# In[32]:


df[' Date'] = pd.to_datetime(df[' Date'])


# In[33]:


plt.figure(figsize=(15,5))
plt.bar(df[' Date'], df[' Estimated Unemployment Rate (%)'])
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate in India')
plt.show()


# In[34]:


plt.figure(figsize=(15,5))
sns.lineplot(x=' Date',y = ' Estimated Unemployment Rate (%)',data=df)


# In[35]:


mean_unemployment = df[' Estimated Unemployment Rate (%)'].mean()
median_unemployment = df[' Estimated Unemployment Rate (%)'].median()
print('Mean Unemployment Rate: ',{mean_unemployment})
print('Median Unemployment Rate:' ,{median_unemployment})


# In[36]:


df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")


# In[37]:


df2.head()


# In[38]:


df2.info()


# In[39]:


plt.figure(figsize=(15,5))
plt.bar(df2[' Date'],df2[' Estimated Unemployment Rate (%)'])
plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.title("Unemployment rate Upto 11-2020")
plt.show()


# In[40]:


plt.figure(figsize=(15,5))
sns.lineplot(x=' Date',y = ' Estimated Unemployment Rate (%)',data=df2)


# In[41]:


mean_unemployment = df2[' Estimated Unemployment Rate (%)'].mean()
median_unemployment = df2[' Estimated Unemployment Rate (%)'].median()
print('Mean Unemployment Rate during Covid-19:', {mean_unemployment})
print('Median Unemployment Rate during Covid-19: ',{median_unemployment})

