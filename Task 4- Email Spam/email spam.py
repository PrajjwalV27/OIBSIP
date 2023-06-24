#!/usr/bin/env python
# coding: utf-8

# ## Prajjwal Vishwakarma
# ## Task 4: Email Spam Detection With Machine Learning

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("spam.csv")


# In[3]:


print(df)


# In[4]:


mail_data = df.where((pd.notnull(df)),'')


# In[5]:


mail_data.head()


# In[6]:


mail_data.shape


# In[7]:


mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# In[8]:


X = mail_data['Message']
Y = mail_data['Category']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[12]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[13]:


print(X_train)


# In[14]:


print(X_train_features)


# In[15]:


model = LogisticRegression()


# In[16]:


model.fit(X_train_features, Y_train)


# In[17]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[18]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[19]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[20]:


print('Accuracy on test data : ', accuracy_on_test_data)


# In[21]:


input_mail = ["Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed Â£1000 cash or Â£5000 prize!"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')

