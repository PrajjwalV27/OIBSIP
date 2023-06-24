#!/usr/bin/env python
# coding: utf-8

# ## Prajjwal Vishwakarma
# ## Task 1: Iris Flower Classification

# In[35]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# In[36]:


df=pd.read_csv("Iris.csv")


# In[37]:


lreg = LogisticRegression(random_state=0)
rf=RandomForestClassifier(random_state=0)
dc=DecisionTreeClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
sv=svm.SVC()
mlp=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()


# In[38]:


x = df.drop("Species", axis=1)
y = df["Species"]


# In[39]:


print(x)


# In[40]:


print(y)


# In[41]:


x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)


# In[42]:


lreg.fit(x_train, y_train)
y_pred=lreg.predict(x_test)
print("Logistic regresion",accuracy_score(y_test, y_pred))


# In[43]:


sv.fit(x_train, y_train)
y_pred1=sv.predict(x_test)
print("svm",accuracy_score(y_test, y_pred1))


# In[44]:


dc.fit(x_train, y_train)
y_pred2=dc.predict(x_test)
print("Decision tree",accuracy_score(y_test, y_pred2))


# In[45]:


rf.fit(x_train, y_train)
y_pred3=rf.predict(x_test)
print("Random Forest",accuracy_score(y_test, y_pred3))


# In[46]:


gb.fit(x_train, y_train)
y_pred4=gb.predict(x_test)
print("Gradient Boosting",accuracy_score(y_test, y_pred4))


# In[47]:


mlp.fit(x_train, y_train)
y_pred5=mlp.predict(x_test)
print("Nueral networks",accuracy_score(y_test, y_pred5))


# In[48]:


gnb.fit(x_train, y_train)
y_pred6=gnb.predict(x_test)
print("Guassian Naive Bayes",accuracy_score(y_test, y_pred6))


# In[49]:


mnb.fit(x_train, y_train)
y_pred7=mnb.predict(x_test)
print("Multiomial Naives bayes",accuracy_score(y_test, y_pred7))

