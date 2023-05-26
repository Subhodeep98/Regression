#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('beer.csv')
df


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(df[['ALC']], df[['COST']], test_size=0.3, random_state=42)


# In[5]:


model = LinearRegression().fit(X_train, y_train)


# In[6]:


print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)


# In[7]:


y_pred = model.predict(X_test)


# In[8]:


r_squared = model.score(X_test, y_test)
print('R-squared: ', r_squared)


# In[ ]:




