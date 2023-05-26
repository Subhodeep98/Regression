#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd


# In[38]:


data=pd.read_csv('HR_DATA.csv')


# In[39]:


data.head()


# In[40]:


data.info()


# In[41]:


data.columns


# In[42]:


data.set_index('Candidate Ref',inplace=True)


# In[43]:


data.columns


# In[44]:


data.shape


# In[45]:


data.isnull().sum()


# In[46]:


X=data.columns


# In[47]:


Y=data.Status


# In[48]:


X=data.drop('Status',axis='columns')


# In[49]:


encoded_data=pd.get_dummies(X)


# In[50]:


Y=data.Status.map(lambda x:int(x=="Joined"))


# In[51]:


Y


# In[52]:


Y.nunique()


# In[53]:


Y.value_counts()


# In[54]:


X


# In[55]:


encoded_data


# In[56]:


from sklearn.preprocessing import StandardScaler


# In[57]:


Scaler=StandardScaler()


# In[58]:


import statsmodels.api as sm


# In[59]:


X_n=sm.add_constant(encoded_data)


# In[60]:


X_scaled=Scaler.fit_transform(X_n)


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y,test_size=0.3,random_state=42)


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


logmodel=LogisticRegression()


# In[65]:


logmodel.fit(X_train,Y_train)


# In[66]:


Y_pred=logmodel.predict(X_test)


# In[67]:


Y_pred


# In[68]:


from sklearn.metrics import classification_report


# In[69]:


classification_report(Y_test,Y_pred)


# In[70]:


from sklearn.metrics import confusion_matrix


# In[71]:


confusion_matrix(Y_test,Y_pred)


# In[72]:


from sklearn.metrics import r2_score


# In[73]:


from sklearn.metrics import roc_auc_score


# In[74]:


roc_auc_score(Y_test,Y_pred)


# In[75]:


logmodel.score(X_test,Y_test)


# In[ ]:





# In[ ]:




