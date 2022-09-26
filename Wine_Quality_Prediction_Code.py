#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df = pd.read_csv("C:\\Users\\sudhi\\Downloads\\winequality-red.csv",sep=";")
df


# In[12]:


df.shape


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[16]:


#Check for duplicate values
df.duplicated().sum()


# In[19]:


df.drop_duplicates(inplace=True)
df


# In[21]:


df.dtypes


# In[22]:


df['quality'].value_counts()


# In[117]:


sns.countplot(df['quality'])


# In[125]:


sns.displot(x=df['quality'],y=df['pH'])


# In[123]:


sns.violinplot(x=df['quality'],y=df['pH'])


# In[124]:


sns.stripplot(x=df['quality'],y=df['pH'])


# In[129]:


sns.stripplot(x=df['quality'],y=df['alcohol'])


# In[130]:


sns.stripplot(x=df['quality'],y=df['fixed acidity'])


# In[127]:


sns.violinplot(x=df['quality'],y=df['density'])


# In[23]:


x = df.drop('quality',axis=1)
y = df['quality']


# In[24]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
print("Train Size = ",x_train.shape)
print("Test Size = ",y_test.shape)


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[34]:


from sklearn.metrics import confusion_matrix,classification_report


# In[38]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[47]:


def model_gen(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    print("Training Score",model.score(x_train,y_train))
    print("Testing Score",model.score(x_test,y_test))
    y_pred=model.predict(x_test)
    print(y_pred)
    cm =confusion_matrix(y_test,y_pred)
    print("\nConfusion Matrix",cm)
    print("\nClassification Report",classification_report(y_test,y_pred))


# In[48]:


def mscore(model):
    print("Training Score",model.score(x_train,y_train))
    print("Testing Score",model.score(x_test,y_test))


# In[70]:


def eval_metrics(y_test,y_pred):
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2s = r2_score(y_test,y_pred)
    print("MSE",mse)
    print("MAE",mae)
    print("RMSE",rmse)
    print("R2_Score",r2s)


# In[71]:


m1 = LogisticRegression()


# In[73]:


m1.fit(x_train,y_train)


# In[74]:


y_pred = m1.predict(x_test) 
y_pred


# In[75]:


mscore(m1)


# In[76]:


model_gen(m1,x_train,x_test,y_train,y_test)


# In[77]:


eval_metrics(y_test,y_pred)


# In[110]:


m2 = DecisionTreeClassifier(criterion='gini')


# In[111]:


m2.fit(x_train,y_train)


# In[112]:


y_pred = m2.predict(x_test)
y_pred


# In[113]:


mscore(m2)


# In[114]:


model_gen(m2,x_train,x_test,y_train,y_test)


# In[115]:


eval_metrics(y_test,y_pred)


# In[116]:


m3 = RandomForestClassifier()


# In[88]:


m3.fit(x_train,y_train)


# In[89]:


y_pred = m3.predict(x_test)


# In[90]:


mscore(m3)


# In[91]:


model_gen(m3,x_train,x_test,y_train,y_test)


# In[92]:


eval_metrics(y_test,y_pred)


# In[106]:


m4 = KNeighborsClassifier(n_neighbors=5)


# In[107]:


m4.fit(x_train,y_train)


# In[108]:


y_pred = m4.predict(x_test)


# In[109]:


mscore(m4)


# In[98]:


model_gen(m4,x_train,x_test,y_train,y_test)


# In[99]:


eval_metrics(y_test,y_pred)


# In[100]:


m5 = SVC()


# In[101]:


m5.fit(x_train,y_train)


# In[102]:


y_pred = m5.predict(x_test)


# In[103]:


mscore(m5)


# In[104]:


model_gen(m5,x_train,x_test,y_train,y_test)


# In[105]:


eval_metrics(y_test,y_pred)


# In[ ]:




