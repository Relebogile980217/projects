#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install scikit-learn==0.23.1')


# In[56]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading data

# In[7]:


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head(10)


# Checking how many of each class is in our data

# In[8]:


df['custcat'].value_counts()


# 281 Plus service , 266 basic-service, 236 total service, and 217 E-service customer  

# In[10]:


df.hist(column = 'income', bins = 50)
df.hist(column = 'age', bins = 50)


# <h2>lest define  our feature sets, X:</h2>

# In[12]:


df.columns


# For us to use scikit-learn we have to change the dataframe to a numpy array

# In[13]:


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  #.astype(float)
X[0:10]


# In[37]:


y = df['custcat'].values
y[0:10]


# Normaize data

# In[38]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train_set:', X_train.shape, y_train.shape)
print('Test_set:', X_test.shape, y_test.shape)


# <b>Classification</b>
# <b> K Nearest neighbor (KNN)</b>

# In[40]:


from sklearn.neighbors import KNeighborsClassifier


# <b>Training the model</b>

# In[41]:


k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
neigh


# <b>Predicting</b>

# In[42]:


yhat = neigh.predict(X_test)
yhat[0:10]


# <b>Accuracy Evaluation</b>

# In[45]:


from sklearn import metrics
print('Train set Accuracy:', metrics.accuracy_score(y_train, neigh.predict(X_train)))
print('Test set Accuracy:', metrics.accuracy_score(y_test, yhat))


# <h2>Practice</h2>

# <b>Building KNN model where k=6 </b>

# In[46]:


k = 6
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh


# <b>Predicting</b>

# In[48]:


yhat = neigh.predict(X_test)
yhat[0:5]


# <b>Accuracy Evaluation</b>

# In[51]:


print('Train set Accuracy:', metrics.accuracy_score(y_train, neigh.predict(X_train)))
print('Test set Accuracy:', metrics.accuracy_score(y_test, yhat))


# Calculating the accuracy of KNN for different values of K

# In[54]:


Ks  = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat  = neigh.predict(X_test)
    
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
    
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    
mean_acc


# Plotting the model accuracy for a different number of neighbors.

# In[63]:


plt.plot(range(1,Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color='green')
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[66]:


print('The best accuracy was with', mean_acc.max(), 'with k =', mean_acc.argmax()+1)


# In[ ]:




