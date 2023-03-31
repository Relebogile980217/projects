#!/usr/bin/env python
# coding: utf-8

# In[23]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# Importing Necessary Libraries

# In[24]:


import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


# <h1>About the Dataset</h1>
# <p> 
# This data contains a set of patients, all whom suffered from the same illness. During their treatment, each patient responded to one of 5 medications, Drug A, Drug B Drug C, Drug X and Drug Y
# </p>
# 
# <br>
#  
# <p>
# Our job is to build a model to find out which drug might appropriate for future patient with the same illness. The Features of this dataset are Age, Sex, Blood Pressure, and the Cholestrol of the patients, and the target is the drug that each patient responded to.
# </p>
# 
# <br>
# 
# <p>
# We should build a Decision Tree and use it to predict the class of an unknown patient, and to prescribe the right drug    
# </p>
# 

# <h3>Collecting the Data</h3>

# In[25]:


my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=',')
my_data.head()


# <h4> The size of the data</h4>

# In[56]:


my_data.shape


# In[57]:


X = my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values


# <p>We should convert categorical data to numerical data on the Sex, BP, Cholestorol and Drug</p>

# In[58]:


from sklearn import preprocessing

##converting sex column

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

## Converting BP Column
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

## converting Cholestrol Column
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3])


# Target Variable

# <h3>Setting up the decision Tree</h3>

# In[66]:


y = my_data['Drug']


# In[68]:


y.value_counts()


# In[63]:


from sklearn.model_selection import train_test_split


# In[69]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[75]:


print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))


# In[77]:


print('Shape of X test set {}'.format(X_testset.shape), '&', 'Size of Y test set {}'.format(y_testset.shape))


# <h2>Building the Model</h2>

# In[79]:


drugTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
drugTree


# In[80]:


drugTree.fit(X_trainset, y_trainset)


# <h3>Prediction</h3>

# In[83]:


predTree = drugTree.predict(X_testset)


# In[88]:


print(predTree[0:5])
print(y_testset[0:5])


# In[89]:


from sklearn import metrics 
import matplotlib.pyplot as plt 
print('DecisionTrees Accuracy:', metrics.accuracy_score(y_testset, predTree))


# <h2>Visualization</h2>

# In[92]:


get_ipython().system('conda install -c conda-forge pydotplus -y')
get_ipython().system('conda install -c conda-forge python-graphviz -y')


# In[93]:


tree.plot_tree(drugTree)
plt.show()


# In[ ]:




