#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt

# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
df = pd.read_csv(
    "C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\Breast-cancer.csv")
# IPython.display allows nice output formatting within the Jupyter notebook
display(df.head())


# In[22]:


#RUN ONLY ONCE:

#Need to filter out the data before entering into train test split:

#Adjust accordingly.

#This pops out id column in df.
df.pop('id')
y = df.pop('diagnosis')
X = df


# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# #### Analyzing KNeighborsClassifier
# 

# In[27]:


#Applying K-Nearest-Neighbors:

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

