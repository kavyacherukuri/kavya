#!/usr/bin/env python
# coding: utf-8

# # 
# ## [ k-Nearest Neighborhood]
# 
# 

# ## Import Libraries
# * See various conventions and acronyms.

# In[57]:


import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.datasets import load_breast_cancer


# # Load the data
# *   Verify the Python type for the dataset.

# In[58]:


CancerDataset = load_breast_cancer()
print(type(CancerDataset))
print(CancerDataset.keys())
#print(CancerDataset.target)


# ## Verify basic data statistics
# * Count the number of features. (i.e., attributes)
# * Count the number of examples. (i.e., instances and labels)
# * Print out the description of each feature.

# In[59]:


def printBasicStats(dataset):
  print(dataset['feature_names'], dataset['target_names'])
  print(len(dataset['feature_names']), type(dataset['feature_names']))  
  print(dataset['data'].shape, dataset['target'].shape)
  print(dataset['DESCR'])

printBasicStats(CancerDataset)


# ## Convert the dataset to a DataFrame
# *   Not necessarily useful. (scikit-learn works well with default libraries such as list, numpy array, and scipy's sparse matrix)
# *   But using pandas provides more intuitive excel or R-like views.

# In[60]:


def getDataFrame(dataset):
  numData = dataset['target'].shape[0]
  newDataset = np.concatenate((dataset['data'], dataset['target'].reshape(numData, -1)), axis=1)
  print(newDataset)
  newNames = np.append(dataset['feature_names'], ['target'])
  return pd.DataFrame(newDataset, columns=newNames)

DataFrame = getDataFrame(CancerDataset)
#print(DataFrame)
  


# ## Inspect label distribution
# *   Check the target label distribution/imbalance.
# 

# In[61]:


def printLabelDist(df, dataset):
  counts = df.target.value_counts(ascending=True)
  print(counts)
  counts.index = dataset['target_names']
  print(counts)  

printLabelDist(DataFrame, CancerDataset)


# ## Data split
# * Split the data into training and test sets.
# * No validation for now.
# 
# 

# In[62]:


from sklearn.model_selection import train_test_split
def splitData(df, size):
  X, y = df[df.columns[:-1]], df.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=X.shape[0] - size, random_state=0)
  return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = splitData(DataFrame, 400)
assert X_train.shape == (400, 30)
assert y_train.shape == (400, )


# # Training
# *   Train a k-NN model on the training data.
# *   Get the training accuracy.
# 
# 

# In[63]:


from sklearn.neighbors import KNeighborsClassifier
def trainKnn(X, y, k=1):
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(X, y)
  pred = model.predict(X)
  accuracy = sum(pred == y) / len(X)    
  return model, accuracy

Model, Acc_train = trainKnn(X_train, y_train, 1)
print(Acc_train)
Model3, Acc_train3 = trainKnn(X_train, y_train, 3)
print(Acc_train3)


# # Test
# *   Test the model on the test data.
# *   Print out the accuracy for different k's.
# 
# 

# In[64]:


def testKnn(model, X, y):
    pred = model.predict(X)
    accuracy = sum(pred == y) / len(X)
    return accuracy 
  
testKnn(Model, X_test, y_test)
for k in range(1, 20):
    Model_k, Acc_train = trainKnn(X_train, y_train, k)
Acc_test = testKnn(Model_k, X_test, y_test)
print('%d-NN --> training accuracy = %.4f  /  test accuracy = %.4f' % (k, Acc_train, Acc_test))



# 
# 

# In[79]:


from collections import Counter
class MyKNeighborsClassifier:
    X_train = None
    y_train = None
    
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        
    @staticmethod
    def distance(src, dst):
        dist = np.sqrt(np.sum((src- dst) ** 2))
        return dist

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)    
    def predict_one(self, x):
        kNN= list()
        distances = []
        for (i, x_train) in enumerate(self.X_train):      
            distances.append([i, self.distance(x, x_train)])      
        distances.sort(key=lambda element: element[1])        
        for i in range(self.k):
            kNN.append(distances[i][0])
        
        targets = [self.y_train[i] for i in kNN]
        return Counter(targets).most_common(1)[0][0]
    
    def predict(self, X):    
        predictions = []
        for (i, x) in enumerate(np.array(X)):
            predictions.append(self.predict_one(x))    
        return np.asarray(predictions)


# In[82]:


def myTrainKnn(X, y, k):
    model = MyKNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    pred = model.predict(X)
    accuracy = sum(pred == y) / len(X)    
    return model, accuracy

Model, Acc_train = myTrainKnn(X_train, y_train, 1)
print(Acc_train)
Model3, Acc_train3 = myTrainKnn(X_train, y_train, 3)
print(Acc_train3)


# In[83]:


def myTestKnn(model, X, y):
    pred = model.predict(X)
    accuracy = sum(pred == y) / len(X)
    return accuracy 
  
myTestKnn(Model, X_test, y_test)
for k in range(1, 20):
    Model_k, Acc_train = myTrainKnn(X_train, y_train, k)
    Acc_test = myTestKnn(Model_k, X_test, y_test)
    print('%d-NN --> training accuracy = %.4f  /  test accuracy = %.4f' % (k, Acc_train, Acc_test))


# In[ ]:




