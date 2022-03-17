#!/usr/bin/env python
# coding: utf-8

# # **IDS575: Machine Learning and Statistical Methods**
# ## [Quiz #03 - Logistic Regression and Binary Classification]
# 
# 

# ## Import Libraries
# * See various conventions and acronyms.

# In[135]:


import numpy as np
import pandas as pd
import csv


# ## Load the data into a DataFrame
# * Read directly from a csv (excel-like) data.

# In[136]:


FraudDataset = pd.read_csv('fraud.csv')
print(type(FraudDataset))
print(FraudDataset.keys())


# ## Verify basic data statistics
# * Count the number of features. (i.e., attributes)
# * Count the number of examples. (i.e., instances and labels)
# * Unfortunately we don't know what each feature means due to privacy concerns.
# * Class variable: 0 (standard) / 1 (fradulent)
# 

# In[137]:


def printBasicStats(dataset):
    print('- # of features = %d' % (len(dataset.keys()) - 1))
    print('- # of examples = %d' % len(dataset))
  
printBasicStats(FraudDataset)
print(FraudDataset)


# ## Data inspection
# * See the label imbalance.
# * Measure the baseline accuracy.
# 

# In[138]:


Counts = FraudDataset['Class'].value_counts()
print(Counts)


# In[139]:


pd.set_option('display.max_columns', 10)
print(FraudDataset.describe(exclude=None))

Counts = FraudDataset['Class'].value_counts()
print(Counts)

BaseLineAcc = Counts[0]/(Counts[0] + Counts[1])
print(BaseLineAcc)


# ## Data inspection Part II.
# * Measure the correlation.
# * Let's draw heatmap as an intuitive visualization.
# 

# In[140]:


print(FraudDataset.corr())

import seaborn as sns
sns.heatmap(FraudDataset.corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True))


# ## Data split
# * Must split into train and test data but with respect to the class distribution.
# 

# In[141]:


from sklearn.model_selection import StratifiedShuffleSplit

def splitTrainTest(df, size):
  split = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=0)

  # For each pair of train and test indices,
  X = df.drop('Class', axis=1)
  y = df.Class  
  for trainIndexes, testIndexes in split.split(X, y):
    X_train, y_train = X.iloc[trainIndexes], y.iloc[trainIndexes]
    X_test, y_test = X.iloc[testIndexes], y.iloc[testIndexes]

  return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = splitTrainTest(FraudDataset, 0.2)
print(X_train)


# In[142]:


print(y_train.value_counts())
print(y_test.value_counts())


# ## Logistic Regression 
# * Train the logistic regression.
# * Train again with normalization.
# 

# In[143]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 

def doLogisticRegression(X, y, normalize=False):
  # If normalize option is enabled,
    if normalize:
        # For each feature (indexed by j as usual)
        for j in X.columns:      
      # Subtract its column mean and update the value.
            X[j] -= X[j].mean()
        
      # Divide by its standard deviation and update the value.
            X[j] /= X[j].std()
        
  # Instanciate an object from Logistic Regression class.
    lr = LogisticRegression()
    
  # Perform training and prediction.
    lr.fit(X, y)
    y_pred = lr.predict(X)
      
  # Return training accuracy and confusion matrix.
    return accuracy_score(y, y_pred), confusion_matrix(y, y_pred), lr

TrainAcc, TrainConf, LR = doLogisticRegression(X_train, y_train, normalize=True)
#print(X_train.columns)
print(TrainAcc)
print(TrainConf)


# In[ ]:


y_test_pred = LR.predict(X_test)
TestAcc, TestConf = accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred)
print(TestAcc)
print(TestConf)


# # Programming Assignment (PA)
# *   Implement logistic()
# *   Implement logLikelihood()
# *   Implement predict()
# *   Implement miniBatchGradientDescent()
# *   Play with testYourCode() that compares your implementations agaisnt scikit-learn's results.
# *   Note that your log-likelihood must increase over epoch as you update the model parameter theta toward its maximum.

# In[144]:


class MyLogisticRegression:
 # Randomly initialize the parameter vector.
   theta = None
   
   def logistic(self, z):
       logisticValue = 1.0/(1 + np.exp(-z))
       return logisticValue

   def logLikelihood(self, X, y):
       if not isinstance(self.theta, np.ndarray):
           return 0.0

       h_theta = self.logistic(np.dot(X, self.theta))
       probability1 = y*np.log(h_theta)
       probability0 = (1-y)*np.log(1-h_theta)
       m = X.shape[0]
       return (1.0/m) * np.sum(probability1 + probability0) 

   def fit(self, X, y, alpha=0.01, epoch=50):
   # Extract the data matrix and output vector as a numpy array from the data frame.
   # Note that we append a column of 1 in the X for the intercept.
       X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
       y = np.array(y)  

   # Run mini-batch gradient descent.
       self.miniBatchGradientDescent(X, y, alpha, epoch)

   def predict(self, X):
   # Extract the data matrix and output vector as a numpy array from the data frame.
   # Note that we append a column of 1 in the X for the intercept.
       X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)

   # Perfrom a prediction only after a training happens.
       if isinstance(self.theta, np.ndarray):
           y_pred = self.logistic(X.dot(self.theta))
     ####################################################################################
     # TO-DO: Given the predicted probability value, decide your class prediction 1 or 0.
           y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
     ####################################################################################
           return y_pred_class
       return None

   def miniBatchGradientDescent(self, X, y, alpha, epoch, batch_size=100):    
       (m, n) = X.shape
 
   # Randomly initialize our parameter vector. (DO NOT CHANGE THIS PART!)
   # Note that n here indicates (n+1) because X is already appended by the intercept term.
       np.random.seed(2) 
       self.theta = 0.1*(np.random.rand(n) - 0.5)
       print('L2-norm of the initial theta = %.4f' % np.linalg.norm(self.theta, 2))
   
   # Start iterations
       for iter in range(epoch):
           if (iter % 5) == 0:
               print('+ currently at %d epoch...' % iter)   
               print('  - log-likelihood = %.4f' % self.logLikelihood(X, y))

     # Create a list of shuffled indexes for iterating training examples.     
       indexes = np.arange(m)
       np.random.shuffle(indexes)
       
     # For each mini-batch,
       for i in range(0, m - batch_size + 1, batch_size):
       # Extract the current batch of indexes and corresponding data and outputs.
           indexSlice = indexes[i:i+batch_size]        
           X_batch = X[indexSlice, :]
           y_batch = y[indexSlice]

       # For each feature
      
       for j in np.arange(n):
            self.theta[j] =  self.theta[j] - alpha * (1.0/batch_size) * (sum([np.dot((np.dot(self.theta , X_batch[i]) - y_batch[i]) ,                                                                                     X_batch[i][j])  for i in range(batch_size)]))
         ####################################################################################
         # TO-DO: Perform like a batch gradient desceint within the current mini-batch.
         # Note that your algorithm must update self.theta[j].
                  
         ####################################################################################
         
 

def doMyLogisticRegression(X, y, alpha, epoch, normalize=False):
       if normalize:
           for j in X.columns:
               X[j] -= X[j].mean()
               X[j] /= X[j].std()
       lr = MyLogisticRegression()

 # Perform training and prediction.
       lr.fit(X, y, alpha, epoch,)
       y_pred = lr.predict(X)
     
 # Return training accuracy and confusion matrix.
       return accuracy_score(y, y_pred), confusion_matrix(y, y_pred), lr





   


# In[145]:


def testYourCode(X_train, y_train, X_test, y_test, alpha, epoch):
    trainAcc, trainConf, lr = doLogisticRegression(X_train, y_train, normalize=True)
    y_test_pred = lr.predict(X_test)
    testAcc, testConf = accuracy_score(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred)
    print("Scikit's training/test accuracies = %.4f / %.4f" % (trainAcc, testAcc))
    print("Scikit's training/test confusion matrix\n %s\n %s" % (trainConf, testConf))
    theta = np.append(lr.coef_[0], lr.intercept_)
    print(theta)

  # Test the code with your own version.
    myTrainAcc, myTrainConf, myLR = doMyLogisticRegression(X_train, y_train, alpha, epoch, normalize=True)
    my_y_test_pred = myLR.predict(X_test)
    myTestAcc, myTestConf = accuracy_score(y_test, my_y_test_pred), confusion_matrix(y_test, my_y_test_pred)
    print("My training/test accuracies = %.4f / %.4f" % (myTrainAcc, myTestAcc))
    print("My training/test confusion matrix\n %s\n %s" % (myTrainConf, myTestConf))
    print(myLR.theta)

testYourCode(X_train, y_train, X_test, y_test, 0.05, 100)


# In[ ]:




