#!/usr/bin/env python
# coding: utf-8

# # **IDS575: Machine Learning and Statistical Methods**
# ## [Linear Regression and Gradient Descent (PA)]
# 
# 

# ## Import Libraries
# * See various conventions and acronyms.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


# ## Load the data
# * Verify Python type for the dataset.

# In[2]:


HousingDataset = load_boston()
print(type(HousingDataset))
print(HousingDataset.keys())


# ## Verify basic data statistics
# * Count the number of features. (i.e., attributes)
# * Count the number of examples. (i.e., instances and labels)
# * Print out the description of each feature.

# In[14]:


def printBasicStats(dataset):
    print(dataset['feature_names'])
    print(len(dataset['feature_names']), type(dataset['feature_names']))  
    print(dataset['data'].shape, dataset['target'].shape)
    print(dataset['DESCR'])

printBasicStats(HousingDataset)


# ## Convert the dataset to a DataFrame
# *   Not necessarily useful. (scikit-learn works well with default libraries such as list, numpy array, and scipy's sparse matrix)
# *   But using pandas provides more intuitive excel or R-like views.

# In[15]:


def getDataFrame(dataset):
    featureColumns = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    targetColumn = pd.DataFrame(dataset.target, columns=['Target'])
    return featureColumns.join(targetColumn)

DataFrame = getDataFrame(HousingDataset)
print(DataFrame)


# ## Data inspection
# * See correlations between features.
# * Check the quantiles with the highest-correlated feature.
# 

# In[16]:


print(DataFrame.corr())
DataFrame[['RM', 'Target']].describe()


# ## Data cleaning
# * Target could have some outliers because the maximum price is almost doubled to 50.0 though 75% of the data less than 25.0. 
# * We can remove excessively expensive houses.

# In[17]:


Df = DataFrame[DataFrame.Target < 22.5328 + 2*9.1971]
Df[['RM', 'Target']].describe()


# * Rescale the data (different from Gaussian regularization).
# 

# In[18]:


def rescaleVector(x):
    min = x.min()
    max = x.max()
    return pd.Series([(element - min)/(max - min) for element in x])

x_rescale = rescaleVector(Df.RM)
y_rescale = rescaleVector(Df.Target)
print(x_rescale.min(), x_rescale.max())
print(y_rescale.min(), x_rescale.max())


# * Plot the correlation between RM and Target.
# * Observe the linear relationship (excluding some outliers).
# 
# 

# In[19]:


def drawScatterAndLines(x, y, lines=[], titles={'main':None, 'x':None, 'y':None}):
    plt.figure(figsize=(20, 5))
    plt.rcParams['figure.dpi'] = 200
    plt.style.use('seaborn-whitegrid')
    plt.scatter(x, y, label='Data', c='blue', s=6)
    for (x_line, y_line) in lines:
        plt.plot(x_line, y_line, c='red', lw=3, label='Regression')
    plt.title(titles['main'], fontSize=14)
    plt.xlabel(titles['x'], fontSize=11)
    plt.ylabel(titles['y'], fontSize=11)
    plt.legend(frameon=True, loc=1, fontsize=10, borderpad=.6)
    plt.tick_params(direction='out', length=6, color='black', width=1, grid_alpha=.6)
    plt.show()

drawScatterAndLines(x_rescale, y_rescale, titles={'main':'correlation', 'x':'Avg # of Rooms', 'y':'Hosue Price'})


# ## Toy Linear Regression 
# * Use only a single feature RM to fit house price.
# * This could be called Simple Linear Regression.
# * Plot the regression line.
# 

# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def toyLinearRegression(df, feature_name, target_name):
  # This function performs a simple linear regression.
  # With a single feature (given by feature_name)
  # With a rescaling (for stability of test)
    x = rescaleVector(df[feature_name])
    y = rescaleVector(df[target_name])
    x_train = x.values.reshape(-1, 1)
    y_train = y.values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_train_pred = lr.predict(x_train)
  
  # Return training error and (x_train, y_train, y_train_pred)
    return mean_squared_error(y_train, y_train_pred), (x_train, y_train, y_train_pred)

ToyTrainingError, (x_rescale_train, y_rescale_train, y_rescale_train_pred) = toyLinearRegression(Df, 'RM', 'Target')
print('training error = %.4f' % ToyTrainingError)
drawScatterAndLines(x_rescale_train, y_rescale_train, lines=[(x_rescale_train, y_rescale_train_pred)], titles={'main':'correlation', 'x':'RM', 'y':'Target'})


# ## Main Linear Regression 
# * Use all of multi-variate features to fit house price.
# * This could be called Multiple Linear Regression.

# In[21]:


from sklearn.model_selection import train_test_split
def splitTrainTest(df, size):
    X, y = df.drop('Target', axis=1), df.Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=X.shape[0] - size, random_state=0)
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = splitTrainTest(Df, 350)
LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.coef_)
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)


# ## Measure training and test accuracy
# * Use Mean Squared Error.

# In[13]:


from sklearn.metrics import mean_squared_error
print('Training error = %.4f' % mean_squared_error(y_train, y_train_pred))
print('Test error = %.4f' % mean_squared_error(y_test, y_test_pred))


# # Programming Assignment (PA)
# * Implement predict().
# * Implement batchGradientDescent().
# * Implement stocGradientDescent().
# * Implement normalEquation().
# * Play with testYourCode() that compares your implementations against scikit-learn's results. **Do not change alpha and epoch options separately provided to bgd and sgd for single-feature simple linear regression.**
# * Once everything is done, then compare your implementations against scikit-learn's results using the entire features. **Now play with different alpha and epoch values, reporting your comparative impressions among bgd, sgd, and normal equations.**

# In[26]:


class MyLinearRegression:  
    theta = None
    
    def fit(self, X, y, option, alpha, epoch):
        X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        y = np.array(y)       
        if option.lower() in ['bgd', 'gd']:
            # Run batch gradient descent.
            self.theta = self.batchGradientDescent(X, y, alpha, epoch)      
        elif option.lower() in ['sgd']:
      # Run stochastic gradient descent.
            self.theta = self.stocGradientDescent(X, y, alpha, epoch)
        else:
      # Run solving the normal equation.      
            self.theta = self.normalEquation(X, y)
        
    def predict(self, X):
        X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
        y_pred = np.array([])
        if isinstance(self.theta, np.ndarray):
            theta = self.theta
            for x in X:
        # TO-DO: #############################################     
                y_pred = np.dot(X, self.theta) 
      ######################################################
            return y_pred
        return None

    def batchGradientDescent(self, X, y, alpha=0.00001, epoch=100000):
        (m, n) = X.shape      
        theta = np.zeros((n, 1), dtype=np.float64)
        for iter in range(epoch):
            if (iter % 1000) == 0:
                print('- currently at %d epoch...' % iter) 
        y_size = y.size
        for j in range(n):
         # TO-DO: ############################################# 
            theta[j] = theta[j] - (alpha * (sum([(np.dot(X[i], theta) - y[i]) * X[i][j] for i in range(m)])[0]) / m)
        ######################################################
        return theta

    def stocGradientDescent(self, X, y, alpha=0.000001, epoch=10000):
        (m, n) = X.shape
        theta = np.zeros((n, 1), dtype=np.float64)
        for iter in range(epoch):
            if (iter % 100) == 0:
                print('- currently at %d epoch...' % iter)
        for i in range(m):
            for j in range(n):
            # TO-DO: ############################################# 
                theta[j] = theta[j] - alpha * (np.dot(X[i], theta) - y[i]) * X[i][j]
          ######################################################    
        return theta

    def normalEquation(self, X, y):
        # TO-DO: ############################################# 
        theta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    
    ######################################################
        return theta

    @staticmethod
    def toyLinearRegression(df, feature_name, target_name, option, alpha, epoch):
        # This function performs a simple linear regression.
    # With a single feature (given by feature_name)
    # With a rescaling (for stability of test)
        x = rescaleVector(df[feature_name])
        y = rescaleVector(df[target_name])
        x_train = x.values.reshape(-1, 1)
        y_train = y.values.reshape(-1, 1)

    # Perform linear regression.    
        lr = MyLinearRegression()
        lr.fit(x_train, y_train, option, alpha, epoch)
        y_train_pred = lr.predict(x_train)
    
    # Return training error and (x_train, y_train, y_train_pred)
        return mean_squared_error(y_train, y_train_pred), (x_train, y_train, y_train_pred)



# In[27]:


def testYourCode(df, feature_name, target_name, option, alpha, epoch):
    trainError0, (x_train0, y_train0, y_train_pred0) = toyLinearRegression(df, feature_name, target_name)
    trainError1, (x_train1, y_train1, y_train_pred1) = MyLinearRegression.toyLinearRegression(df, feature_name, target_name, option, alpha, epoch)
    drawScatterAndLines(x_train0, y_train0, lines=[(x_train0, y_train_pred0)], titles={'main':'Linear Regression', 'x':feature_name, 'y':target_name})
    drawScatterAndLines(x_train1, y_train1, lines=[(x_train1, y_train_pred1)], titles={'main':'Linear Regression', 'x':feature_name, 'y':target_name})
    return trainError0, trainError1

TrainError0, TrainError1 = testYourCode(Df, 'DIS', 'Target', option='sgd', alpha=0.001, epoch=500)
print("Scikit's training error = %.6f / My training error = %.6f --> Difference = %.4f" % (TrainError0, TrainError1, np.abs(TrainError0 - TrainError1)))
TrainError0, TrainError1 = testYourCode(Df, 'RM', 'Target', option='bgd', alpha=0.1, epoch=5000)
print("Scikit's training error = %.6f / My training error = %.6f --> Difference = %.4f" % (TrainError0, TrainError1, np.abs(TrainError0 - TrainError1)))


# In[28]:


MyLR = MyLinearRegression()
MyLR.fit(X_train, y_train.values.reshape(-1, 1), option='sgd', alpha=0.000001, epoch=10000)
print(MyLR.theta)
y_train_pred = MyLR.predict(X_train)
y_test_pred = MyLR.predict(X_test)

print('Training error = %.4f' % mean_squared_error(y_train, y_train_pred))
print('Test error = %.4f' % mean_squared_error(y_test, y_test_pred))


# In[ ]:




