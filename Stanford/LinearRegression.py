# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:48:30 2018

@author: User
"""

#%% 单变量线性回归

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:/Users/User/Desktop/ex1-linear regression/'
file =  path + 'ex1data1.txt'
data = pd.read_csv(file, header=None, names=['Population', 'Profit']) #pandas.core.frame.DataFrame
data.head()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1] #X是所有行，去掉最后一列，pandas.core.frame.DataFrame
y = data.iloc[:,cols-1:cols] #X是所有行，最后一列

X = np.matrix(X.values) #转换成numpy matrix，才可以使用向量化形式处理数据
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

computeCost(X, y, theta)


#%% 批量梯度下降：用于求解theta
#function1
def gradientDescent1(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        cost_min = min(cost)
        
    return theta, cost_min

g, cost = gradientDescent1(X, y, theta, alpha=0.01, iters=1000)

#function2
def gradientDescent2(X, y, theta, alpha, iters):
    input_x = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta).T #2*1
    loss = 10           #loss先定义一个数，为了进入循环迭代
    eps = 0.0001         #精度要求
    iter_count = 0      #当前迭代次数
    while( loss > eps and iter_count < iters):
        pred_y = input_x * theta
        err = input_x.T * (pred_y -y)
        theta = theta - alpha * err/input_x.shape[0]
        pred_y = input_x * theta
        loss = (1/(2*input_x.shape[0]))*((pred_y - y).T*(pred_y - y))[0,0] #cost function
        iter_count += 1
        #print ("iters_count", iter_count)
    #print ('theta: ',theta )
    #print ('final loss: ', loss)
    #print ('iters: ', iter_count)
    return theta.T, loss

g, cost = gradientDescent2(X, y, theta, alpha=0.01, iters=1000)
#y_pred = X_test*theta

computeCost(X, y, g)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


#%% 多变量线性回归
path =  'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()

data2 = (data2 - data2.mean()) / data2.std()
data2.head()

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent1(X2, y2, theta2, alpha=0.01, iters=1000)

g2, cost2 = gradientDescent2(X2, y2, theta2, alpha=0.01, iters=1000)

# get the cost (error) of the model
computeCost(X2, y2, g2)


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters=1000), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


#%% scikit-learn model
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y) #X is X_train, y is y_train, 训练集模型

x = np.array(X[:, 1].A1)
f = model.predict(X).flatten() #X is X_test, 测试集预测结果
model.score(X,y) #X is X_test, y is y_test, 测试集预测结果的优劣得分

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

'''
备注：
sklearn.linear_model.LinearRegression求解线性回归方程参数时，
首先判断训练集X是不是稀疏矩阵，如是，就用Golub & Kahan双对角线化过程方法来求解；
否则就调用C库LAPACK中的用基于分治法的奇异值分解来求解,
这些解法都跟梯度下降没有半毛钱的关系。
'''

#%% 正规方程：针对线性回归求解theta的简单方法
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y #X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)#感觉和批量梯度下降的theta的值有点差距
computeCost(X, y, final_theta2.T)
#y_pred = X * final_theta2




