# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:05:29 2018

@author: User
"""

# Logistic Regression

#%% 代价函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) #h(x)

nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    z = X * theta.T
    h = sigmoid(z)
    #h = h + 0.0001 #避免h=0时，log(h)为负无穷大程序报错
    first = np.multiply(-y, np.log(h)) #e为底
    second = np.multiply((1 - y), np.log(1 - h))
    return np.sum(first - second) / (len(X))

cost(theta, X, y) #即，代价函数 J(theta)=0.693


#%% 梯度下降
#注意：实际上没有在这个函数中执行梯度下降，我们仅仅在计算一个梯度步长
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    z = X * theta.T
    h = sigmoid(z) 
    error = h - y
    
    grad = (1/X.shape[0])* X.T * error
    
    #for i in range(parameters):
        #term = np.multiply(error, X[:,i])
        #grad[i] = np.sum(term) / len(X)
    #loss = cost(grad.T, X, y) #
    
    return grad

grad = gradient(theta, X, y)
theta = grad.T
cost(theta,X,y) #总是存在log(0)的情况

#多个步长迭代，在计算h(x)时，总是存在log(0)的情况
def gradientDescent2(X, y, theta, iters):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta).T #2*1
    loss = 10           #loss先定义一个数，为了进入循环迭代
    eps = 0.0001         #精度要求
    iter_count = 0      #当前迭代次数
    while( loss > eps and iter_count < iters):
        z = X * theta
        h = sigmoid(z) #that is, h(x)
        err = X.T * (h -y)
        #theta = theta - alpha * err/X.shape[0]
        theta = theta - err/X.shape[0]
        z = X * theta
        h = sigmoid(z)
        #h = h + 0.0001 #避免h=0时，log(h)为负无穷大程序报错
        first = np.multiply(-y, np.log(h)) #e为底
        second = np.multiply((1 - y), np.log(1 - h))
        loss = np.sum(first - second) / (len(X))
        #loss = cost(theta.T, X, y)
        iter_count += 1
    return theta.T, loss

theta = np.zeros(3)
g, cost = gradientDescent2(X, y, theta, alpha=0.01, iters=1)
g, cost = gradientDescent2(X, y, theta, iters=1)



#%% scipy.optimize
#可以用SciPy's truncated newton（TNC）实现寻找最优参数，类似Octave中的fminunc函数
import scipy.optimize as opt
theta = np.zeros(3)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
cost(result[0], X, y) #0.203

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy)) #89%


#%% 正则化
path =  'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
data2.head()

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

data2.head()

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    #reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta,2)) # (alpha/2*m)*sum(theta^2)
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    
    return grad


# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)

learningRate = 1

costReg(theta2, X2, y2, learningRate)

gradientReg(theta2, X2, y2, learningRate)

result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))

theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))


#%% 调用sklearn的线性回归包
from sklearn import linear_model
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
model.score(X2, y2)

