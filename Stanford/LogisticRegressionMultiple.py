# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:44:44 2018

@author: Administrator
"""

# 逻辑回归的多分类问题

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


#%% 目标函数
data = loadmat('ex3data1.mat')
data['X'].shape, data['y'].shape
X = data['X']
y = data['y']
rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10, params + 1))
#X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
theta = np.zeros(params + 1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))
X.shape, y_0.shape, theta.shape, all_theta.shape


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

#%% 梯度下降
# 使用循环方式的梯度函数
def gradient_with_loop(theta, X, y, learningRate):
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

# 向量化梯度函数
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    #parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()


#%% 多分类情况的最优化目标函数
from scipy.optimize import minimize

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        #y_i = np.reshape(y_i, (rows, 1))
        y_i = np.reshape(y_i, (len(y_i),1))
        
        # minimize the objective function
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        #result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
        #dir(fmin)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

number = len(np.unique(data['y'])) #查看有几类标签
all_theta = one_vs_all(data['X'], data['y'], number, 1) #number*(X.shape[1])

#%% 结果预测
'''
计算每个类的类概率，对于每个训练样本,
输出类标签为具有最高概率的类
'''
def predict_all(X, all_theta):
    rows = X.shape[0]
    #params = X.shape[1]
    #num_labels = all_theta.shape[0]
    
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta) #10*401
    
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)
    
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1) #按行返回最大值第一次出现所在的索引
    
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax

y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
#or print ('accuracy = %s' % str(accuracy*100) + '%')

