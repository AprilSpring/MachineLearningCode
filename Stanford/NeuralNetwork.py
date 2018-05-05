# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:47:02 2018

@author: User
"""

# neural network learning
# 通过反向传播算法实现神经网络代价函数和梯度计算的非正则化和正则化版本

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('ex4data1.mat')

X = data['X']
y = data['y']

X.shape, y.shape

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
'''
OneHotEncoder(categorical_features='all', dtype=<class 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=False)
'''
y_onehot = encoder.fit_transform(y) #针对y标签进行一次one-hot编码
y_onehot.shape #5000*10
'''
one-hot编码：
假设一共有10类，那么，
类别2的one-hot编码就是[0,1,0,0,0,0,0,0,0,0]，
类别5的one-hot编码就是[0,0,0,0,1,0,0,0,0,0]，
因此，5000个样本对应的y的one-hot编码是5000*10维的0/1矩阵。
'''

#神经元的激活函数多用sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#%% 向前传播
#用于构建神经网络：原始特征(400 + 1) -> 隐藏层节点 (25 + 1) -> 最终分类数目 (10)
#计算分类结果
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values=np.ones(m), axis=1) #为了方便向量化计算，添加theta0的特征，为1，5000*401
    z2 = a1 * theta1.T #5000*25
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) #5000*26
    z3 = a2 * theta2.T #5000*10
    h = sigmoid(z3) #5000*10
    
    return a1, z2, a2, z3, h


#%% 代价函数
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y) #is y_onehot, 5000*10
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # compute the cost
    '''
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:])) #1*10
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:])) #1*10
        J += np.sum(first_term - second_term)
    
    J = J / m
    '''
    J = -(1/m) * np.sum(np.multiply(y,np.log(h)) + np.multiply((1-y),np.log(1-h))) 
    
    return J


# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# 随机初始化完整网络参数大小的参数数组
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

# 将参数数组解开为每个层的参数矩阵
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

#or
#theta1 = (np.random.random([hidden_size,input_size+1]) -0.5)*0.25
#theta1 = np.matrix(theta1)
#theta2 = (np.random.random([num_labels,hidden_size+1]) -0.5)*0.25
#theta2 = np.matrix(theta2)

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)


#%% 正则化代价函数
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # compute the cost
    '''
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    # theta0不参与正则化
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    '''
    
    J = -(1/m) * np.sum(np.multiply(y,np.log(h)) + np.multiply((1-y),np.log(1-h)))
    J = J + (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))
    
    return J

cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)


#%% 反向传播，梯度加正则化
#获得使得代价函数J最小的theta
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
'''

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    '''
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    '''
    J = -(1/m) * np.sum(np.multiply(y,np.log(h)) + np.multiply((1-y),np.log(1-h)))
    J = J + (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))
    
    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d3t = ht - yt  # (1, 10)
        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)


#%% 结果预测
from scipy.optimize import minimize

# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 250})
dir(fmin)
#['fun', 'jac', 'message', 'nfev', 'nit', 'status', 'success', 'x']
fmin.fun #J
fmin.jac #?
fmin.message #'Linear search failed'
fmin.nfev #223?
fmin.nit #15?
fmin.status #4?
fmin.success #False
fmin.x #theta

X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))


