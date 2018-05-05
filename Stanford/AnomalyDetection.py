# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:31:51 2018

@author: User
"""

# Anomaly detection

# 高斯分布=正态分布

#%% 1.计算矩阵每个特征的均值和方差
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat('data/ex8data1.mat')
X = data['X'] #test set

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
plt.show()

def estimate_gaussian(X):
    mu = X.mean(axis=0) #axis=0，按列求均值
    sigma = X.var(axis=0)
    
    return mu, sigma

mu, sigma = estimate_gaussian(X)

Xval = data['Xval'] #training set
yval = data['yval'] #training label


#%% 2.正态分布概率密度计算
from scipy import stats
dist = stats.norm(mu[0], sigma[0]) #定义一个均值为mu[0]，方差为sigma[0]的正态分布
# dist.pdf(x, loc=0, scale=1) #返回x在均值为0、方差为1的概率分布中的密度函数，即正态分布的概率
# dist.cdf() #累积概率函数，其实是 pdf 积分，也就是密度曲线下的面积，整个的面积为 1
dist.pdf(15)
#dist.pdf(X[:,0])[0:50]

p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])


#%% 3.确定p(x) < epsilon的阈值
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    
    step = (pval.max() - pval.min()) / 1000
    
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    
    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)


#%% 4.确定异常样本
outliers = np.where(p < epsilon) #p为test set依据正态分布计算的概率
anormal = Xval[np.where(p < epsilon)[0],]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
plt.show()



