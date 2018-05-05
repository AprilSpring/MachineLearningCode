# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:43:57 2018

@author: Administrator
"""

# SVM python script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm


# test1
raw_data = loadmat('data/ex6data1.mat')
type(raw_data) #查看数据类型
#raw_data.keys() #查看dict的keys
#raw_data['X'] #显示第一个key对应的value

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2']) #dict转换成dataframe
data['y'] = raw_data['y']
#data.head()
#data.shape #查看数据维度

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.legend()
plt.show()

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000) #创建线性分类器
svc.fit(data[['X1','X2']], data['y']) #用训练数据拟合分类器模型
data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']]) #返回每个样本到超平面的距离
y_pred = svc.predict(data[['X1','X2']]) #分类器预测新数据
svc.score(data[['X1','X2']],data['y']) #预测结果与真实结果的平均准确性
#svc.score(data[['X1','X2']], data['y'])

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
svc2.score(data[['X1', 'X2']], data['y'])

data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()


#Gaussian kernel
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2

gaussian_kernel(x1, x2, sigma)


# test2
raw_data = loadmat('data/ex6data2.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax.legend()
plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True) #创建非线性分类器
svc.fit(data[['X1', 'X2']], data['y']) #使用训练集拟合模型
y_pred = svc.predict(data[['X1','X2']]) #分类器预测新数据
y_pred = svc.predict_proba(data[['X1','X2']]) 
y_pred = svc.predict_log_proba(data[['X1','X2']])
svc.score(data[['X1', 'X2']], data['y'])

data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
plt.show()


# test3
# 网格搜索
raw_data = loadmat('data/ex6data3.mat')

X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C': None, 'gamma': None} #初始化

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)
        
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

best_score, best_params


# test4
spam_train = loadmat('data/spamTrain.mat')
spam_test = loadmat('data/spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

svc = svm.SVC()
svc.fit(X, y)

print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))





