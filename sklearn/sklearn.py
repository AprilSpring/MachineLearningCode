# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:36:48 2018

@author: Administrator
"""

#%% sklearn model learning

import sklearn

'''
从功能来分： 
1. Preprocessing
2. Classification
3. Regression
4. Clustering
5. Dimensionality reduction
6. Model selection


[Preprocessing]
数据归一化：sklearn.preprocessing.scale(X)
           sklearn.preprocessing.normalize(X)
特征选择：sklearn.ensemble.ExtraTreesClassifier()


[Classification]
logistic regression逻辑回归: sklearn.linear_model.LogisticRegression
Naive Bayes朴素贝叶斯: sklearn.naive_bayes
svm支持向量机: sklearn.svm
KNN最近邻:sklearn.neighbors
Decision Tree决策树: sklearn.tree
Neural network神经网络: sklearn.neural_network


[Clustering]
KMeans: K均值聚类
AffinityPropagation: 吸引子传播
AgglomerativeClustering: 层次聚类
Birch
DBSCAN
FeatureAgglomeration: 特征聚集
MiniBatchKMeans
MeanShift
SpectralClustering: 谱聚类
'''

#%% 数据预处理问题[1]：数据归一化(Data Normalization) 
#将特征数据缩放到0-1范围
from sklearn import preprocessing

#scale the data attributes
scaled_X = preprocessing.scale(X)
 
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
 
# standardize the data attributes
standardized_X = preprocessing.scale(X)


#%% 数据预处理问题[2]：特征选择(Feature Selection)
#树算法(Tree algorithms)计算特征的信息量

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_) #返回每个特征的重要度


#%% 回归问题[1]：Linear regression
'''
备注：
sklearn.linear_model.LinearRegression求解线性回归方程参数时，
首先判断训练集X是不是稀疏矩阵，如是，就用Golub & Kahan双对角线化过程方法来求解；
否则就调用C库LAPACK中的用基于分治法的奇异值分解来求解,
这些解法都跟梯度下降没有半毛钱的关系。
'''
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train) 
'''
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
'''
f = model.predict(X_test) #预测结果
model.score(X_test,y_test) #y is True labels for X


#%% 分类问题[1]：Logistic regression (一般用于二分类，也可用于多分类)
from sklearn import linear_model
from sklearn import metrics

model = linear_model.LogisticRegression(penalty='l2', C=1.0) 
#penalty is 'l1' or 'l2'，如果选择'l2'还是过拟合的话，可以考虑'l1'正则项
#C是正则化参数
model.fit(X, y)

print('MODEL')
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print('RESULT')
print(metrics.classification_report(expected, predicted))
print('CONFUSION MATRIX')
print(metrics.confusion_matrix(expected, predicted))


#%% 分类问题[2]：Naive Bayes (尤其适用于多分类)
from sklearn import metrics
from sklearn import naive_bayes
model = naive_bayes.GaussianNB()
model.fit(X, y)
print('MODEL')
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print('RESULT')
print(metrics.classification_report(expected, predicted))
print('CONFUSION MATRIX')
print(metrics.confusion_matrix(expected, predicted))


#%% 分类问题[3]：svm
from sklearn import metrics
from sklearn import svm
'''
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

dir(svm) #显示svm模块中的可用函数
help(svm.LinearSVC)

svm parameters：
decision_function(X)	Distance of the samples X to the separating hyperplane.
fit(X, y[, sample_weight])	Fit the SVM model according to the given training data.
get_params([deep])	Get parameters for this estimator.
predict(X)	Perform classification on samples in X.
score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels. y is True labels for X.
set_params(**params)	Set the parameters of this estimator.
predict_proba Compute probabilities of possible outcomes for samples in X.
predict_log_proba Compute log probabilities of possible outcomes for samples in X.
'''
# fit a SVM model to the data
model = svm.SVC()
model.fit(X, y)
#svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000) #线性SVM
#svc = svm.SVC(C=100, gamma=10, probability=True) #非线性SVM
print(model)
# make predictions
expected = y
predicted = model.predict(X)
y_pred = svc.predict_proba(X_test) #预测结果和概率，线性模型不适用
y_pred = svc.predict_log_proba(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

#返回测试集和标签的平均精确度，衡量模型效能
svc.score(X_test,y_test) #y is True labels for X


#%% 分类问题[4]：KNN (K近邻)
from sklearn import metrics
from sklearn import neighbors
# fit a k-nearest neighbor model to the data
model = neighbors.KNeighborsClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


#%% 分类问题[5]：决策树 (Decision Tree) (用于分类或回归)
from sklearn import metrics
from sklearn import tree
# fit a CART model to the data
model = tree.DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


#%% 聚类问题[1]：K-mean (K均值)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(X)
#dir(model)
centroids = model.cluster_centers_
C = model.predict(X)


#%% 网格搜索，获取最佳模型参数
import numpy as np
from sklearn import linear_model
from sklearn import grid_search
'''
grid.fit()：运行网格搜索
grid_scores_：给出不同参数情况下的评价结果
best_params_：描述了已取得最佳结果的参数的组合
best_score_：成员提供优化过程期间观察到的最好的评分
'''
# prepare a range of alpha values to test
alphas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
# create and fit a ridge regression model, testing each alpha
model = linear_model.Ridge()
grid = grid_search.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X, y)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)


