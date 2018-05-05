# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:45:38 2018

@author: User
"""

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """for ex5
    d['X'] shape = (12, 1)
    pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
    the results
    """
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

X, y, Xval, yval, Xtest, ytest = load_data()

df = pd.DataFrame({'water_level':X, 'flow':y})

sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
plt.show()

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]


#%% 代价函数
def cost(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    m = X.shape[0]

    inner = X @ theta - y  # R(m*1)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    
    square_sum = inner.T @ inner  #等同于np.dot(inner,inner)
    
    cost = square_sum / (2 * m)
    
    return cost

theta = np.ones(X.shape[1])
cost(theta, X, y)


#%% 正则化代价函数
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum() #正则化向不包括theta0

    return cost(theta, X, y) + regularized_term


#%% 梯度函数
def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)

    return inner / m

gradient(theta, X, y)


#%% 正则化梯度函数
def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta

    regularized_term = (l / m) * regularized_term

    return gradient(theta, X, y) + regularized_term

regularized_gradient(theta, X, y)


#%% 线性回归拟合数据
def linear_regression_np(X, y, l=1):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

theta = np.ones(X.shape[0])

final_theta = linear_regression_np(X, y, l=0).get('x')

b = final_theta[0] # intercept
m = final_theta[1] # slope

plt.scatter(X[:,1], y, label="Training data")
plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()

training_cost, cv_cost = [], []

#%% 绘制学习曲线
'''
1.使用训练集的子集来拟合应模型
2.在计算训练代价和交叉验证代价时，没有用正则化
3.记住使用相同的训练集子集来计算训练代价
'''
m = X.shape[0]
for i in range(1, m+1):
#     print('i={}'.format(i))
    res = linear_regression_np(X[:i, :], y[:i], l=0)
    
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
#     print('tc={}, cv={}'.format(tc, cv))
    
    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend(loc=1)
plt.show()

#这个模型拟合不太好, 欠拟合了

#%% 创建多项式特征
def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_ndarray else df

X, y, Xval, yval, Xtest, ytest = load_data()

poly_features(X, power=3)

'''
扩展特征到8阶,或者你需要的阶数
使用归一化来合并xn
don't forget intercept term
'''
def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())

def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).as_matrix()

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]

X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8)


#%% 绘制学习曲线
def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)


#lambda=0    
plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.show()    
#你可以看到训练的代价太低了，不真实. 这是 过拟合了

#lambda=1
plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
plt.show()
#太多正则化了，变成欠拟合状态  


#%% 找到最佳正则参数 λ
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)
    
    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)
    
    training_cost.append(tc)
    cv_cost.append(cv)
    
plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()

# best cv I got from all those candidates
l_candidate[np.argmin(cv_cost)] # λ=1

#计算测试集代价函数最小的λ值
test_cost = []
for l in l_candidate:
    #print(l)
    model = linear_regression_np(X_poly, y, l) #training set创建模型
    theta = model.x
    J = cost(theta, Xtest_poly, ytest) #test set计算最小代价函数
    print('test cost(l={}) = {}'.format(l, J))
    #or print('test cost(l = %s) = %s' % (l, J))
    test_cost.append(J)

l_candidate[np.argmin(test_cost)] #调参后，λ=0.3是最优选择，这个时候测试代价最小


# or
#使用sklearn模块的网格搜索，寻找最优正则参数
from sklearn import linear_model
from sklearn import grid_search

alphas = l_candidate
model = linear_model.Ridge() #创建一个ridge regression model
grid = grid_search.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X_poly, y)
print(grid) #summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha) #λ=0.3


#%% 使用sklearn model
from sklearn import linear_model
from sklearn import metrics

model = linear_model.LinearRegression()

model.fit(X_poly, y) #traing set构建模型

cvPredict = model.predict(Xval_poly) #cv set预测结果
model.score(Xval_poly,yval) #cv set验证模型结果性能

testPredict = model.predict(Xtest_poly) #test set预测结果
model.score(Xtest_poly,ytest) #test set验证模型结果性能


