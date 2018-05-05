# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:46:40 2017

@author: BRYAN
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import xgboost as xgb

data=pd.read_csv('D:\\shop\\code\\o2o\\data\\dataset1.csv').drop('user_id',axis=1)
data.label.replace(-1,0,inplace=True)
data=data.fillna(0)
y=data['label']
x=data.drop('label',axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':7
	    }

watchlist = [(dtrain,'train'),(dtest,'test')]
model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
lr=LogisticRegression(random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))


from scipy.stats import pearsonr
columns=X_train.columns

feature_impotance=[(column,pearsonr(X_train[column],y_train)[0]) for column in columns]
feature_impotance.sort(key=lambda x:x[1])

delete_feature=['merchant_max_distance','discount_man','discount_jian']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
watchlist = [(dtrain,'train'),(dtest,'test')]
model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


'''
使用minepy做特征选择
'''
from minepy import MINE
m = MINE()
m.compute_score(X_train['merchant_max_distance'],y)


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_split=10,min_samples_leaf=10,n_jobs=7,random_state=0).fit(X_train,y_train)
pred=rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))



'''
使用随机森林做特征选择
'''
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
feature_impotance = []
for i in range(len(columns)):
     score = cross_val_score(rf, X_train.values[:, i:i+1], y_train, scoring="r2",cv=ShuffleSplit(len(X_train), 3, 0.3))
     feature_impotance.append((columns[i],round(np.mean(score), 3)))
feature_impotance.sort(key=lambda x:x[1])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
delete_feature=['merchant_min_distance']
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]

rf=RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_split=10,min_samples_leaf=10,n_jobs=7,random_state=0).fit(X_train,y_train)
pred=rf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))



'''
使用xgboost做特征选择
'''
feature_impotance = model.get_fscore()
feature_impotance = sorted(feature_impotance.items(), key=lambda x:x[1])
delete_feature=['user_mean_distance','user_max_distance']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
watchlist = [(dtrain,'train'),(dtest,'test')]
model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)


'''
使用线性模型L1做特征选择
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
lr=LogisticRegression(penalty='l1',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))

feature_impotance=[(i[0],i[1]) for i in zip(columns,lr.coef_[0])]
feature_impotance.sort(key=lambda x:np.abs(x[1]))

delete_feature=['user_mean_distance']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]
lr=LogisticRegression(penalty='l1',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))


'''
使用线性模型L2做特征选择
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
lr=LogisticRegression(penalty='l2',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))

feature_impotance=[(i[0],i[1]) for i in zip(columns,lr.coef_[0])]
feature_impotance.sort(key=lambda x:np.abs(x[1]))

delete_feature=['total_coupon']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
X_train=X_train[[i for i in columns if i not in delete_feature]]
X_test=X_test[[i for i in columns if i not in delete_feature]]
lr=LogisticRegression(penalty='l2',random_state=0,n_jobs=-1).fit(X_train,y_train)
pred=lr.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred))



'''
使用平均精确率减少做特征选择
'''

from sklearn.metrics import r2_score
from collections import defaultdict

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
rf=RandomForestClassifier(n_estimators=300,max_depth=7,min_samples_split=10,min_samples_leaf=10,n_jobs=7,random_state=0)
scores = defaultdict(list)
# crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X_train), 3, 0.3):
    x_train, X_test = X_train.values[train_idx], X_train.values[test_idx]
    Y_train, Y_test = y_train.values[train_idx], y_train.values[test_idx]
    r = rf.fit(x_train, Y_train)
    acc = r2_score(Y_test, rf.predict_proba(X_test)[:,1])
    for i in range(x_train.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict_proba(X_t)[:,1])
        scores[columns[i]].append((acc - shuff_acc) / acc)
feature_impotance=sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()])
feature_impotance

'''
顶层特征选择算法
'''

#稳定性选择
from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston

boston = load_boston()

# using the Boston housing data.
# Data gets scaled automatically by sklearn's implementation
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X, Y)

print("Features sorted by their score:")
feature_impotance=sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),names))
feature_impotance



#递归特征消除
from sklearn.feature_selection import RFE

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
# use linear regression as the model
rf = RandomForestClassifier()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(rf, n_features_to_select=1,verbose=1)
rfe.fit(X_train, y_train)

print("Features sorted by their rank:")
feature_impotance=(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), columns),reverse=True))
feature_impotance

