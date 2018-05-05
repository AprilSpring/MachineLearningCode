# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:46:40 2017

@author: BRYAN
"""
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.neighbors.classification import KNeighborsClassifier


data=pd.read_csv('d:/trainCG.csv').fillna(0,axis=1)
x=data.drop(['label'],axis=1)
# min_max_scaler =MinMaxScaler()
# x=min_max_scaler.fit_transform(x)
y=data['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

start=time.time()
reg = linear_model.LogisticRegression(penalty='l1',max_iter=100,C=1.5)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred))
print(end-start)


start=time.time()
reg = DecisionTreeClassifier(max_depth=10,min_samples_split=10,min_samples_leaf=5,max_features=0.75)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred))
print(end-start)

start=time.time()
reg = RandomForestClassifier(n_jobs=8)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred))
print(end-start)


start=time.time()
reg = RandomForestClassifier(n_estimators=200,n_jobs=8,random_state=0)
reg.fit (X_train,y_train)
end=time.time()
y_pred_rf=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred_rf))
print(end-start)


start=time.time()
reg = MLPClassifier(batch_size=50,hidden_layer_sizes=(20),learning_rate_init=0.1,max_iter=300,random_state=0,early_stopping=True)
reg.fit (X_train,y_train)
end=time.time()
y_pred_mlp=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred_mlp))
print(end-start)


start=time.time()
reg = SVC(kernel='rbf',C=0.1, max_iter=100,probability=True)
reg.fit (X_train,y_train)
end=time.time()
y_pred_svc=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred_svc))
print(end-start)


start=time.time()
reg = XGBClassifier(max_depth=4,n_estimators=500,min_child_weight=10,subsample=0.7, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred))
print(end-start)

start=time.time()
reg = LGBMClassifier(num_leaves=40,max_depth=7,n_estimators=200,min_child_weight=10,subsample=0.7, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)
reg.fit (X_train,y_train)
end=time.time()
y_pred_lgb=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred_lgb))
print(end-start)


start=time.time()
reg = ExtraTreesClassifier(n_estimators=100,max_depth=7,min_samples_leaf=10,n_jobs=8,random_state=4)
reg.fit (X_train,y_train)
end=time.time()
y_pred_et=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred_et))
print(end-start)


start=time.time()
reg = KNeighborsClassifier(n_neighbors=4,algorithm='kd_tree')
reg.fit (X_train,y_train)
end=time.time()
y_pred_knn=reg.predict_proba(X_test)[:,1]
print(metrics.roc_auc_score(y_test,y_pred_knn))
print(end-start)

y_pred=0.5*y_pred_et+0.5*y_pred_knn
y_pred=0.5*pd.DataFrame(y_pred_et).rank()+0.5*pd.DataFrame(y_pred_knn).rank()
print(metrics.roc_auc_score(y_test,y_pred))