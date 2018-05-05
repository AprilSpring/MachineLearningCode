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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR,LinearSVR
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.neighbors.regression import KNeighborsRegressor

data=pd.read_csv('G:\\Competition\\cainiao\\data\\all_window.csv').fillna(0,axis=1)
x=data.drop('label',axis=1)
# min_max_scaler =MinMaxScaler()
# x=min_max_scaler.fit_transform(x)
y=data['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

start=time.time()
reg = linear_model.LinearRegression()
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = linear_model.Ridge()
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = linear_model.Ridge(alpha=0.5)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = linear_model.Lasso()
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = linear_model.Lasso(alpha=2)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = linear_model.Lasso(alpha=2,max_iter=10)
reg.fit (X_train,y_train)
end=time.time()
y_pred_lr=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred_lr))
print(end-start)


start=time.time()
reg = DecisionTreeRegressor(max_depth=10,min_samples_split=10,min_samples_leaf=5,max_features=0.75)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)

start=time.time()
reg = RandomForestRegressor(n_jobs=8)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = RandomForestRegressor(n_estimators=200,n_jobs=8,random_state=0)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = MLPRegressor(batch_size=50,hidden_layer_sizes=(20),learning_rate_init=0.1,max_iter=300,random_state=0,early_stopping=True)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = SVR(kernel='rbf',C=0.1, epsilon=0.1,max_iter=100)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = XGBRegressor(max_depth=4,n_estimators=500,min_child_weight=10,subsample=0.7, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)

start=time.time()
reg = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=200,min_child_weight=10,subsample=0.7, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)
reg.fit (X_train,y_train)
end=time.time()
y_pred_lgb=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred_lgb))
print(end-start)


start=time.time()
reg = ExtraTreesRegressor(n_estimators=100,max_depth=7,min_samples_leaf=10,n_jobs=8)
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


start=time.time()
reg = KNeighborsRegressor(n_neighbors=4,algorithm='kd_tree')
reg.fit (X_train,y_train)
end=time.time()
y_pred=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(end-start)


y_pred=(y_pred_lr+y_pred_lgb)/2
print(metrics.mean_squared_error(y_test,y_pred))


def cross_search(x,y,model,param_grid):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1,cv=5)
    grid_search.fit(x,y)
    # best_parameters = grid_search.best_estimator_.get_params()
    return  grid_search

def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

start=time.time()
reg = linear_model.Lasso(alpha=2,max_iter=10)
reg.fit (X_train,y_train)
end=time.time()
y_pred_lr=reg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred_lr))
print(end-start)