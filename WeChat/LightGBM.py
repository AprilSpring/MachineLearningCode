#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:41:38 2018

@author: tinghai
"""

#LightGBM

#%% 代码形式
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score


#训练集与测试集
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


#模型构建和预测1
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

gbm.save_model('lightgbm/model.txt')

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)


#模型构建和预测2
param = {
    'max_depth':6,
    'num_leaves':64,
    'learning_rate':0.03,
    'scale_pos_weight':1,
    'num_threads':40,
    'objective':'binary',
    'bagging_fraction':0.7,
    'bagging_freq':1,
    'min_sum_hessian_in_leaf':100
}

bst=lgb.cv(param,train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)

estimators = lgb.train(param,train_data,num_boost_round=len(bst['auc-mean']))

ypred = estimators.predict(dtest[predictors])


#结果评估
print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )


# dump model to json (and save to file)
model_json = gbm.dump_model()

with open('lightgbm/model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)


#特征重要性
print('Feature names:', gbm.feature_name())
print('Feature importances:', list(gbm.feature_importance()))




#%% 配置文件形式

#train.conf内容如下：

# 配置目标是用于训练
task = train

# 训练方式
boosting_type = gbdt

#目标 二分类
objective = binary

# 损失函数
metric = binary_logloss,auc

# frequence for metric output
metric_freq = 1

# true if need output metric for training data, alias: tranining_metric, train_metric
is_training_metric = true

# 特征最大分割 
max_bin = 255

#训练数据地址
data = '/Users/shuubiasahi/Documents/githup/LightGBM/examples/binary_classification/binary.train'

#测试数据
#valid_data = binary.test

# 树的棵树
num_trees = 100

# 学习率
learning_rate = 0.1

# number of leaves for one tree, alias: num_leaf
num_leaves = 63

tree_learner = serial

# 最大线程个数
# num_threads = 8

# feature sub-sample, will random select 80% feature to train on each iteration 
# alias: sub_feature
feature_fraction = 0.8

# Support bagging (data sub-sample), will perform bagging every 5 iterations
bagging_freq = 5

# Bagging farction, will random select 80% data on bagging
# alias: sub_row
bagging_fraction = 0.8

# minimal number data for one leaf, use this to deal with over-fit
# alias : min_data_per_leaf, min_data
min_data_in_leaf = 50

# minial sum hessians for one leaf, use this to deal with over-fit
min_sum_hessian_in_leaf = 5.0

# save memory and faster speed for sparse feature, alias: is_sparse
is_enable_sparse = true

# when data is bigger than memory size, set this to true. otherwise set false will have faster speed
# alias: two_round_loading, two_round
use_two_round_loading = false

# true if need to save data to binary file and application will auto load data from binary file next time
# alias: is_save_binary, save_binary
is_save_binary_file = false

# 模型输出文件
output_model = '/Users/shuubiasahi/Documents/githup/LightGBM/examples/binary_classification/LightGBM_model.txt'
machine_list_file = '/Users/shuubiasahi/Documents/githup/LightGBM/examples/binary_classification/'

# end



#模型训练
#./lightgbm config=train.conf










