#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:13:22 2018

根据模型和交叉验证结果，寻找最佳参数

@author: tinghai
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import scipy as sp
import os
import random


#评估指标
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll


def trainClassifierLGBM_AdjustParameter(x_train,y_train,x_test,y_test):
    
    max_depth = [3,4,5,6,7]
    eta = [0.01, 0.03, 0.1, 0.25, 0.3]
    num_leaves = [50,100,200]
    num_boost_round = [100,200,300,400]
        
    print('LIGHTBGM start...\n')
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_test = lgb.Dataset(x_test,y_test)
    
    # adjust parameter
    loss_all = list()
    para_all = list()
    for m in range(0,len(max_depth)):
        for e in range(0,len(eta)):
            for n in range(0,len(num_leaves)):
                for b in range(0,len(num_boost_round)):
                    params = {'task': 'train','boosting_type': 'gbdt','objective': 'binary',
                              'metric': 'binary_logloss','max_depth':max_depth[m], 'num_leaves': num_leaves[n], 
                              'max_bin':150, 'learning_rate': eta[e],'feature_fraction': 0.5, 
                              'bagging_fraction': 0.85, 'bagging_freq': 5,'verbose': 0}
                    
                    watchlist=[lgb_test]
                    lgbm = lgb.train(params=params, train_set=lgb_train, num_boost_round=num_boost_round[b],
                                     valid_sets=watchlist, verbose_eval=10)
                    
                    p = '{ max_depth: ' + str(max_depth[m]) + ', eta: ' + str(eta[e]) + ', num_leaves: ' + str(num_leaves[n]) + ', num_boost_round:' + str(num_boost_round[b]) + ' }'
                    print(p)
                    para_all.append(p)
                    
                    best_score = list(list(lgbm.best_score.values())[0].values())[0]
                    loss_all.append(best_score)
                    
    return loss_all, para_all



def trainClassifierXGB_AdjustParameter(x_train,y_train,x_test,y_test,test_feature,
                                       test_index,f_names=None,buffer=False):
    
    max_depth = [3,4,5,6,7]
    eta = [0.01,0.03,0.1,0.25,0.3]
    num_round = [100,200,300]
    lambda_value = [10,50,100]
    
        
    
    print('XGBOOST start...\n')
    if buffer==True:
        dtrain=xgb.DMatrix("data/train.buffer")
    else:
        dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=f_names)
        dval=xgb.DMatrix(x_test,label=y_test,feature_names=f_names)
    
    # adjust parameter
    loss_all = list()
    para_all = list()
    for m in range(0,len(max_depth)):
        for e in range(0,len(eta)):
            for n in range(0,len(num_round)):
                for l in range(0,len(lambda_value)):
                    param = {'max_depth':max_depth[m],'eta':eta[e],'min_child_weight':1, 
                             'silent':1, 'subsample':1,'colsample_bytree':1,
                             'gamma':0,'scale_pos_weight':1,'lambda':lambda_value[l],
                             'objective':'binary:logistic'}
                    plst = list(param.items())
                    plst += [('eval_metric', 'auc')]
                    
                    watchlist = [(dtrain, 'train'), (dval, 'eval')]
                    bst = xgb.train(params=plst,dtrain=dtrain,num_boost_round=num_round[n],evals=watchlist,
                                    early_stopping_rounds=500,verbose_eval=10)
                    
                    p = '{ max_depth: ' + str(max_depth[m]) + ', eta: ' + str(eta[e]) + ', num_round: ' + str(num_round[n]) + ', lambda: '+ str(lambda_value[l]) + ' }'
                    print(p)
                    para_all.append(p)
                    
                    best_score = bst.best_score
                    loss_all.append(best_score)
                    
    return loss_all, para_all



#主函数
def main():
    
    path = '/Users/tinghai/Learning/GuanggaoData'
    os.chdir(path + '/source')
    import analysis3
    
    os.chdir(path)
    
    [train_pd, train_feature, train_label, train_index, test_pd, test_feature, test_index, fnames] = analysis3.dataProcessing(
            path=path, train_name='round1_ijcai_18_train_20180301.txt', test_name='round1_ijcai_18_test_a_20180301.txt',
            choice_date=False, choose_date=None, 
            featureExtract=False, featureDrop=False, missingValue=True)
    
    ##2 数据划分
    [x_train,x_test,y_train,y_test] = analysis3.dataSplit(
            train_pd=train_pd,train_feature=train_feature,train_label=train_label,
            f_names=fnames,date_split=True,
            train_d=['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22'],
            test_d=['2018-09-23','2018-09-24'])
    
    #s = random.sample(range(0,train_label.shape[0]), 10000)
    
    #1 lightGBM parameter
    [lgb_loss, lgb_para] = trainClassifierLGBM_AdjustParameter(
            x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
    lgb_loss2 = np.array(lgb_loss)
    lgb_loss2 = lgb_loss2[np.argsort(lgb_loss)]
    lgb_para2 = np.array(lgb_para)
    lgb_para2 = lgb_para2[np.argsort(lgb_loss)]
    lgb_res = pd.concat([pd.DataFrame(lgb_para2),pd.DataFrame(lgb_loss2)],axis=1)    
    print(lgb_res)
    #print('Min loss is ' + str(min(lgb_loss)))
    #print('Best parameter is '+ lgb_para[lgb_loss.index(min(lgb_loss))])
        
    #2 XGBoost parameter
    [xgb_loss, xgb_para] = trainClassifierXGB_AdjustParameter(
            x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
            test_feature=test_feature,test_index=test_index,f_names=fnames,buffer=False)
    xgb_loss2 = np.array(xgb_loss)
    xgb_loss2 = xgb_loss2[np.argsort(xgb_loss)]
    xgb_para2 = np.array(xgb_para)
    xgb_para2 = xgb_para2[np.argsort(xgb_loss)]
    xgb_res = pd.concat([pd.DataFrame(xgb_para2),pd.DataFrame(xgb_loss2)],axis=1)    
    print(xgb_res)
    #print('Min loss is ' + str(min(xgb_loss)))
    #print('Best parameter is '+ xgb_para[xgb_loss.index(min(xgb_loss))])
    #Min loss is 0.09837867231108248
    #Best parameter is { max_depth: 3, eta: 0.1, num_round: 100, lambda: 100 }


if __name__=='__main__':
    
    main()
    
    print('Parameter adjust is end.')
 

