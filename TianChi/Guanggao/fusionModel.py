#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:54:53 2018

@author: tinghai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 23:23:57 2018

考量因素：
1 特征提取
2 归一化、缺失值
3 模型选择和参数
4 模型融合
5 数据泄露

参考代码：
ref: http://blog.csdn.net/haphapyear/article/details/75057407/

@author: tinghai
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
#from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import scipy as sp
import os
#import datetime



#评估指标
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll



#lightGBM模型（need parameterAdjust.py进行调参）
def trainClassifierLGBM(x_train,y_train):
    
    print('LIGHTBGM start...\n')
    
    lgb_train = lgb.Dataset(x_train, y_train)
    
    #Best parameter is { max_depth: 3, eta: 0.03, num_leaves: 100, num_boost_round:200 }
    
    params = {'task': 'train','boosting_type': 'gbdt',
        'objective': 'binary','metric': 'binary_logloss',
        'max_depth':3, 'num_leaves': 100, 'max_bin':150,  
        'learning_rate': 0.03, 'feature_fraction': 0.5,
        'bagging_fraction': 0.85,'bagging_freq': 5, 'verbose': 0}
    
    lgbm = lgb.train(params, lgb_train, num_boost_round=200)
    print(params)
    return lgbm



#xgboost建模（need parameterAdjust.py进行调参）
def trainClassifierXGB(x_train,y_train,f_names=None,buffer=False):
    #用30W条数据测试下（只要是调参）
    
    print('XGBOOST start...\n')
    if buffer==True:
        dtrain=xgb.DMatrix("data/train.buffer")
    else:
        dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=f_names)
        
    #Best parameter is { max_depth: 3, eta: 0.1, num_round: 100, lambda: 100 }

    param = {'max_depth':3,'eta':0.1,'min_child_weight':1, 
             'silent':1, 'subsample':1,'colsample_bytree':1,
             'gamma':0,'scale_pos_weight':1,'lambda':10,
             'objective':'binary:logistic'}
             
    #不用设置线程，XGboost会自行设置所有的
    #param['nthread'] = 8

    plst = list(param.items())
    plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way 
    plst += [('eval_metric', 'ams@0')]
    
    bst = xgb.train(plst,dtrain,num_boost_round=100)
    print(param)
    return bst



#模型融合：交叉验证
def five_fold_cross_validation(train_feature, train_label, test_feature, method, fnames):
        
    skf = list(StratifiedKFold(train_label, 5)) #1/5 data: skf[i][1], 4/5 data: skf[i][0]
    
    train_pd_add = pd.DataFrame()
    test_pred = pd.DataFrame()
    loss = list()
    if method == 'lightGBM':
        for i in range(0,5):
            print(method + ': fold_' + str(i+1))
            x_train = train_feature[skf[i][0]]
            y_train = train_label[skf[i][0]]
            x_test = train_feature[skf[i][1]]
            y_test = train_label[skf[i][1]]
            
            lgbm = trainClassifierLGBM(x_train,y_train)
            
            #1/5 data test
            prob = lgbm.predict(x_test,num_iteration=lgbm.best_iteration)
            tmp = pd.concat([pd.DataFrame(x_test),pd.DataFrame(prob),pd.DataFrame(y_test)],axis=1)
            train_pd_add = pd.concat([train_pd_add,tmp], axis=0)
            
            loss.append(logloss(y_test,prob))
            
            #test set
            prob = lgbm.predict(test_feature,num_iteration=lgbm.best_iteration)
            test_pred = pd.concat([test_pred,pd.DataFrame(prob)],axis=1)
    
    if method == 'XGBoost':
        for i in range(0,5):
            print(method + ': fold_' + str(i+1))
            x_train = train_feature[skf[i][0]]
            y_train = train_label[skf[i][0]]
            x_test = train_feature[skf[i][1]]
            y_test = train_label[skf[i][1]]
            
            bst = trainClassifierXGB(x_train,y_train,f_names=fnames,buffer=False)
            
            #1/5 data test
            dtest = xgb.DMatrix(x_test,feature_names=fnames)
            prob = bst.predict(dtest)
            tmp = pd.concat([pd.DataFrame(x_test),pd.DataFrame(prob),pd.DataFrame(y_test)],axis=1)
            train_pd_add = pd.concat([train_pd_add,tmp], axis=0)
            
            loss.append(logloss(y_test,prob))            

            #test set
            dtest = xgb.DMatrix(test_feature,feature_names=fnames)
            prob = bst.predict(dtest)
            test_pred = pd.concat([test_pred,pd.DataFrame(prob)],axis=1)
    

    columns=list(fnames.values)
    columns.append('predicted_score')
    test_feature_add = pd.concat([pd.DataFrame(test_feature), pd.DataFrame(np.mean(test_pred, axis=1))],axis=1) #按行求均值
    test_feature_add.to_csv(('./result_fusion/' + method + '_test_feature_add.csv'), index=False, header=columns, sep=' ')
    
    columns.append('is_trade')
    train_pd_add.to_csv(('./result_fusion/' + method + '_train_pd_add.csv'), index=False, header=columns, sep=' ')
    
    loss_mean = np.mean(loss)
    print('Average loss is ' + str(loss_mean))
    
    #return train_pd_add, test_feature_add



#模型融合： 建模和结果预测
def fusionModel_predict_test_prob(train_feature, train_label, test_feature, index_file, fnames, method1, method2):
       
    #建模和预测
    if method2 == 'lightGBM':
        lgbm = trainClassifierLGBM(train_feature,train_label)
        print('lgbm训练完成\n')
        prob = lgbm.predict(test_feature, num_iteration=lgbm.best_iteration)
        print('lgbm预测完成\n')
        model = lgbm
    
    if method2 == 'XGBoost':
        bst = trainClassifierXGB(train_feature,train_label,f_names=fnames,buffer=False)
        print('xgboost训练完成\n')
        #print('模型特征重要性：\n')
        #print(bst.get_score())
        test_feature2 = xgb.DMatrix(test_feature,feature_names=fnames)
        prob = bst.predict(test_feature2)
        model = bst
    
    test_feature2 = pd.DataFrame(test_feature)
    prob = pd.DataFrame(prob, index=index_file, columns=['predicted_score'])
    #prob.to_csv(('./result_fusion/submit_construct_' + method1 + '_predict_' + method2 + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), 
    #            index=True, index_label='instance_id', header=['predicted_score'], sep=' ')
    prob.to_csv(('./result_fusion/submit_construct_' + method1 + '_predict_' + method2 + '.csv'), 
                index=True, index_label='instance_id', header=['predicted_score'], sep=' ')
  
    return model
    


def fusionModel(train_feature,train_label,test_feature,index_file,method1,method2,fnames):
    
    #method1建模
    five_fold_cross_validation(train_feature=train_feature, train_label=train_label, 
            test_feature=test_feature, method=method1, fnames=fnames)
    
    #预测后训练集
    trainFile_add = './result_fusion/lightGBM_train_pd_add.csv'
    train_pd_add = pd.read_csv(trainFile_add, header=0, sep=' ')
    fnames_add = train_pd_add.drop('is_trade',axis=1).columns
    train_feature_add = train_pd_add.drop('is_trade',axis=1).values
    train_label_add = train_pd_add.is_trade.values
    
    #预测后测试集
    testFile_add = './result_fusion/lightGBM_test_feature_add.csv'
    test_pd_add = pd.read_csv(testFile_add, header=0, sep=' ')
    test_feature_add = test_pd_add.values
    
    #method2预测
    model = fusionModel_predict_test_prob(train_feature_add, train_label_add, test_feature_add, 
                              index_file=index_file, fnames=fnames_add, method1=method1, method2=method2)
    return model





#主函数
def main():
    
    path = '/Users/tinghai/Learning/GuanggaoData'
    os.chdir(path)
    
    if(os.path.exists('./result_fusion/')):
        os.rmdir('./result_fusion/')
    else:
        os.mkdir('./result_fusion/')
    
    [train_pd, train_feature, train_label, fnames, test_pd, test_feature, test_index] = dataProcessing(
            path, train_name='round1_ijcai_18_train_20180301.txt', test_name='round1_ijcai_18_test_a_20180301.txt')
    
    
    ##1 lightGBM建模，xgboost预测
    fusionModel(train_feature,train_label,test_feature, test_index, method1='lightGBM',method2='XGBoost',fnames=fnames) #0.0907792

    ##2 xgboost建模，lightGBM预测
    fusionModel(train_feature,train_label,test_feature, test_index, method1='XGBoost',method2='lightGBM',fnames=fnames) #0.0938532
      
    
    

if __name__=='__main__':
    print('2. Model fusion start!')
    
    main()
    
    print('2. Model fusion is end.')
    
   

