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
6 基于时间戳等特征权重

参考代码：
ref: http://blog.csdn.net/haphapyear/article/details/75057407/

@author: tinghai
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import scipy as sp
import os
import datetime
import time


#%% 评估指标
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll


#%%单独模型
#lightGBM模型（need parameterAdjust.py进行调参）
def trainClassifierLGBM(x_train,y_train):
    
    print('LIGHTBGM start...\n')
    
    lgb_train = lgb.Dataset(x_train, y_train)
    
    #Best parameter is { max_depth: 3, eta: 0.03, num_leaves: 100, num_boost_round:200 }
    
    params = {'task': 'train','boosting_type': 'gbdt',
        'objective': 'binary','metric': 'binary_logloss',
        'max_depth':3, 'num_leaves': 100, 'max_bin':150,  
        'learning_rate': 0.03, 'feature_fraction': 0.5,
        'bagging_fraction': 0.85,'bagging_freq': 5, 'verbose': 0, 
        'scale_pos_weight':1}
    
    lgbm = lgb.train(params, lgb_train, num_boost_round=200)
    
    #LGBMClassifier(boosting_type='gbdt', objective="multiclass", nthread=8, seed=42)
    
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
    
    #参数设置
    #eta[默认0.3]：通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2
    #min_child_weight[默认1]：XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
    #max_depth[默认6]:和GBM中的参数相同，这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本
    #subsample[默认1]和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1
    #colsample_bytree:和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
    #(没咋用)gamma [default=0]：模型在默认情况下，对于一个节点的划分只有在其loss function 得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值gamma值使得算法更conservation，且其值依赖于loss function ，在模型中应该进行调参。
    #scale_pos_weight[default=1]A value greater than 0 can be used in case of high class imbalance as it helps in faster convergence.大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
    #binary:logistic返回预测的概率(不是类别,正类)
    #lambda[默认1]权重的L2正则化项。(和Ridge regression类似)。 这个参数是用来控制XGBoost的正则化部分的。
    #--------笔记--------
    #基本只要改max_depth,eta,num_round这3个，其他的没发现什么太大的提高
    #学习率（eta）调小,num_round调大可以提高效果，但是如果eta太大，num_round也大效果会很差

    #0.25 200
    #0.1 500
    
    #Best parameter is { max_depth: 3, eta: 0.1, num_round: 100, lambda: 100 }

    param = {'max_depth':3,'eta':0.1,'min_child_weight':1, 
             'silent':1, 'subsample':1,'colsample_bytree':1,
             'gamma':0,'scale_pos_weight':1,'lambda':10,
             'objective':'binary:logistic'}
             
    
    #param['nthread'] = 8  #不用设置线程，XGboost会自行设置所有的

    plst = list(param.items())
    plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way 
    plst += [('eval_metric', 'ams@0')]
    
    bst = xgb.train(plst,dtrain,num_boost_round=100)
    
    #通过xgboost模型，衡量特征重要性，删除不重要特征，减少模型复杂度
    #feature_import = sorted(bst.get_score().items(), key=lambda d: d[1], reverse=True) 
        
    print(param)
    return bst


'''
def trainClassifierXGB2(x_train,y_train):
    
    bst2 = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, 
                     min_child_weight=1,gamma=0,subsample=1,colsample_bytree=1,
                     objective='binary:logistic',scale_pos_weight=1, seed=27)
    bst2.fit(x_train,y_train)
    
    return bst2
'''


#交叉验证
def cross_validation(train_feature, train_label, method, fnames, test_size=0.2, random_state=10):
        
    x_train,x_test,y_train,y_test = train_test_split(train_feature,train_label,test_size=test_size,random_state=random_state)
    print('完成训练集和测试集划分...\n')
    #del feature_all
    #del label_all
    
    if method == 'lightGBM':
        lgbm = trainClassifierLGBM(x_train,y_train)
        print('lgbm训练完成\n')
        prob = lgbm.predict(x_test,num_iteration=lgbm.best_iteration)
        loss = logloss(y_test,prob)
        print('lgbm交叉验证损失为:',loss)
    
    if method == 'XGBoost':
        bst = trainClassifierXGB(x_train,y_train,f_names=fnames,buffer=False)
        #feature_import = sorted(bst.get_score().items(), key=lambda d: d[1], reverse=True) 
        print('xgboost训练完成\n')
        dtest = xgb.DMatrix(x_test,feature_names=fnames)
        prob = bst.predict(dtest)
        #prob = bst2.predict(x_test)
        loss = logloss(y_test,prob)
        print('xgboost交叉验证损失为:',loss)
    
    return loss


#建模和结果预测
def predict_test_prob(train_feature, train_label, test_feature, test_index, fnames, method):
       
    #建模和预测
    if method == 'lightGBM':
        lgbm = trainClassifierLGBM(train_feature,train_label)
        print('lgbm训练完成\n')
        prob = lgbm.predict(test_feature, num_iteration=lgbm.best_iteration)
        print('lgbm预测完成\n')
        model = lgbm
    
    if method == 'XGBoost':
        bst = trainClassifierXGB(train_feature,train_label,f_names=fnames,buffer=False)
        print('xgboost训练完成\n')
        #print('模型特征重要性：\n')
        #print(bst.get_score())
        test_feature2 = xgb.DMatrix(test_feature,feature_names=fnames)
        prob = bst.predict(test_feature2)
        model = bst
    
    test_feature2 = pd.DataFrame(test_feature)
    prob = pd.DataFrame(prob, index=test_index, columns=['predicted_score'])
    #prob.to_csv(('./result/submit_' + method + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), 
    #            index=True, index_label='instance_id', header=['predicted_score'], sep=' ')
    prob.to_csv(('./result/submit_' + method + '.csv'), index=True, 
                index_label='instance_id', header=['predicted_score'], sep=' ')

    return model
    

#%% 模型融合
#交叉验证
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


#建模和结果预测
def fusionModel_predict_test_prob(train_feature, train_label, test_feature, test_index, fnames, method1, method2):
       
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
    prob = pd.DataFrame(prob, index=test_index, columns=['predicted_score'])
    prob.to_csv(('./result_fusion/submit_construct_' + method1 + '_predict_' + method2 + '.csv'), 
                index=True, index_label='instance_id', header=['predicted_score'], sep=' ')
  
    return model
    

#模型2预测模型1的新特征
def fusionModel(train_feature,train_label,test_feature,test_index,method1,method2,fnames):
    
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
                              test_index=test_index, fnames=fnames_add, method1=method1, method2=method2)
    return model


#%% 模型参数调优
'''
def adjustParameter(method,x_train,y_train):
    
    if method == 'lightGBM':
        lgbm = trainClassifierLGBM(x_train, y_train)
        max_depth = [3,4,5,6,7]
        learning_rate = [0.001, 0.01, 0.03, 0.1, 0.25, 0.3]
        num_leaves = [20,50,100,200]
        num_boost_round = [100,200,300]
        param_grid = dict(max_depth=max_depth, learning_rate=learning_rate, num_leaves=num_leaves, num_boost_round=num_boost_round)
        #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        gsearch = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring='neg_log_loss', n_jobs=-1, cv=5)
        gsearch.fit(x_train, y_train)
        
    if method == 'XGBoost':    
        bst2 = XGBClassifier()
        #bst = trainClassifierXGB(x_train, y_train)
        max_depth = [3,4,5,6,7]
        learning_rate = [0.001,0.01,0.03,0.1,0.25,0.3]
        num_round = [100,200,300]
        lambda_value = [10,50,100]
        param_grid = dict(max_depth=max_depth, learning_rate=learning_rate, num_round=num_round, lambda_value=lambda_value)
        #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        gsearch = GridSearchCV(estimator=bst2, param_grid=param_grid, scoring='neg_log_loss', n_jobs=-1, cv=5)
        gsearch.fit(x_train, y_train)
        
    print('CV Results: ', gsearch.cv)
    print('Best Params: ', gsearch.best_params_)
    print('Best Score: ', gsearch.best_score_)
    return gsearch.grid_scores_
'''    


#%% 训练集和测试集特征数据处理
def dataProcessing(path, train_name='round1_ijcai_18_train_20180301.txt', test_name='round1_ijcai_18_test_a_20180301.txt'):
    
    #path = '/Users/tinghai/Learning/GuanggaoData'
    os.chdir(path)
    
    drop_feature_train = ['instance_id','is_trade']
    drop_feature_test = ['instance_id']
    
    #日期脱敏
    def time2cov(time_):
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))
    
    

    
    
    
    
    def grep_count(file):
        r = list()
        for i in range(0,file.shape[0]):
            r1 = file.iloc[i,0].count(";") + 1
            r.append(r1)
        r = pd.DataFrame(r)
        return r   
    
    
    #1 训练数据
    trainFile = './data/' + train_name
    train_pd = pd.read_csv(trainFile, header=0, sep=' ')
    
    #1.1 for 'item_category_list'
    if(os.path.exists('./data/item_category_list_train.csv')):
        pass
    else:
        os.system("cut -f3 -d ' ' " + trainFile + " |cut -f2 -d ';' > ./data/item_category_list_train.csv")
    item_category_file_train = pd.read_csv('./data/item_category_list_train.csv', header=0, sep=' ')
    train_pd.item_category_list = item_category_file_train.values




    



 for i in range(0,3):
        train_pd['item_category_list_%d'%(i)] = train_pd['item_category_list'].apply(
                lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")





    data = train_pd




    #1.2 for 'item_property_list'
    train_pd['item_category_list'] = train_pd['item_category_list'].apply(lambda x: x.split(";")[1])
    
    
    if(os.path.exists('./data/item_property_num_train.csv')):
        pass
    else:
        os.system("cut -f4 -d ' ' " + trainFile + " > ./data/item_property_list_train.csv")
        file = pd.read_csv('./data/item_property_list_train.csv', header=0, sep=' ')
        res = grep_count(file)
        res.to_csv(('./data/item_property_num_train.csv'), index=False, header=['item_property_list'], sep=' ')
    item_property_file_train = pd.read_csv('./data/item_property_num_train.csv', header=0, sep=' ')
    train_pd.item_property_list = item_property_file_train.values
    
    #1.3 for 'predict_category_property'
    if(os.path.exists('./data/predict_category_property_num_train.csv')):
        pass
    else:
        os.system("cut -f4 -d ' ' " + trainFile + " > ./data/predict_category_property_train.csv")
        file = pd.read_csv('./data/predict_category_property_train.csv', header=0, sep=' ')
        res = grep_count(file)
        res.to_csv(('./data/predict_category_property_num_train.csv'), index=False, header=['predict_category_property'], sep=' ')
    predict_category_property_file_train = pd.read_csv('./data/predict_category_property_num_train.csv', header=0, sep=' ')
    train_pd.predict_category_property = predict_category_property_file_train.values
    
    #1.4 for 'context_timestamp'
    train_pd['context_timestamp'] = train_pd['context_timestamp'].apply(time2cov)
    
    
    
    
    
    
    
    
    train_pd.index = train_pd.instance_id
    
    fnames = train_pd.drop(drop_feature_train,axis=1).columns
    train_feature = train_pd.drop(drop_feature_train,axis=1).values
    train_label = train_pd.is_trade.values
    
    
    #2 测试数据
    testFile = './data/' + test_name
    test_pd = pd.read_csv(testFile, header=0, sep=' ')
    
    #2.1 for 'item_category_list'
    if(os.path.exists('./data/item_category_list_test.csv')):
        pass
    else:
        os.system("cut -f3 -d ' ' " + testFile + " |cut -f2 -d ';' > ./data/item_category_list_test.csv")
    item_category_file_test = pd.read_csv('./data/item_category_list_test.csv', header=0, sep=' ')
    test_pd.item_category_list = item_category_file_test.values
    
    #2.2 for 'item_property_list'
    if(os.path.exists('./data/item_property_num_test.csv')):
        pass
    else:
        os.system("cut -f4 -d ' ' " + testFile + " > ./data/item_property_list_test.csv")
        file = pd.read_csv('./data/item_property_list_test.csv', header=0, sep=' ')
        res = grep_count(file)
        res.to_csv(('./data/item_property_num_test.csv'), index=False, header=['item_property_list'], sep=' ')
    item_property_file_test = pd.read_csv('./data/item_property_num_test.csv', header=0, sep=' ')
    test_pd.item_property_list = item_property_file_test.values      
    
    #2.3 for 'predict_category_property'
    if(os.path.exists('./predict_category_property_num_test.csv')):
        pass
    else:
        os.system("cut -f4 -d ' ' " + testFile + " > ./data/predict_category_property_test.csv")
        file = pd.read_csv('./data/predict_category_property_test.csv', header=0, sep=' ')
        res = grep_count(file)
        res.to_csv(('./data/predict_category_property_num_test.csv'), index=False, header=['predict_category_property'], sep=' ')
    predict_category_property_file_test = pd.read_csv('./data/predict_category_property_num_test.csv', header=0, sep=' ')
    test_pd.predict_category_property = predict_category_property_file_test.values
    
    test_pd.index = test_pd.instance_id
    test_index = test_pd.index
    test_feature = test_pd.drop(drop_feature_test,axis=1).values
    
    return train_pd, train_feature, train_label, fnames, test_pd, test_feature, test_index
    

#%% 主函数
def main():
    
    path = '/Users/tinghai/Learning/GuanggaoData'
    #path = 'D:/Learn/DataAnalyst/GuanggaoData'
    os.chdir(path)
    
    [train_pd, train_feature, train_label, fnames, test_pd, test_feature, test_index] = dataProcessing(
            path=path, train_name='round1_ijcai_18_train_20180301.txt', test_name='round1_ijcai_18_test_a_20180301.txt')
    
    
    ## 1 单独模型
    if(os.path.exists('./result/')):
        os.rmdir('./result/')
    else:
        os.mkdir('./result/')
    
    #1.1 交叉验证，计算loss
    print('交叉验证...\n')
    loss_lgbm = cross_validation(train_feature=train_feature, train_label=train_label, 
                                 method='lightGBM', fnames=fnames, test_size=0.2, random_state=10)
    print(loss_lgbm) #0.09175184 0.091711149 0.09164522
    
    loss_bst = cross_validation(train_feature=train_feature, train_label=train_label, 
                                method='XGBoost', fnames=fnames, test_size=0.2, random_state=10)
    print(loss_bst) #0.0911904 0.0911939 0.09116922
    
    #1.2 全量训练样本建模和预测
    predict_test_prob(train_feature=train_feature, train_label=train_label, 
                      test_feature=test_feature, test_index=test_index, fnames=fnames, method='lightGBM')
    predict_test_prob(train_feature=train_feature, train_label=train_label, 
                      test_feature=test_feature, test_index=test_index,fnames=fnames, method='XGBoost')
    
    '''
    #样本不均衡
    #y0 = train_pd.is_trade[train_pd.is_trade.isin([0])]
    #y1 = train_pd.is_trade[train_pd.is_trade.isin([1])]
    
    #仅选取进3天的数据进行建模
    stamp = train_pd.context_timestamp
    time.localtime(min(train_pd.context_timestamp))
    time.localtime(max(train_pd.context_timestamp))
    
    #通过xgboost模型，衡量特征重要性，删除不重要特征，减少模型复杂度
    feature_import = sorted(bst.get_score().items(), key=lambda d: d[1], reverse=True) 
    [('item_price_level', 90),
 ('item_sales_level', 84),
 ('shop_score_delivery', 68),
 ('item_property_list', 53),
 ('item_category_list', 49),
 ('user_age_level', 45),
 ('user_star_level', 37),
 ('context_timestamp', 37),
 ('context_id', 23),
 ('shop_score_service', 20),
 ('shop_review_num_level', 19),
 ('item_city_id', 18),
 ('item_brand_id', 17),
 ('item_collected_level', 17),
 ('shop_review_positive_rate', 16),
 ('item_pv_level', 16),
 ('shop_id', 15),
 ('context_page_id', 15),
 ('shop_score_description', 10),
 ('user_occupation_id', 10),
 ('item_id', 9),
 ('user_gender_id', 7),
 ('user_id', 2),
 ('shop_star_level', 2)]    
    '''
    
    
    ## 2 模型融合
    if(os.path.exists('./result_fusion/')):
        os.rmdir('./result_fusion/')
    else:
        os.mkdir('./result_fusion/')
    
    #2.1 lightGBM建模，xgboost预测
    fusionModel(train_feature=train_feature, train_label=train_label, test_feature=test_feature,
                test_index=test_index, method1='lightGBM',method2='XGBoost',fnames=fnames) #0.0907792
    #0.09070716
    
    #2.2 xgboost建模，lightGBM预测
    fusionModel(train_feature=train_feature, train_label=train_label, test_feature=test_feature, 
                test_index=test_index, method1='XGBoost',method2='lightGBM',fnames=fnames) #0.0938532
    #0.09285086
    
    
    ## 3 多模型预测结果线性加权
    lightGBM = pd.read_csv('./result/submit_lightGBM.csv', header=0, sep=' ')
    xgboost = pd.read_csv('./result/submit_XGBoost.csv', header=0, sep=' ')
    lightGBM_xgboost = pd.read_csv('./result_fusion/submit_construct_lightGBM_predict_XGBoost.csv', header=0, sep=' ')
    xgboost_lightGBM = pd.read_csv('./result_fusion/submit_construct_XGBoost_predict_lightGBM.csv', header=0, sep=' ')
    
    result = 0.25 * xgboost.iloc[:,1] + 0.25 * lightGBM.iloc[:,1] + 0.4 * lightGBM_xgboost.iloc[:,1] + 0.1 * xgboost_lightGBM.iloc[:,1]
    result2 = pd.concat([lightGBM.iloc[:,0],pd.DataFrame(result)],axis=1)
    result2.to_csv(('./result/submit_integrate_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"), 
                index=False, index_label=None, header=['instance_id','predicted_score'], sep=' ')



if __name__=='__main__':
    print('analysis.py is starting!\n')
    
    main()
    
    print('analysis.py is end.')
    
   

