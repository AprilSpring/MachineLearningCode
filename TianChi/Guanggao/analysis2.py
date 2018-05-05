#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 23:23:57 2018

步骤：
特征提取和分析
特征融合
特征筛选
模型选择和调参
模型融合

参考代码：
ref: http://blog.csdn.net/haphapyear/article/details/75057407/

@author: tinghai
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
#from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
#from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#from scipy import sparse
from scipy.stats import ttest_ind
from scipy.stats import ranksums
import scipy as sp
import os
import datetime
import time
from collections import Counter




#%% 评估指标
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll



#%% 数据特征处理
def dataProcessing(path, train_name='round1_ijcai_18_train_20180301.txt', 
                   test_name='round1_ijcai_18_test_a_20180301.txt', 
                   choice_date=True, choose_date=['2018-09-23','2018-09-24'], 
                   featureExtract=True, featureDrop=True, missingValue=True):
    
    #path = '/Users/tinghai/Learning/GuanggaoData'
    os.chdir(path)
   
    #日期脱敏
    def time2cov(time_):
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))   
    
    
    #1 测试数据
    testFile = './data/' + test_name
    test_pd = pd.read_csv(testFile, header=0, sep=' ')
    test_pd['item_category_list'] = test_pd['item_category_list'].apply(lambda x: x.split(';')[1])
    test_pd['item_property_list'] = test_pd['item_property_list'].apply(lambda x: len(x.split(';')))
    test_pd['predict_category_property'] = test_pd['predict_category_property'].apply(lambda x: len(x.split(';')))
    test_pd['context_timestamp'] = test_pd['context_timestamp'].apply(time2cov)  
    test_pd['date'] = test_pd['context_timestamp'].apply(lambda x: x.split(' ')[0])
    test_pd['time'] = test_pd['context_timestamp'].apply(lambda x: x.split(' ')[1].split(':')[0])
    test_pd.index = test_pd.instance_id
    test_index = test_pd.index
    
    
    #2 训练数据
    trainFile = './data/' + train_name
    train_pd = pd.read_csv(trainFile, header=0, sep=' ')
    train_pd['item_category_list'] = train_pd['item_category_list'].apply(lambda x: x.split(';')[1])
    train_pd['item_property_list'] = train_pd['item_property_list'].apply(lambda x: len(x.split(';')))
    train_pd['predict_category_property'] = train_pd['predict_category_property'].apply(lambda x: len(x.split(';')))
    train_pd['context_timestamp'] = train_pd['context_timestamp'].apply(time2cov)
    train_pd['date'] = train_pd['context_timestamp'].apply(lambda x: x.split(' ')[0])
    train_pd['time'] = train_pd['context_timestamp'].apply(lambda x: x.split(' ')[1].split(':')[0])
    train_pd.index = train_pd.instance_id
    train_index = train_pd.index
    
    #选择既定日期用于训练集
    if choice_date == True:
        print('Test set date is: ' + str(Counter(test_pd['date']))) #Counter({'2018-09-25': 18371})
        #Counter(train_pd['date'])
        train_pd = train_pd[train_pd['date'].isin(choose_date)]
        #train = train[train['context_timestamp']<='2018-09-21 23:59:59']
    
    
    #数据其他特征提取
    if featureExtract == True:
        #1 用户特征：浏览广告的总次数（点击率）、用户浏览过的商家数目、用户浏览了几种广告、
        #        用户对某广告的浏览次数、
        #        用户浏览最多的商家ID、用户浏览最多的广告ID、
        #        用户浏览的频繁程度（总次数/总时间）、用户成功转化率、用户转化成功的商家数目、
        #2 商家特征：被浏览的广告总次数、拥有广告的数目、被浏览的用户数目、
        #        转化成功的用户数目、
        #3 广告特征：被浏览的总次数、被多少个用户浏览、属于多少个商家（一般是1个）、
        #       转化成功的概率、
        #4 用户-商家行为特征：针对既定用户和商家
        print('Extracting other feature ... \n')
        
        def userFeature(data, user_list):
            user_feature1 = pd.DataFrame()
            user_feature1['user_id'] = Counter(data.user_id).keys()
            user_feature1['user_itemViewCount'] = Counter(data.user_id).values() 
            user_feature1 = user_feature1.iloc[np.argsort(user_feature1['user_id']),:] #每个用户浏览广告的总次数（点击率）
            
            user_feature2 = pd.DataFrame()
            user_shop = data[['user_id','shop_id']].drop_duplicates()
            user_feature2['user_id'] = Counter(user_shop.user_id).keys()
            user_feature2['user_shopCount'] = Counter(user_shop.user_id).values()
            user_feature2 = user_feature2.iloc[np.argsort(user_feature2['user_id']),:] #每个用户浏览过的商家数目
            
            user_feature3 = pd.DataFrame()
            user_item = data[['user_id','item_id']].drop_duplicates()
            user_feature3['user_id'] = Counter(user_item.user_id).keys()
            user_feature3['user_itemCount'] = Counter(user_item.user_id).values()
            user_feature3 = user_feature3.iloc[np.argsort(user_feature3['user_id']),:] #每个用户浏览过几种广告
            
            user_feature = pd.concat([user_feature1['user_itemViewCount'],user_feature2['user_shopCount'],user_feature3['user_itemCount']],axis=1)
            user_feature.index = user_feature1['user_id']
            
            del user_feature1
            del user_feature2
            del user_feature3
            
            return user_feature
            
        
        def shopFeature(data, shop_list):       
            shop_feature1 = pd.DataFrame()
            shop_feature1['shop_id'] = Counter(data.shop_id).keys()
            shop_feature1['shop_itemViewCount'] = Counter(data.shop_id).values() 
            shop_feature1 = shop_feature1.iloc[np.argsort(shop_feature1['shop_id']),:] #每个商家被浏览的广告总次数
            
            shop_feature2 = pd.DataFrame()
            shop_item = data[['shop_id','item_id']].drop_duplicates()
            shop_feature2['shop_id'] = Counter(shop_item.shop_id).keys()
            shop_feature2['shop_itemCount'] = Counter(shop_item.shop_id).values() 
            shop_feature2 = shop_feature2.iloc[np.argsort(shop_feature2['shop_id']),:] #每个商家拥有的广告数目
            
            shop_feature3 = pd.DataFrame()
            shop_user = data[['shop_id','user_id']].drop_duplicates()
            shop_feature3['shop_id'] = Counter(shop_user.shop_id).keys()
            shop_feature3['shop_userCount'] = Counter(shop_user.shop_id).values() 
            shop_feature3 = shop_feature3.iloc[np.argsort(shop_feature3['shop_id']),:] #每个商家被浏览的用户数目
            
            shop_feature = pd.concat([shop_feature1['shop_itemViewCount'],shop_feature2['shop_itemCount'],shop_feature3['shop_userCount']],axis=1)
            shop_feature.index = shop_feature1['shop_id']
            
            del shop_feature1
            del shop_feature2
            del shop_feature3
            
            return shop_feature
        
        
        def itemFeature(data, item_list): 
            item_feature1 = pd.DataFrame()
            item_feature1['item_id'] = Counter(data.item_id).keys()
            item_feature1['item_ViewCount'] = Counter(data.item_id).values()
            item_feature1 = item_feature1.iloc[np.argsort(item_feature1['item_id']),:] #每个广告浏览的总次数
            
            item_feature2 = pd.DataFrame()
            item_user = data[['item_id','user_id']].drop_duplicates()
            item_feature2['item_id'] = Counter(item_user.item_id).keys()
            item_feature2['item_userCount'] = Counter(item_user.item_id).values()
            item_feature2 = item_feature2.iloc[np.argsort(item_feature2['item_id']),:] #每个广告被多少个用户浏览
            
            item_feature = pd.concat([item_feature1['item_ViewCount'],item_feature2['item_userCount']],axis=1)
            item_feature.index = item_feature1['item_id']
            
            del item_feature1
            del item_feature2
            
            return item_feature
        
        #整合特征矩阵
        def integrate_feature(data, user_feature, shop_feature, item_feature):
            
            ff = list()
            for j in range(0,data.shape[0]):
                
                if j%1000 == 0:
                    print(j)
                
                u = data.user_id.iloc[j]
                s = data.shop_id.iloc[j]
                i = data.item_id.iloc[j]
                
                f = [user_feature.at[u, 'user_itemViewCount'], user_feature.at[u, 'user_shopCount'], 
                     user_feature.at[u, 'user_itemCount'], shop_feature.at[s,'shop_itemViewCount'], 
                       shop_feature.at[s, 'shop_itemCount'], shop_feature.at[s, 'shop_userCount'], 
                       item_feature.at[i, 'item_ViewCount'], item_feature.at[i, 'item_userCount']]
                
                ff.append(f)         
                
            ff = pd.DataFrame(ff, index=data.index, columns=['user_itemViewCount','user_shopCount','user_itemCount',
                                                            'shop_itemViewCount','shop_itemCount','shop_userCount',
                                                            'item_ViewCount','item_userCount'])
            inte_feature = pd.concat([data, ff], axis=1)
            inte_feature.index = data.index
            return inte_feature
        
        
        #test与train用户的交叠情况
        test_user = set(test_pd.user_id) #13573
        train_user = set(train_pd.user_id) #197694
        
        #test与train商家的交叠情况
        test_shop = set(test_pd.shop_id) #2015
        train_shop = set(train_pd.shop_id) #3959
        #inter_shop = test_shop.intersection(train_shop) #1971
        
        #test与train广告的交叠情况
        test_item = set(test_pd.item_id) #3695
        train_item = set(train_pd.item_id) #10075
        
        #函数调用
        test_user_feature = userFeature(data=test_pd, user_list=test_user) 
        test_shop_feature = shopFeature(data=test_pd, shop_list=test_shop) 
        test_item_feature = itemFeature(data=test_pd, item_list=test_item) 
        
        test_integate = integrate_feature(data=test_pd, user_feature=test_user_feature, 
                                      shop_feature=test_shop_feature, item_feature=test_item_feature) 
        
        train_user_feature = userFeature(data=train_pd, user_list=train_user) 
        train_shop_feature = shopFeature(data=train_pd, shop_list=train_shop) 
        train_item_feature = itemFeature(data=train_pd, item_list=train_item)     
        
        train_user_feature = train_user_feature/len(Counter(train_pd['date']))
        train_shop_feature = train_shop_feature/len(Counter(train_pd['date']))
        train_item_feature = train_item_feature/len(Counter(train_pd['date']))      
     
        train_integate = integrate_feature(data=train_pd, user_feature=train_user_feature, 
                                      shop_feature=train_shop_feature, item_feature=train_item_feature)           
 
        test_pd = test_integate
        train_pd = train_integate
        
        print('New feature extract done! \n')    
        
    
    #特征删除
    if featureDrop == True:
        
        #特征类型转换，例如item_id=2275895163219263378，numpy.int64 -> str
        '''
        #特征类型：
        instance_id: <class 'numpy.int64'> c,delete
        item_id: <class 'str'> c
        item_category_list: <class 'str'> 
        item_property_list: <class 'numpy.int64'> p
        item_brand_id: <class 'numpy.int64'> c
        item_city_id: <class 'numpy.int64'> c
        item_price_level: <class 'numpy.int64'> p
        item_sales_level: <class 'numpy.int64'> p
        item_collected_level: <class 'numpy.int64'> p
        item_pv_level: <class 'numpy.int64'> p
        user_id: <class 'numpy.int64'> c
        user_gender_id: <class 'numpy.int64'> c,p
        user_age_level: <class 'numpy.int64'> p
        user_occupation_id: <class 'numpy.int64'> c,p
        user_star_level: <class 'numpy.int64'> c,p
        context_id: <class 'numpy.int64'> c
        context_timestamp: <class 'str'> delete
        context_page_id: <class 'numpy.int64'> c
        predict_category_property: <class 'numpy.int64'> p
        shop_id: <class 'numpy.int64'> c
        shop_review_num_level: <class 'numpy.int64'> p
        shop_review_positive_rate: <class 'numpy.float64'> p
        shop_star_level: <class 'numpy.int64'> c
        shop_score_service: <class 'numpy.float64'> p
        shop_score_delivery: <class 'numpy.float64'> p
        shop_score_description: <class 'numpy.float64'> p
        date: <class 'str'> delete
        time: <class 'str'> c int, p
        
        新增特征：p
        user_itemViewCount
        user_shopCount
        user_itemCount
        shop_itemViewCount
        shop_itemCount
        shop_userCount
        item_ViewCount
        item_userCount
        '''
    
        def type_conversion(data):
            data['instance_id'] = data['instance_id'].astype(str)
            data['item_id'] = data['item_id'].astype(str)
            data['item_brand_id'] = data['item_brand_id'].astype(str)
            data['item_city_id'] = data['item_city_id'].astype(str)
            data['user_id'] = data['user_id'].astype(str)
            data['user_gender_id'] = data['user_gender_id'].astype(str)
            data['user_star_level'] = data['user_star_level'].astype(str)
            data['context_id'] = data['context_id'].astype(str)
            data['context_page_id'] = data['context_page_id'].astype(str)
            data['shop_id'] = data['shop_id'].astype(str)
            data['shop_star_level'] = data['shop_star_level'].astype(str)
            data['time'] = data['time'].astype(int)
            return data
        
        
        #1、删除线上线下差异较大的特征
        test_pd = type_conversion(test_pd)
        train_pd = type_conversion(train_pd)
        
        f1_list = ['item_property_list','item_price_level','item_sales_level',
                 'item_collected_level','item_pv_level','user_age_level',
                 'predict_category_property','shop_review_num_level',
                 'shop_review_positive_rate','shop_score_service',
                 'shop_score_delivery','shop_score_description','time']
        if featureExtract == True:
            f1_list = f1_list + ['user_itemViewCount','user_shopCount','user_itemCount',
                            'shop_itemViewCount','shop_itemCount','shop_userCount',
                            'item_ViewCount','item_userCount']
        f2_list = ['user_gender_id','user_occupation_id','user_star_level']
        f_list = f1_list + f2_list
        
        p = []      
        for i in f1_list:
            p.append(ttest_ind(train_pd[i],test_pd[i]).pvalue) #连续特征
        for ii in f2_list:
            p.append(ranksums(train_pd[ii],test_pd[ii]).pvalue) #离散特征
        
        tmp1 = pd.DataFrame(p,index=f_list) 
        tmp2 = tmp1[tmp1.iloc[:,0] < 0.05]
        drop_f1 = list(tmp2.index)
        
        
        #2、通过xgboost计算的特征重要性，删除重要性较低的特征；
        drop_f2 = ['user_id','user_gender_id','context_page_id','user_occupation_id']
        
        drop_f = list(set(drop_f1 + drop_f2))
        test_pd = test_pd.drop(drop_f,axis=1)
        train_pd = train_pd.drop(drop_f,axis=1)
        
        
        #3、通过wrapper的方式选择特征？？
        
        print('Feature drop is done! \n')
        
    
    '''
    if leakeage==True:
        inter_user = test_user.intersection(train_user) #3626
        inter_item = test_item.intersection(train_item) #3534
        
        train_pos = train_pd[train_pd.is_trade.isin([1])] #9021
        train_neg = train_pd[train_pd.is_trade.isin([0])] #469117
        
        train_pos_user = set(train_pos.user_id).intersection(inter_user) #97
        train_neg_user = set(train_neg.user_id).intersection(inter_user) #3613
        
        train_pos_item = set(train_pos.item_id).intersection(inter_item) #1716
        train_neg_item = set(train_neg.item_id).intersection(inter_item) #3530
        
        test_leakeage = []
        for uu in train_pos_user:
            for ii in train_pos_item:
                tmp = test_pd[test_pd['user_id'].isin([uu]) & test_pd['item_id'].isin([ii])]    
                test_leakeage.append(tmp)
        test_leakeage = pd.DataFrame(test_leakeage) #0
    '''
    
    #将数据集中标记为“-1”的值，改为该列中值(int)或频数最高的值(str)
    if missingValue == True:
        
        def missValue(data):
            for i in data.columns:
                if type(data[i].iloc[0]) == np.int64:
                    data[i][data[i].isin([-1])] = np.median(data[i])
                if type(data[i].iloc[0]) == int:
                    data[i][data[i].isin([-1])] = np.median(data[i])
                if type(data[i].iloc[0]) == str:   
                    data[i][data[i].isin(['-1'])] = max(Counter(data[i]))
            return data
    
        test_pd = missValue(test_pd)
        train_pd = missValue(train_pd) 
    
    
    #去掉不加入的特征
    test_feature = test_pd.drop(['context_timestamp','date','instance_id'],axis=1).values
    fnames = train_pd.drop(['context_timestamp','date', 'instance_id','is_trade'],axis=1).columns
    train_feature = train_pd.drop(['context_timestamp','date','instance_id','is_trade'],axis=1).values
    train_label = train_pd.is_trade.values
    #test_pd.to_csv(('./data/test_pd_integate.csv'),index=True, index_label='instance_id', header=test_pd.colums, sep=' ')
    #train_pd.to_csv(('./data/train_pd_integate.csv'),index=True, index_label='instance_id', header=train_pd.colums, sep=' ')
    
    print('Feature extract is over! \n')
    print('============================================================\n')
    
    return train_pd, train_feature, train_label, train_index, test_pd, test_feature, test_index, fnames 
    

#%% 训练集与测试集划分
def dataSplit(train_pd,train_feature,train_label,f_names=None,date_split=True,
              train_d=['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22'],
              test_d=['2018-09-23']):
    
    if date_split == True:
        print('按照日期划分训练集和测试集...\n')
        #按照日期划分训练集和测试集，否则可能存在数据泄漏，导致线上、线下预测结果差异     
        x_train = train_feature[train_pd['date'].isin(train_d)]
        y_train = train_label[train_pd['date'].isin(train_d)]
        x_test = train_feature[train_pd['date'].isin(test_d)]
        y_test = train_label[train_pd['date'].isin(test_d)]
    else:
        print('随机划分训练集和测试集...\n')
        test_size=0.2
        random_state=10
        x_train,x_test,y_train,y_test = train_test_split(train_feature,train_label,test_size=test_size,random_state=random_state)
    
    return x_train,x_test,y_train,y_test



#%%单独模型
#1 lightGBM模型和预测（need parameterAdjust.py进行调参）
def trainClassifierLGBM(x_train,y_train,x_test,y_test,test_feature,test_index,f_names=None):
    
            
    print('LIGHTBGM start...\n')
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_test = lgb.Dataset(x_test,y_test)
    
    #Best parameter is { max_depth: 3, eta: 0.03, num_leaves: 100, num_boost_round:200 }
    params = {'task': 'train','boosting_type': 'gbdt',
        'objective': 'binary','metric': 'binary_logloss',
        'max_depth':3, 'num_leaves': 100, 'max_bin':150,  
        'learning_rate': 0.03, 'feature_fraction': 0.5,
        'bagging_fraction': 0.85,'bagging_freq': 5, 'verbose': 0, 
        'scale_pos_weight':1}
    
    watchlist=[lgb_test]
    lgbm = lgb.train(params=params, train_set=lgb_train, num_boost_round=200,
                     valid_sets=watchlist, verbose_eval=10)
    best_score = list(list(lgbm.best_score.values())[0].values())[0]
    print('lgbm.best_score: ' + str(best_score) + '\n')
    print('lgbm.best_iteration: ' + str(lgbm.best_iteration) + '\n')
    
    feature_import = dict(zip(list(f_names),list(lgbm.feature_importance())))
    feature_import = sorted(feature_import.items(), key=lambda d: d[1], reverse=True) 
    print('lgbm.feature_import: ', str(feature_import) + '\n')
    
    prob = lgbm.predict(test_feature, num_iteration=lgbm.best_iteration)
    prob = pd.DataFrame(prob, index=test_index, columns=['predicted_score'])
    #prob.to_csv(('./result/submit_lightGBM.csv'), index=True, 
    #            index_label='instance_id', header=['predicted_score'], sep=' ')
    return lgbm, prob, best_score


#2 xgboost建模和预测（need parameterAdjust.py进行调参）
def trainClassifierXGB(x_train,y_train,x_test,y_test,test_feature,test_index,f_names=None,buffer=False):
    
    print('XGBOOST start...\n')
    if buffer==True:
        dtrain=xgb.DMatrix("data/train.buffer")
        dval=xgb.DMatrix("data/val.buffer")
    else:
        dtrain=xgb.DMatrix(x_train,label=y_train,feature_names=f_names)
        dval=xgb.DMatrix(x_test,label=y_test,feature_names=f_names)

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
    #verbose_eval=10 每10次迭代输出一次结果
    
    #Best parameter is { max_depth: 3, eta: 0.1, num_round: 100, lambda: 100 }
    param = {'max_depth':3,'eta':0.1,'min_child_weight':1, 
             'silent':1, 'subsample':1,'colsample_bytree':1,
             'gamma':0,'scale_pos_weight':1,'lambda':10,
             'objective':'binary:logistic'}
    #param['nthread'] = 8  #不用设置线程，XGboost会自行设置所有的
    plst = list(param.items())
    plst += [('eval_metric', 'logloss')]
    #plst += [('eval_metric', 'roc')]  #Multiple evals can be handled in this way 
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params=plst,dtrain=dtrain,num_boost_round=100,evals=watchlist, 
                    early_stopping_rounds=200,verbose_eval=10)
    best_score = bst.best_score
    print('bst.best_score: ' + str(best_score))
    print('bst.best_iteration: ' + str(bst.best_iteration))
    print('bst.best_ntree_limit: ' + str(bst.best_ntree_limit))
    
    feature_import = sorted(bst.get_score().items(), key=lambda d: d[1], reverse=True) 
    print('bst.feature_import: ' + str(feature_import)) 
    
    test_feature = xgb.DMatrix(test_feature,feature_names=f_names)
    prob = bst.predict(test_feature,ntree_limit=bst.best_ntree_limit)
    prob = pd.DataFrame(prob, index=test_index, columns=['predicted_score'])
    #prob.to_csv(('./result/submit_XGBoost.csv'), index=True, 
    #            index_label='instance_id', header=['predicted_score'], sep=' ')
    return bst, prob, best_score
    

'''
#交叉验证
def cross_validation(train_pd, train_feature, train_label, method, fnames, date_split,
                     train_d=['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22'],
                     test_d = ['2018-09-23']):
    
    if date_split == True:
        print('基于日期的训练集和测试集划分...\n')
        #按照日期划分训练集和测试集，否则可能存在数据泄漏，导致线上、线下预测结果差异
        #cv-train：18-21号
        #cv-test：22-23号       
        x_train = train_feature[train_pd['date'].isin(train_d)]
        y_train = train_label[train_pd['date'].isin(train_d)]
        x_test = train_feature[train_pd['date'].isin(test_d)]
        y_test = train_label[train_pd['date'].isin(test_d)]
    else:
        print('训练集和测试集划分...\n')
        test_size=0.2
        random_state=10
        x_train,x_test,y_train,y_test = train_test_split(train_feature,train_label,test_size=test_size,random_state=random_state)
        
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
    
    return x_train, y_train, x_test, y_test, loss


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
'''    

#%% 模型融合
#交叉验证
def fiveFoldCV(train_pd,train_feature,train_label,test_feature,test_index,method,f_names,date_split=True):       
    
    train_pd_add = pd.DataFrame()
    test_pred = pd.DataFrame()
    loss = list()
    
    if method == 'lightGBM': 
        if date_split == True:
            d = ['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22','2018-09-23','2018-09-24']
            for i in range(0,5):
                [x_train1,x_test1,y_train1,y_test1] = dataSplit(
                        train_pd=train_pd,train_feature=train_feature,train_label=train_label,
                        f_names=f_names,date_split=True,train_d=d[i:i+2],test_d=d[i+3])
                
                #training set: 4/5 train data
                #test set: 1/5 train data
                [lgbm,test1_prob,best_score] = trainClassifierLGBM(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=x_test1, test_index=None, f_names=f_names)
                tmp = pd.concat([pd.DataFrame(x_test1),pd.DataFrame(test1_prob),pd.DataFrame(y_test1)],axis=1)
                train_pd_add = pd.concat([train_pd_add,tmp], axis=0) #不包括原始训练集中前2天的数据哦！
                            
                #training set: 4/5 train data
                #test set: test data
                [lgbm,test_prob,best_score] = trainClassifierLGBM(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=test_feature, test_index=test_index, f_names=f_names)
                
                test_pred = pd.concat([test_pred,pd.DataFrame(test_prob)],axis=1)
                loss.append(best_score)
                #model = lgbm
                
        else:
            skf = list(StratifiedKFold(train_label, 5)) #1/5 data: skf[i][1], 4/5 data: skf[i][0]
            for i in range(0,5):
                print(method + ': fold_' + str(i+1))
                x_train1 = train_feature[skf[i][0]]
                y_train1 = train_label[skf[i][0]]
                x_test1 = train_feature[skf[i][1]]
                y_test1 = train_label[skf[i][1]]
                            
                #training set: 4/5 train data
                #test set: 1/5 train data
                [lgbm,test1_prob,best_score] = trainClassifierLGBM(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=x_test1, test_index=None, f_names=f_names)
                tmp = pd.concat([pd.DataFrame(x_test1),pd.DataFrame(test1_prob),pd.DataFrame(y_test1)],axis=1)
                train_pd_add = pd.concat([train_pd_add,tmp], axis=0)
                            
                #training set: 4/5 train data
                #test set: test data
                [lgbm,test_prob,best_score] = trainClassifierLGBM(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=test_feature, test_index=test_index, f_names=f_names)
                
                test_pred = pd.concat([test_pred,pd.DataFrame(test_prob)],axis=1)
                loss.append(best_score)
                #model = lgbm
                
    if method == 'XGBoost':
        if date_split == True:
            d = ['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22','2018-09-23','2018-09-24']
            for i in range(0,5):
                [x_train1,x_test1,y_train1,y_test1] = dataSplit(
                        train_pd=train_pd,train_feature=train_feature,train_label=train_label,
                        f_names=f_names,date_split=True,train_d=d[i:i+2],test_d=d[i+3])
                
                #training set: 4/5 train data
                #test set: 1/5 train data
                [bst,test1_prob,best_score] = trainClassifierXGB(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=x_test1,test_index=None,f_names=f_names,buffer=False)
                tmp = pd.concat([pd.DataFrame(x_test1),pd.DataFrame(test1_prob),pd.DataFrame(y_test1)],axis=1)
                train_pd_add = pd.concat([train_pd_add,tmp], axis=0)
                
                #training set: 4/5 train data
                #test set: test data
                [bst,test_prob,best_score] = trainClassifierXGB(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=test_feature,test_index=test_index,f_names=f_names,buffer=False)
                
                test_pred = pd.concat([test_pred,pd.DataFrame(test_prob)],axis=1)
                loss.append(best_score)

        else:
            skf = list(StratifiedKFold(train_label, 5)) #1/5 data: skf[i][1], 4/5 data: skf[i][0]
            for i in range(0,5):
                print(method + ': fold_' + str(i+1))
                x_train1 = train_feature[skf[i][0]]
                y_train1 = train_label[skf[i][0]]
                x_test1 = train_feature[skf[i][1]]
                y_test1 = train_label[skf[i][1]]
            
                #training set: 4/5 train data
                #test set: 1/5 train data
                [bst,test1_prob,best_score] = trainClassifierXGB(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=x_test1,test_index=None,f_names=f_names,buffer=False)
                tmp = pd.concat([pd.DataFrame(x_test1),pd.DataFrame(test1_prob),pd.DataFrame(y_test1)],axis=1)
                train_pd_add = pd.concat([train_pd_add,tmp], axis=0)
                
                #training set: 4/5 train data
                #test set: test data
                [bst,test_prob,best_score] = trainClassifierXGB(
                        x_train=x_train1,y_train=y_train1,x_test=x_test1,y_test=y_test1,
                        test_feature=test_feature,test_index=test_index,f_names=f_names,buffer=False)
                
                test_pred = pd.concat([test_pred,pd.DataFrame(test_prob)],axis=1)
                loss.append(best_score)
                #model = bst
                    
    columns=list(f_names.values)
    columns.append('predicted_score')
    test_feature_add = pd.concat([pd.DataFrame(test_feature), pd.DataFrame(np.mean(test_pred, axis=1))],axis=1) #按行求均值
    test_feature_add.to_csv(('./result_fusion/' + method + '_test_feature_add.csv'), index=False, header=columns, sep=' ')
    
    columns.append('is_trade')
    train_pd_add.to_csv(('./result_fusion/' + method + '_train_pd_add.csv'), index=False, header=columns, sep=' ')
    
    loss_mean = np.mean(loss)
    print('Average loss is ' + str(loss_mean))
    #return model


'''
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
'''    

#模型2预测模型1的新特征
def fusionModel(train_pd,train_feature,train_label,test_feature,
                test_index,method1,method2,f_names,date_split):
    
    ##1 method1建模
    #通过交叉验证，产生train set和test set的预测值特征 
    fiveFoldCV(train_pd=train_pd,train_feature=train_feature,train_label=train_label,
               test_feature=test_feature,test_index=test_index,method=method1,f_names=f_names,
               date_split=True)    
    
    #导入添加预测值的train set
    trainFile_add = './result_fusion/' + method1 + '_train_pd_add.csv'
    train_pd_add = pd.read_csv(trainFile_add, header=0, sep=' ')
    fnames_add = train_pd_add.drop('is_trade',axis=1).columns
    train_feature_add = train_pd_add.drop('is_trade',axis=1).values
    train_label_add = train_pd_add.is_trade.values
    
    #导入添加预测值的test set
    testFile_add = './result_fusion/'+ method1 + 'M_test_feature_add.csv'
    test_pd_add = pd.read_csv(testFile_add, header=0, sep=' ')
    test_feature_add = test_pd_add.values
    
    #重新划分数据集
    #if date_split is true, train_d not include '2018-09-18' and '2018-09-19'
    [x_train,x_test,y_train,y_test] = dataSplit(
            train_pd=train_pd_add,train_feature=train_feature_add,train_label=train_label_add,
            f_names=None,date_split=True,
            train_d=['2018-09-20','2018-09-21','2018-09-22'],
            test_d=['2018-09-23','2018-09-24'])
    
    ##2 method2预测
    if method2 =='XGBoost':
        [bst,bst_prob,bst_best_score] = trainClassifierXGB(
                x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                test_feature=test_feature_add,test_index=test_index,f_names=fnames_add,buffer=False)
        model = bst
        prob = bst_prob
        #best_score = bst_best_score
    if method2 =='lightGBM':
        [lgbm,lgbm_prob,lgbm_best_score] = trainClassifierLGBM(
                x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                test_feature=test_feature_add,test_index=test_index,f_names=fnames_add)
        model = bst
        prob = bst_prob
        #best_score = bst_best_score
        
    prob = pd.DataFrame(prob, index=test_index, columns=['predicted_score'])
    prob.to_csv(('./result_fusion/submit_construct_' + method1 + '_predict_' + method2 + '.csv'), 
                index=True, index_label='instance_id', header=['predicted_score'], sep=' ')
    
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




#%% 主函数
def main():
    
    path = '/Users/tinghai/Learning/TianChi/GuanggaoData'
    os.chdir(path)
    
    ##1 特征提取   
    [train_pd, train_feature, train_label, train_index, test_pd, test_feature, test_index, fnames] = dataProcessing(
            path=path, train_name='round1_ijcai_18_train_20180301.txt', test_name='round1_ijcai_18_test_a_20180301.txt',
            choice_date=True, choose_date=['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22','2018-09-23','2018-09-24'], 
            featureExtract=False, featureDrop=False, missingValue=False)
    
    ##2 数据划分
    [x_train,x_test,y_train,y_test] = dataSplit(
            train_pd=train_pd,train_feature=train_feature,train_label=train_label,
            f_names=None,date_split=True,
            train_d=['2018-09-18','2018-09-19','2018-09-20','2018-09-21','2018-09-22'],
            test_d=['2018-09-23'])
    
    ##3 单独模型预测
    if(os.path.exists('./result/')):
        os.rmdir('./result/')
    else:
        os.mkdir('./result/')
    
    [lgbm,lgbm_prob,lgbm_best_score] = trainClassifierLGBM(
            x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
            test_feature=test_feature,test_index=test_index,f_names=fnames)
    lgbm_prob.to_csv(('./result/submit_XGBoost.csv'), index=True, index_label='instance_id', header=['predicted_score'], sep=' ')

    [bst,bst_prob,bst_best_score] = trainClassifierXGB(
            x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
            test_feature=test_feature,test_index=test_index,f_names=fnames,buffer=False)
    bst_prob.to_csv(('./result/submit_XGBoost.csv'), index=True, index_label='instance_id', header=['predicted_score'], sep=' ')
    
    
    ##4 模型融合
    if(os.path.exists('./result_fusion/')):
        os.rmdir('./result_fusion/')
    else:
        os.mkdir('./result_fusion/')
    
    #3.1 lightGBM建模，xgboost预测
    fusionModel(train_pd=train_pd,train_feature=train_feature,train_label=train_label,
                test_feature=test_feature,test_index=test_index,method1='lightGBM',method2='XGBoost',
                f_names=fnames,date_split=True)
    
    fusionModel(train_pd=train_pd,train_feature=train_feature,train_label=train_label,
                test_feature=test_feature,test_index=test_index,method1='XGBoost',method2='lightGBM',
                f_names=fnames,date_split=True)
    
    
    ##5 多模型预测结果线性加权
    lightGBM = pd.read_csv('./result/submit_lightGBM.csv', header=0, sep=' ')
    xgboost = pd.read_csv('./result/submit_XGBoost.csv', header=0, sep=' ')
    lightGBM_xgboost = pd.read_csv('./result_fusion/submit_construct_lightGBM_predict_XGBoost.csv', header=0, sep=' ')
    xgboost_lightGBM = pd.read_csv('./result_fusion/submit_construct_XGBoost_predict_lightGBM.csv', header=0, sep=' ')
    
    result = 0.3 * xgboost.iloc[:,1] + 0.3 * lightGBM.iloc[:,1] + 0.2 * lightGBM_xgboost.iloc[:,1] + 0.2 * xgboost_lightGBM.iloc[:,1]
    result2 = pd.concat([lightGBM.iloc[:,0],pd.DataFrame(result)],axis=1)
    result2.to_csv(('./result/submit_integrate_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"), 
                index=False, index_label=None, header=['instance_id','predicted_score'], sep=' ')

    
    
    
    
    '''
    #2.1 交叉验证，计算loss
    print('交叉验证...\n')
    loss_lgbm = cross_validation(train_pd=train_pd, train_feature=train_feature, 
                                 train_label=train_label, method='lightGBM', 
                                 fnames=fnames, date_split=True,
                                 train_d=['2018-09-18','2018-09-19'],
                                 test_d = ['2018-09-20'])
    print(loss_lgbm) 
    #0.08775840 0.08491805 
    
    loss_bst = cross_validation(train_pd=train_pd, train_feature=train_feature, 
                                train_label=train_label, method='XGBoost', 
                                fnames=fnames, date_split=True,
                                train_d=['2018-09-18','2018-09-19','2018-09-20','2018-09-21'],
                                test_d = ['2018-09-22','2018-09-23'])
    print(loss_bst) 
    #0.087036403 0.08429225
    
    #2.2 全量训练样本建模和预测
    predict_test_prob(train_feature=train_feature, train_label=train_label, 
                      test_feature=test_feature, test_index=test_index, fnames=fnames, method='lightGBM')
    predict_test_prob(train_feature=train_feature, train_label=train_label, 
                      test_feature=test_feature, test_index=test_index,fnames=fnames, method='XGBoost')
    
    #2.3 根据日期筛选训练集用于建模和预测
    train_d = ['2018-09-22','2018-09-23'] #'2018-09-18','2018-09-19','2018-09-20','2018-09-21',
    train_feature2 = train_feature[train_pd['date'].isin(train_d)]
    train_label2 = train_label[train_pd['date'].isin(train_d)]
    predict_test_prob(train_feature=train_feature2, train_label=train_label2, 
                      test_feature=test_feature, test_index=test_index, fnames=fnames, method='lightGBM')
    predict_test_prob(train_feature=train_feature2, train_label=train_label2, 
                      test_feature=test_feature, test_index=test_index,fnames=fnames, method='XGBoost')
    '''
    
    
    '''
    #其他考量内容：
    
    #1 样本不均衡
    #y0 = train_pd.is_trade[train_pd.is_trade.isin([0])]
    #y1 = train_pd.is_trade[train_pd.is_trade.isin([1])]
    
    #2 onehot编码成稀疏矩阵
    
    
    #3 删除不重要特征
    feature_import = sorted(bst.get_score().items(), key=lambda d: d[1], reverse=True) 
    [('item_price_level', 74),
     ('item_sales_level', 74),
     ('shop_score_delivery', 67),
     ('user_itemCount', 66),
     ('user_age_level', 44),
     ('time', 38),
     ('item_category_list', 35),
     ('shop_itemCount', 31),
     ('user_shopCount', 30),
     ('context_id', 28),
     ('user_star_level', 26),
     ('item_property_list', 21),
     ('shop_review_positive_rate', 15),
     ('shop_itemViewCount', 12),
     ('item_ViewCount', 12),
     ('shop_score_description', 11),
     ('shop_review_num_level', 11),
     ('item_brand_id', 10),
     ('shop_score_service', 9),
     ('item_collected_level', 8),
     ('user_itemViewCount', 8),
     ('context_page_id', 8),
     ('item_id', 6),
     ('item_city_id', 6),
     ('shop_userCount', 4), -
     ('user_gender_id', 4), 
     ('predict_category_property', 3), -
     ('shop_id', 3), -
     ('user_occupation_id', 3), -
     ('item_pv_level', 3), -
     ('user_id', 2), -
     ('item_userCount', 1)] -
    '''
    
    
    #统计trainset中每天的转化率，分析是否有特殊日期
    for k in Counter(train_pd['date']).keys():
        tmp = Counter(train_pd[train_pd['date'] == k]['is_trade'])
        tmp2 = list(tmp.values())[1] / list(tmp.values())[0]
        print(k + ': ' + str(tmp2))
    
    #比较预测结果与cv中20号的预测结果差异
    result = pd.read_csv('./result/submit_integrate_20180322_223645.txt', header=0, sep=' ')
    
    


if __name__=='__main__':
    print('analysis.py is starting!\n')
    
    main()
    
    print('analysis.py is end.')
    
   

