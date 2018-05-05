# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:41:09 2018

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os


#path='D:/Learn/DataAnalyst/TianWenData/'
path='/Users/tinghai/Learning/TianWenData/'


#模型参数
#按标签比例抽样
#十倍交叉验证评估模型性能
#随机森林的特征重要性？
#数据不需要归一化？缺失值？
#F-score计算
#GBDT和Xgboost模型调参？



#%% training set extract

def X_format(path,label_name,random_state):
    label = pd.read_csv(path + label_name + '.csv', header=0, names=['id','type'], sep=',') #有列名
    #label = label.iloc[m:n,:]
    #label.iloc[:,1].describe()
    
    #按照各组比例分层抽样
    star=label[label.iloc[:,1].isin(['star'])]
    galaxy=label[label.iloc[:,1].isin(['galaxy'])]
    qso=label[label.iloc[:,1].isin(['qso'])]
    unknown=label[label.iloc[:,1].isin(['unknown'])]
    all=label.shape[0]
    weights=[star.shape[0]/all, galaxy.shape[0]/all, qso.shape[0]/all, unknown.shape[0]/all] #向下采样：star取0.1
    print(weights)
    star_sub = star.sample(n=None, frac=0.025, replace=False, weights=None, random_state=random_state, axis=0)
    galaxy_sub = galaxy.sample(n=None, frac=1, replace=False, weights=None, random_state=random_state, axis=0) #copy 2 times
    qso_sub = qso.sample(n=None, frac=1, replace=False, weights=None, random_state=1, axis=0) #copy 8 times
    unknown_sub = unknown.sample(n=None, frac=0.3, replace=False, weights=None, random_state=random_state, axis=0)
    
    label_sub = pd.DataFrame(columns=['id','type'])
    label_sub = label_sub.append(star_sub)
    label_sub = label_sub.append(galaxy_sub)
    label_sub = label_sub.append(galaxy_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(qso_sub)
    label_sub = label_sub.append(unknown_sub)
    
    y = label_sub.iloc[:,1] #1.4w+, id is label_sub.iloc[:,0]
    files = pd.DataFrame(index=label_sub.iloc[:,0], columns=range(0,1))
    for i in range(0,len(y.index)):
        files.iloc[i,0] = str(label_sub.iloc[i,0]) + '.txt'
    files.to_csv(path + 'subtrain_label_4w_times.csv', index=False, index_label=None, header=None)
    
    #提取subtrain_label.csv中对应的文件信息，shell下形成subtrain.txt文件
    X = pd.read_csv(path + label_name + '/subtrain_4w_times.txt', header=None, sep=',')
    
    for i in range(0,X.shape[0]):
        X.iloc[i,2599] = X.iloc[i,2599].split('\\n')[0]
    
    '''
    X = pd.DataFrame(index=label_sub.iloc[:,0], columns=range(0,2600))
    for f in range(0,label_sub.iloc[:,0].size):
        print(f)
        one = pd.read_table(path + label_name + '/'+ str(label_sub.iloc[f,0]) + '.txt',header=None, sep=',') #无列名
        X.iloc[f,:] = one.iloc[0,:]
    '''
    
    return X,y


[X,y] = X_format(path=path,label_name='first_train_index_20180131',random_state=10)    
X.to_csv(path + 'X_sample_4w_times.csv', index=True)
y.to_csv(path + 'y_sample_4w_times.csv', index=True, index_label='id', header=['type'])



#%% model construct
X = pd.read_csv(path + 'X_sample_4w_times.csv', header=0, index_col=0, sep=',')
y = pd.read_csv(path + 'y_sample_4w_times.csv', header=0, index_col=0, sep=',')

yy = y
yy[y.iloc[:,0].isin(['star'])] = 1
yy[y.iloc[:,0].isin(['galaxy'])] = 2
yy[y.iloc[:,0].isin(['qso'])] = 3
yy[y.iloc[:,0].isin(['unknown'])] = 4

#先使用划分数据验证不同模型的优劣，然后用所有训练集进行建模
X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.4, random_state=0) #训练集:测试集=6:4


#1 随机森林模型
# 划分训练集建模
rf2 = RandomForestClassifier(oob_score=True, random_state=10, criterion='gini', n_estimators=100, 
                             n_jobs=-1, warm_start=True, max_features='auto')
#class_weight=[{3:20}]
#class_weight = [{0:1, 1:20}, {0:1, 1:5}, {0:1, 1:1}, {0:1, 1:10}] instead of [{1:20}, {2:5}, {3:1}, {4:10}]

rf2.fit(X_train,y_train) #sample_weight=None
print(rf2.oob_score_) #oob_score=True才有的参数，类似交叉验证的评估结果，0.8104
print(rf2.feature_importances_) #查看特征重要性
y_pred = rf2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) #0.8030
f1_score_mean(y_test.iloc[:,0],y_pred) #0.7602


# 所有训练集建模，后用测试集预测
rf = RandomForestClassifier(oob_score=True, random_state=10, criterion='gini', n_estimators=100, 
                             n_jobs=-1, warm_start=True, max_features='auto')
rf.fit(X,yy)
print(rf.oob_score_)


#2 GBDT模型
# 划分训练集建模
gbdt = GradientBoostingClassifier(random_state=10)
#gbdt = GradientBoostingClassifier(learning_rate=0.05,n_estimators=100,max_depth=7, 
#                                      min_samples_leaf=60, min_samples_split=300,
#                                      subsample=0.8, random_state=10)
gbdt.fit(X_train,y_train) 
y_pred= gbdt.predict(X_test) 
#y_predprob= gbdt.predict_proba(X_test)[:,1]  
accuracy = accuracy_score(y_test, y_pred) #0.80127
#roc_auc_score(y_test, y_predprob)
f1_score_mean(y_test.iloc[:,0],y_pred) #0.7604


# 所有训练集建模，后用测试集预测
gbdt = GradientBoostingClassifier(random_state=10)
gbdt.fit(X,yy) 


#3 xgboost包
#3.1 use xgb (有报错！查看原因)
dtrain=xgb.DMatrix(X_train,label=y_train.iloc[:,0])
dtest=xgb.DMatrix(X_test)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]
bst = xgb.train(params, dtrain, num_boost_round=10, evals=watchlist) #报错？？
ypred=bst.predict(dtest)


#or
#3.2 use XGBClassifier
# 划分训练集建模
#bst2 = XGBClassifier()
bst2 = XGBClassifier(learning_rate =0.05, n_estimators=100, max_depth=5, 
                     min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,
                     objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
bst2.fit(X_train,y_train)
y_pred = bst2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) #.80439
f1_score_mean(y_test.iloc[:,0],y_pred) #0.7643

# 所有训练集建模，后用测试集预测
bst2 = XGBClassifier()
bst2.fit(X,yy)



#%% model validation
#1 十倍交叉证实得分评估，默认是f1-score进行评定
obj = rf #选择模型 rf/bst/bst2/gbdt
scores = cross_val_score(obj, X, y, cv=10)
print(scores.mean())

#2 精确性评估
obj.score(X_train,y_train) #适用于sklearn包的结果
#or
accuracy = accuracy_score(y_test, y_pred)
#or
correct = [1 if (a == b) else 0 for (a, b) in zip(y_test, y_pred)]   
accuracy = (sum(map(int, correct))/len(correct))

#3 F-score
def precision_score(y_true, y_pred, key):
    return ((y_true==key)*(y_pred==key)).sum()/(y_pred==key).sum()

def recall_score(y_true, y_pred, key):
    return ((y_true==key)*(y_pred==key)).sum()/(y_true==key).sum()

def f1_score(y_true, y_pred, key):
    num = 2*precision_score(y_true, y_pred, key)*recall_score(y_true, y_pred, key)
    deno = (precision_score(y_true, y_pred, key)+recall_score(y_true, y_pred, key))
    return num/deno

def f1_score_mean(y_true,y_pred):
    f_star = f1_score(y_true, y_pred, key=1)
    f_galaxy = f1_score(y_true, y_pred, key=2)
    f_qso = f1_score(y_true, y_pred, key=3)
    f_unknown = f1_score(y_true, y_pred, key=4)
    f_score_m = (f_star + f_galaxy + f_qso + f_unknown)/4
    return f_score_m

f1_score_mean(y_test,y_pred)



#%% test validation

#1 use test data


#2 use rank data
#先使用cat将10w条数据进行合并，每个文件1w数据量，而后导入
obj = rf # rf or gbdt
obj2 = 'rf'
file_path = path + 'rank_' + obj2 + '_times'
os.mkdir(file_path)
label_name='first_rank_index_20180307'
for num in range(1,11):
    print(num)
    label3 = pd.read_csv(path +label_name+'/sub' + str(num) + '_label.txt', header=None, names=['id'], sep=',')
    index = label3.iloc[:,0].str.split('.txt')
    for i in range(0,10000):
        index[i] = index[i][0]
    index = list(index)
    X3 = pd.read_csv(path +label_name +'/sub' + str(num) + '.txt', header=None, sep=',')
    
    for i in range(0,10000):
        X3.iloc[i,2599] = X3.iloc[i,2599].split('\\n')[0]
    
    y3_pred = obj.predict(X3)

    y3_pred = pd.DataFrame(y3_pred,index=index, columns=['type'])
    
    y3_pred2 =y3_pred
    y3_pred2[y3_pred.iloc[:,0].isin([1])] = 'star'
    y3_pred2[y3_pred.iloc[:,0].isin([2])] = 'galaxy'
    y3_pred2[y3_pred.iloc[:,0].isin([3])] = 'qso'
    y3_pred2[y3_pred.iloc[:,0].isin([4])] = 'unknown'
     
    y3_pred2.to_csv(file_path + '/y_rank_sub' + str(num) + '.csv', index=True, header=None)


# cat y_rank_sub*.csv > y_rank.csv


obj = bst2 #选择模型 bst2
obj2 = 'bst2'
os.mkdir(path + 'rank_' + obj2)
label_name='first_rank_index_20180307'
for num in range(1,11):
    print(num)
    label3 = pd.read_csv(path +label_name+'/sub' + str(num) + '_label.txt', header=None, names=['id'], sep=',')
    index = label3.iloc[:,0].str.split('.txt')
    for i in range(0,10000):
        index[i] = index[i][0]
    index = list(index)
    X3 = pd.read_csv(path +label_name +'/sub' + str(num) + '.txt', header=None, names=X.columns, sep=',')
    
    for i in range(0,10000):
        X3.iloc[i,2599] = X3.iloc[i,2599].split('\\n')[0]
    
    X3 = X3.convert_objects(convert_numeric=True) #转换了每列数据的类型，否则2599列为object类型，程序将报错！
    
    y3_pred = obj.predict(X3)
        
    y3_pred = pd.DataFrame(y3_pred,index=index, columns=['type'])
    
    y3_pred2 =y3_pred
    y3_pred2[y3_pred.iloc[:,0].isin([1])] = 'star'
    y3_pred2[y3_pred.iloc[:,0].isin([2])] = 'galaxy'
    y3_pred2[y3_pred.iloc[:,0].isin([3])] = 'qso'
    y3_pred2[y3_pred.iloc[:,0].isin([4])] = 'unknown'
     
    y3_pred2.to_csv(path + 'rank_' + obj2 + '/y_rank_sub' + str(num) + '.csv', index=True, header=None)

# cat y_rank_sub*.csv > y_rank.csv



#整合三种模型的预测结果
y_rank_rf = pd.read_csv(path + 'rank_rf/y_rank.csv', header=None, names=['id','type'], sep=',')
y_rank_gbdt = pd.read_csv(path + 'rank_gbdt/y_rank.csv', header=None, names=['id','type'], sep=',')
y_rank_bst2 = pd.read_csv(path + 'rank_bst2/y_rank.csv', header=None, names=['id','type'], sep=',')

r1 = y_rank_rf.iloc[:,1]
res1=r1
res1[r1.isin(['star'])] = 1
res1[r1.isin(['galaxy'])] = 2
res1[r1.isin(['qso'])] = 3
res1[r1.isin(['unknown'])] = 4

r2 = y_rank_gbdt.iloc[:,1]
res2=r2
res2[r2.isin(['star'])] = 1
res2[r2.isin(['galaxy'])] = 2
res2[r2.isin(['qso'])] = 3
res2[r2.isin(['unknown'])] = 4

r3 = y_rank_bst2.iloc[:,1]
res3=r3
res3[r3.isin(['star'])] = 1
res3[r3.isin(['galaxy'])] = 2
res3[r3.isin(['qso'])] = 3
res3[r3.isin(['unknown'])] = 4

merge = pd.concat([res1,res2,res3], axis=1)
merge.index = y_rank_rf.iloc[:,0]

res = pd.DataFrame(r1,index=y_rank_rf.iloc[:,0], columns=['type'])
for i in range(0,merge.shape[0]):  
    res.iloc[i,0] = np.argmax(np.bincount(list(merge.iloc[i,:])))

res2 =res
res2[res.iloc[:,0].isin([1])] = 'star'
res2[res.iloc[:,0].isin([2])] = 'galaxy'
res2[res.iloc[:,0].isin([3])] = 'qso'
res2[res.iloc[:,0].isin([4])] = 'unknown'

res2.to_csv(path + 'y_rank_integrate.csv', index=True, header=None)





