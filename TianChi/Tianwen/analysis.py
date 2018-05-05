# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:41:09 2018
@author: ltt
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
import datetime


path = sys.argv[1] #为Project的路径: /project
#path='/Users/tinghai/Learning/TianWenData/project'

os.chdir(path)


#%% 1 training set extract

def X_format(path,label_name,folder_name,random_state):
    label = pd.read_csv(path + '/data/' + label_name + '.csv', header=0, names=['id','type'], sep=',')
    
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
    files.to_csv(path + '/data/subtrain_label_4w_times.csv', index=False, index_label=None, header=None)
    
    os.system('bash '+ path + '/code/trainMerge.sh ' + path + '/data/' + folder_name + '/') #run trainMerge.sh
    
    #提取subtrain_label.csv中对应的文件信息，shell下形成subtrain.txt文件
    X = pd.read_csv(path + '/data/' + folder_name + '/subtrain_4w_times.txt', header=None, sep=',')
    
    for i in range(0,X.shape[0]):
        X.iloc[i,2599] = X.iloc[i,2599].split('\\n')[0]
    
    return X,y


[X,y] = X_format(path=path, label_name='first_train_index_20180131',folder_name='first_train_data_20180131',random_state=10)    
#X.to_csv(path + 'X_sample_4w_times.csv', index=True)
#y.to_csv(path + 'y_sample_4w_times.csv', index=True, index_label='id', header=['type'])


#%% 2 训练集划分（训练集:测试集=6:4）
#X = pd.read_csv(path + 'X_sample_4w_times.csv', header=0, index_col=0, sep=',')
#y = pd.read_csv(path + 'y_sample_4w_times.csv', header=0, index_col=0, sep=',')

yy = y
yy[y.iloc[:,0].isin(['star'])] = 1
yy[y.iloc[:,0].isin(['galaxy'])] = 2
yy[y.iloc[:,0].isin(['qso'])] = 3
yy[y.iloc[:,0].isin(['unknown'])] = 4


#先使用划分数据验证不同模型的优劣，然后用所有训练集进行建模
X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.4, random_state=0)


#%% 3 定义F-score
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

#f1_score_mean(y_test,y_pred)



#%% 4 构建随机森林模型
rf2 = RandomForestClassifier(oob_score=True, random_state=10, criterion='gini', n_estimators=100, 
                             n_jobs=-1, warm_start=True, max_features='auto')
rf2.fit(X_train,y_train) #sample_weight=None
#print(rf2.oob_score_) #oob_score=True才有的参数，类似交叉验证的评估结果，0.8104
#print(rf2.feature_importances_) #查看特征重要性

#对划分数据集进行预测
y_pred = rf2.predict(X_test)

#评估预测结果
accuracy = accuracy_score(y_test, y_pred) #0.8030
f_score = f1_score_mean(y_test.iloc[:,0],y_pred) #0.7602
print(accuracy)
print(f_score)


# 使用所有训练集建模，用于测试预测
rf = RandomForestClassifier(oob_score=True, random_state=10, criterion='gini', n_estimators=100, 
                             n_jobs=-1, warm_start=True, max_features='auto')
rf.fit(X,yy)
print(rf.oob_score_)



#%% 5 test set validation

# using rank data
# 先使用cat将10w条数据进行合并，每个文件1w数据量，而后导入
#label_name='first_rank_index_20180307'
folder_name='first_rank_data_20180307'
os.system('bash '+ path + '/code/testMerge.sh ' + path + '/data/' + folder_name + '/') #run testMerge.sh

obj = rf
obj2 = 'rf'

for num in range(1,11):
    print(num)
    label3 = pd.read_csv(path + '/data/' + folder_name+'/sub' + str(num) + '_label.txt', header=None, names=['id'], sep=',')
    index = label3.iloc[:,0].str.split('.txt')
    for i in range(0,10000):
        index[i] = index[i][0]
    index = list(index)
    X3 = pd.read_csv(path + '/data/' + folder_name +'/sub' + str(num) + '.txt', header=None, sep=',')
    for i in range(0,10000):
        X3.iloc[i,2599] = X3.iloc[i,2599].split('\\n')[0]
    
    y3_pred = obj.predict(X3)
    y3_pred = pd.DataFrame(y3_pred,index=index, columns=['type'])
    y3_pred2 =y3_pred
    y3_pred2[y3_pred.iloc[:,0].isin([1])] = 'star'
    y3_pred2[y3_pred.iloc[:,0].isin([2])] = 'galaxy'
    y3_pred2[y3_pred.iloc[:,0].isin([3])] = 'qso'
    y3_pred2[y3_pred.iloc[:,0].isin([4])] = 'unknown'
    y3_pred2.to_csv(path + '/data/y_rank_sub' + str(num) + '.csv', index=True, header=None)
    

os.system('bash '+ path + '/code/resultMerge.sh ' + path + '/data/') #run resultMerge.sh

result = pd.read_csv(path + '/data/y_rank.csv', header=None, sep=',')
result.to_csv((path + '/submit/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

#%% main function

def main(argv):
   print('Succeed!')


if __name__ == "__main__":
   main(sys.argv[1:])
   
