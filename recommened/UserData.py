# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:24:19 2018

@author: User

Ref: https://github.com/Lockvictor/MovieLens-RecSys
"""

#data description
#users.dat：UserID::Gender::Age::Occupation::Zip-code 
#movies.dat：MovieID::Title::Genres 
#ratings.dat：UserID::MovieID::Rating::Timestamp

import pandas as pd
import os
import random

#数据准备
path = '/Users/tinghai/Learning/MachineLearning/recommend/'
os.chdir(path)

unames=['user_id','gender','age','occupation','zip']
users=pd.read_table('ml-1m/users.dat',sep='::',header=None,names=unames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=mnames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames)

all_data = pd.merge(pd.merge(ratings, users), movies)
data = pd.DataFrame(data=all_data, columns=['user_id','movie_id'])
data.to_csv('ml-1m/data/data.csv')


#数据分析
data = pd.read_csv('data/data.csv')
X = data['user_id']
Y = data['movie_id']


#划分训练集和测试集
def SplitData(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0,M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train, test

[train,test] = SplitData(data, M=8, k=1, seed=10)
#or
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0) #训练集:测试集=6:4



#测评指标：召回率和精度
def Recall(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def Precision(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)

def Coverage(train, test, N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)
 
def Popularity(train, test, N):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
                item_popularity[item] += 1
    ret = 0
    n=0
    for user in train.keys():
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
            ret /= n * 1.0
    return ret


