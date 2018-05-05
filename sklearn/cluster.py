# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:46:40 2017

@author: BRYAN
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler=StandardScaler()
path='d:/trainCG.csv'
data=pd.read_csv(path).fillna(0)
label=data.label
feature=data.drop('label',axis=1)
feature=scaler.fit_transform(data)

cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean')#affinity 度量方式
cluster=KMeans(n_clusters=2,init='k-means++',n_init=200,precompute_distances=True,n_jobs=-1)
cluster=DBSCAN(algorithm='kd_tree',n_jobs=-1)
cluster.fit(feature)
pred=cluster.fit_predict(feature)

def f(x):
    return 1 if x==0 else 0
pred=[i for i in map(f,pred)]
print(accuracy_score(label,pred))

