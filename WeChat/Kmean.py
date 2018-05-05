# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:04:52 2018

@author: User
"""

#Import Library
from sklearn.cluster import KMeans
 
#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model
k_means = KMeans(n_clusters=3, random_state=0)
 
# Train the model using the training sets and check score
model.fit(X)
 
#Predict Output
predicted= model.predict(x_test)