# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:01:17 2018

@author: User
"""

#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
 
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
 
#Equation coefficient and Intercept
print('Coefficient: n', model.coef_)
print('Intercept: n', model.intercept_)
 
#Predict Output
predicted= model.predict(x_test)

