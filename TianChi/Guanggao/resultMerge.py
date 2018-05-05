#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:37:08 2018

@author: tinghai
"""

#结果线性加权计算
#0.3 * xgboost + 0.3 * lightGBM + 0.2 lightGBM_xgboost + 0.2 * xgboost_lightGBM

import pandas as pd
import os
import datetime


def main():
    
    path = '/Users/tinghai/Learning/GuanggaoData'
    os.chdir(path + '/source')
    
    import analysis as an
    an.main()
    
    os.chdir(path)
    
    lightGBM = pd.read_csv('./result/submit_lightGBM.csv', header=0, sep=' ')
    xgboost = pd.read_csv('./result/submit_XGBoost.csv', header=0, sep=' ')
    lightGBM_xgboost = pd.read_csv('./result_fusion/submit_construct_lightGBM_predict_XGBoost.csv', header=0, sep=' ')
    xgboost_lightGBM = pd.read_csv('./result_fusion/submit_construct_XGBoost_predict_lightGBM.csv', header=0, sep=' ')
    
    result = 0.25 * xgboost.iloc[:,1] + 0.25 * lightGBM.iloc[:,1] + 0.35 * lightGBM_xgboost.iloc[:,1] + 0.15 * xgboost_lightGBM.iloc[:,1]
    result2 = pd.concat([lightGBM.iloc[:,0],pd.DataFrame(result)],axis=1)
    result2.to_csv(('./result/submit_integrate_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"), 
                index=False, index_label=None, header=['instance_id','predicted_score'], sep=' ')


if __name__=='__main__':
    
    print('Guang Gao analysis is starting!\n')
    
    main()
    
    print('Result integrate is end.')
    
   