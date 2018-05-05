#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:04:41 2018

@author: tinghai
"""

import numpy as np
import lda
import lda.datesets
titles = lda.datasets.load_reuters_titles()
for i in range(0,380):
    print(titles[i])
    


