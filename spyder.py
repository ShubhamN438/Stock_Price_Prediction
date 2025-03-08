# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:29:10 2024

@author: asus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)