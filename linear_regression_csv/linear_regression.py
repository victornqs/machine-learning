# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 22:35:40 2016

@author: Victor Barreto
"""
import pandas as pd
import matplotlib.pyplot as plot
df = pd.read_csv('linear_regression.csv')
to_forecast = df.Rendimento.values
dates = df.Frequencia.values, df.PressaoRetaguarda.values

import numpy as np

def organize_data(to_forecast, window, horizon):
    
    shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast, 
                                        shape=shape, 
                                        strides=strides)
    y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
    return X[:-horizon], y

k = 4   # number of previous observations to use
h = 1   # forecast horizon
X,y = organize_data(to_forecast, k, h)
from sklearn.linear_model import LinearRegression
 
m = 10 # number of samples to take in account
regressor = LinearRegression(normalize=True)
regressor.fit(X[:m], y[:m])
def mape(ypred, ytrue):    
    idx = ytrue != 0.0
    return 100*np.mean(np.abs(ypred[idx]-ytrue[idx])/ytrue[idx])

print 'Error %0.2f%%' % mape(regressor.predict(X[m:]),y[m:])

plot.plot(regressor.predict(X))
plot.plot(y)
plot.title('Real x Previsto')
plot.xlabel('Previsto (Azul), Real (Verde)')
plot.show()

print(regressor.predict(X))
