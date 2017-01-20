#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiple regression using python.
"""

from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r'./multi_lr.csv'
deliveryData = genfromtxt(dataPath, delimiter = ',')

print (">>> csv data\n", deliveryData)

#[1:, :-1]中的 1 表示是忽略第 0 行, 从 1 到 last;
#后面的 :-1 是表示从 0 到 last-1 列
X = deliveryData[1:, :-1]
Y = deliveryData[1:, -1]

print (">>> X\n", X)
print (">>> Y\n", Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

#coef is the cofficients not including the intercept
print (">>> cofficients: \n", regr.coef_)
print (">>> intercept: \n", regr.intercept_)


xPred = np.array([[102, 6]])
yPred = regr.predict(xPred)
print (">>> predicted y: \n", yPred)

