#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Gradient Descent using python.
"""

import numpy as np
import random

#m denotes the number of examples here, not the number of features
#gradientDescent is for updating the theta
#numIterations means how many steps this function goes
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        #avg cost per example(the 2 in 2*m doesnot really matter here.
        #but to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost %f" % (i, cost))
        #avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

def genData(numPoints, bias, variance):
    x = np.zeros(shape = (numPoints, 2))
    #for `shape=numPoints`, this means only one column
    y = np.zeros(shape = numPoints)
    #basically a straight line
    for i in range(0, numPoints):
        #bias feature
        x[i][0] = 1
        x[i][1] = i
        #our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

#gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData (100, 25, 10)
print (">>> x: \n", x)
print (">>> y: \n", y)

m, n = np.shape(x)
numIterations = 100000
alpha = 0.0005
theta = np.ones(n)

theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print (theta)








































