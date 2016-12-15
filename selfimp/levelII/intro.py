#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-18 root <root@VM-17-202-debian>

import numpy as np
import cPickle as pickle
import os

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'r') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        """X is N*D where each row is a example. Y is 1D of size N"""
        #the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """X is N*D where each row is an example we wish to predict for """
        num_test = X.shape[0]
        #lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        #loop over all test rows
        for i in xrange(num_test):
            #find the nearest training image to the i'th test image
            #Using L1 distance(sum of absolute value  differences
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distance)
            Ypred[i] = self.ytr[min_index]
        return Ypred








#Xtr holds all the images in the training set, and a corrsponding 1-dimensional array ytr holds the training labels(from 0~9)[length: 50000]

# a magic function we provide
Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar/')
# flatten out all images to be one-dimensional

# Xtr_rows becomes 50000 x 3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
# Xte_rows becomes 10000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

# create a nearest neighbor classifier class
nn = NearestNeighbor()
# train the classifier on the training images and labels
nn.train(Xtr_rows, Ytr)
#predict labels on the test images
Yte_predict = nn.predict(Xte_rows)

# and now print the classification accuracy, which is the 
#average number of examples that are correctly predicted
#(i.e. label matches)
print 'accuracy: %f' % (np.mean(Yte_predict == Yte))










