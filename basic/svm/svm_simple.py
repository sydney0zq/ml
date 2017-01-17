#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 root <root@localdebian>
#
# Distributed under terms of the MIT license.

"""
Calculate three points' hyperplane by SVM. 
A simple case.
"""

from sklearn import svm
import numpy as np

X = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
#kernel function: linear --> can be divided by line
clf = svm.SVC(kernel = 'linear')
clf.fit(X, y)

print(clf)

#get Support Vectors
#it will print the points on the side gutter
print("* support vectors: ", clf.support_vectors_)
#* support vectors:  [[ 1.  1.]
# [ 2.  3.]]

#get indices of support vectors
#take out the indexes of points on the side gutter
print("* indices of support vectors: ", clf.support_)
#* indices of support vectors:  [1 2]

#get number of support vectors for each class
#how many number belong to support vectors
print("* number of support vectors for each class: ", clf.n_support_)
#* number of support vectors for each class:  [1 1]

#It seems that it must be np.array type that solves the warning
print(clf.predict(np.array([[2, 0]])))





