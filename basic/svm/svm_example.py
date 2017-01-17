#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 root <root@localdebian>
#
# Distributed under terms of the MIT license.

"""
A bigger example of SVM.
Some doc are writing here:
1. axis question
axis=0, stands for COLUMN!
axis=1, stands for ROW!

All the codes are in <http://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html#sphx-glr-auto-examples-svm-plot-svm-margin-py>
"""

print(__doc__)

import numpy as np
import pylab as pl
from sklearn import svm

#create 40 separable points
np.random.seed(2)
"""np.r_: <https://docs.scipy.org/doc/numpy/reference/generated/numpy.r_.html>"""
#Translates slice objects to concatenation along the first axis.
#>>> np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
#array([1, 2, 3, 0, 0, 4, 5, 6])
#>>> np.r_[-1:1:6j, [0]*3, 5, 6]
#array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])
#randn generates Gauss Distribute 20 points, and 2 columns
"""randn: <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html>"""
X = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
Y = [0] * 50 + [1] * 50

#fit the model
clf = svm.SVC(kernel = "linear")
clf.fit(X, Y)

#get the separating hyperplane
"""
这里因为要画的直线和w是垂直的, 所以斜率积为-1, 那么用-w[0]/w[1]可以得到要画直线的斜率
intercept_: Constants in decision function.
"""
w = clf.coef_[0]
a = - w[0]/w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0] / w[1])

#plot the parallels to the separating hyperplane that pass through the 
#support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print("w: ", w)
print("a: ", a)
print("support_vectors_: ", clf.support_vectors_)
print("number of support vectors for each class: ", clf.n_support_)
print("clf.coef_: ", clf.coef_)
print("clf.intercept_: ", clf.intercept_)

#plot the line, the points and the nearest vectors to the plane
pl.plot(xx, yy, "k-")
pl.plot(xx, yy_down, "k--")
pl.plot(xx, yy_up, "k--")

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s = 200, facecolors = "none")
pl.scatter(X[:, 0], X[:, 1], c = Y, cmap = pl.cm.Paired)

pl.axis('tight')
pl.savefig("./SVM-gauss.png")























