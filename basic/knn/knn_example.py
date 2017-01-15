#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 root <root@localdebian>
#
# Distributed under terms of the MIT license.

"""
A simple example for knn
datasets: iris
"""

from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
for key in iris:
    print('"', key, '"', "corrspond to ", iris[key], "\n")

#iris.data is 150*4 matrix, iris.target is the class label which is 150*1
knn.fit(iris.data, iris.target)

#create a instance
predictedLabel = knn.predict([[4.5, 7.7, 6.6, 7.4]])
print(predictedLabel)







