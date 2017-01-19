#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
A classic example by neural network using python.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from nn_application import NeuralNetwork
from sklearn.cross_validation import train_test_split


digits = load_digits()
X = digits.data
y = digits.target
#For normalize to 0~1
X -= X.min()        #normalize the values to bring them into the range 0-1
X /= X.max()

#Every instance is 64 pixels(dimensions), the outputs are 0~9
#一般的话隐藏层要比输入层多一些, 这样模型会更好
nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
#把 label 所表达的数字进行编码(必须这么做, 不能直接是 0~9)
#0  1   2   3   4   5   6   7   8   9
#假如类别是 0 的话
#1  0   0   0   0   0   0   0   0   0
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print("Start Fitting...")
#print(len(X_train))
#print(len(X_test))
#print(len(labels_train))
#print(len(labels_test))
nn.fit(X_train, labels_train, epochs = 30000)

predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    #0~9每一个都有一个概率值, 我们需要选一个最大的对应的整数
    predictions.append(np.argmax(o))

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))



