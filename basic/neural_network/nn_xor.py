#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 root <root@localdebian>
#
# Distributed under terms of the MIT license.

"""
Depend on nn_application.py

XOR
00          0
10          1
01          1
11          0
"""

from nn_application import NeuralNetwork
import numpy as np

#传入layer: 几层网络, 每层几个神经单元; 
#[2, 2, 1]表示的是 **输入层有两个神经单元, 隐藏层有两个神经单元, 输出层有一个神经单元**
nn = NeuralNetwork([2, 2, 1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
print("The weights are:\n", nn.weights, "\n")

print("The outputs are:")
for i in [[0, 0], [0, 1] , [1, 0], [1, 1]]:
    print(i, nn.predict(i))


