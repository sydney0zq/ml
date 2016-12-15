#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])

print a*b

b = 2.0
print a*b
print "*" * 40
x = np.arange(4)
print x.shape
xx = x.reshape(4,1)
print "xx is "
print xx
y = np.ones(5)
z = np.ones((3,4))

print y 
print z
#error shape mismatch: objects cannot be broadcast to a single shape
#print x + y
print "*" * 40
print xx.shape
print xx
print y.shape
print y
print (xx+y).shape
print xx+y

print (x+z).shape
print x+z
print z 
print x

print "*" * 40

a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
print a
print b
print a[:, np.newaxis]
print a[:, np.newaxis] + b

#Here the newaxis index operator inserts a new axis into a, making it a two-dimensional 4x1 array. Combining the 4x1 array with b, which has shape (3,), yields a 4x3 array.





