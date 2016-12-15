#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

#Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. 

import numpy as np

#let numpy choose the datatype
x = np.array([1,2])
print x.dtype

#let numpy choose the datatypes
x = np.array([1.0, 2.0])
print x.dtype

#force a particular datatype
x = np.array([1,2], dtype = np.int64)
print x.dtype
