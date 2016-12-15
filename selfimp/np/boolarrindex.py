#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-14 root <root@VM-17-202-debian>

#Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:

import numpy as np

a = np.array([[1,2],[3,4],[5,6]])

# Find the elements of a that are bigger than 2;
# this returns a numpy array of Booleans of the same
# shape as a, where each slot of bool_idx tells
# whether that element of a is > 2.
bool_idx = (a > 2)

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print bool_idx
print a[bool_idx]


#We can do all of the above in a single concise statement:

print a[a>2]




