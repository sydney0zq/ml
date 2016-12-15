#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

import numpy as np

x = np.array([[1,2], [3,4]])

#compute sum of all elements
print np.sum(x)

#compute sum of each column
print np.sum(x, axis = 0)

#print sum of each row
print np.sum(x, axis = 1)



