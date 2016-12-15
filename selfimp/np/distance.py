#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-17 root <root@VM-17-202-debian>

import numpy as np
from scipy.spatial.distance import pdist, squareform

x = np.array([[0,1],[1,0],[2,0]])
print x

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],

d =squareform(pdist(x, 'euclidean'))
print d




