#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-06 root <root@VM-17-202-debian>

import numpy as np


f = np.array([123, 456, 789])
#p = np.exp(f) / np.sum(np.exp(f))

#print f
f -= np.max(f)
p = np.exp(f) / np.sum(np.exp(f))
print p



