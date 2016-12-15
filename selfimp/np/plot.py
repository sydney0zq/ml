#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-17 root <root@VM-17-202-debian>

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)

plt.xlabel('x is sb')
plt.ylabel('y is sb')
plt.title('all is sb')
plt.legend(['SINE','COSINE'])

plt.savefig('./pic/to.png')  
#plt.plot(x,y)

#savefig('./pic/plot.png', dpi=100)

