#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-18 root <root@VM-17-202-debian>

import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 3*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2,1,1)
plt.plot(x, y_sin)
plt.title('sine')

plt.subplot(2,1,2)
plt.plot(x, y_cos)
plt.title('cosine')

plt.savefig('./pic/subplot.png')


