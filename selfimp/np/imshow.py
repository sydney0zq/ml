#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-18 root <root@VM-17-202-debian>

import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('./pic/bdlogo.png')
img_tinted = img * [0.5, 0.2, 0.7]

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)

plt.imshow(np.uint8(img_tinted))
plt.savefig('./pic/testimage.png')
