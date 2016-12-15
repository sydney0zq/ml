#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-11-15 root <root@VM-17-202-debian>

from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('./pic/bdlogo.png')
print img.dtype, img.shape

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.

img_tinted = img * [0.0, 1.0, 1.0]

img_tinted = imresize(img_tinted, (300,300))


imsave('./pic/bdlogo_tinted.png', img_tinted)

