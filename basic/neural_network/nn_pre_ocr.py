#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Just see the datasets of handwriting digits.
"""

from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)
#output:
# (1797, 64)
#1797 instances, every instance 64 pixels

import pylab as pl

pl.gray()
pl.matshow(digits.images[0])
pl.savefig("./demo.png")



