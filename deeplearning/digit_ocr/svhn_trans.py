#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
SVHN loader for assignment.
"""

import scipy.io as sio
import cPickle as cPkl
import pickle as pkl


train_data = sio.loadmat("../data/test_32x32.mat")

train_x = train_data['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
train_y = train_data['y'].reshape((-1)) - 1

print "Saving training data"
with open("../data/test_svhn.pkl", "w") as f:
    pkl.dump([train_x, train_y], f, protocol = cPkl.HIGHEST_PROTOCOL)
    f.close()





