#! /usr/bin/env python
# -*- coding: utf-8 -*-
#2016-12-15 root <root@VM-17-202-debian>

import numpy as np


class Perceptron(object):
    """Perceptron classifier.
    
    Parameters
    ========
    eta: float
        Learning rate(between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.

    Attributes
    ========
    w_: id_array
        Weights after fitting
    errors_: list
        Number of misclassification in every epoch
    """

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data.
    
        Parameters
        =========
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and n_features is the 
            number of features.
        y: array-like, shape = [n_samples]
            Target values.

        Returns
        =========
        self: object
        """

        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X, y):
                updates = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calc net input"""
        return np.dot(X, self.w_[1:]+self.w_[0])

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) <= 0.0, 1, -1)






