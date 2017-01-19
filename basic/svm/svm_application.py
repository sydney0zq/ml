#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Face recognition example using SVM.
"""

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

print (__doc__)

#Display progress logs on tty
logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(message)s")

######################################################################
#Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)

#introspect the images arrays to find the shapes(for plotting)
n_samples, h, w = lfw_people.images.shape

#for maching learning we use the 2 data directly(as relative pixel)
#position info is ignored in this model
X = lfw_people.data
n_features = X.shape[1]      # return the column number

#the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print ("Total dataset size: ")
#这个数据集一共有n_samples个实例
print ("n_samples: %d" % n_samples)
#特征值有1850, 非常高, 肯定需要降维
print ("n_features: %d" % n_features)
#有多少个人
print ("n_classes: %d" % n_classes)


########################################################
#split into a training set and a test set using a stratified k flod

X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size = 0.25)

########################################################
#compute a PCA(eigenfaces) on the face dataset(treated as unlabeled)
#dataset: unsuperival feature extraction / dimensionality reduction

n_components = 150
print ("Extracting the top %d eigenfaces from %d faces" %
        (n_components, X_train.shape[0]))
t0 = time()
#reduce the dimension
pca = RandomizedPCA(n_components = n_components, whiten = True).fit(X_train)
print ("done in %0.3fs" % (time() - t0))

#pick up some eigen features in pictures
eigenfaces = pca.components_.reshape((n_components, h, w))

print ("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)
print ("done in %0.3fs" % (time() - t0))

########################################################
#Train a SVM classification model, the most important

print ("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.001, 0.005, 0.001, 0.005, 0.01, 0.1], }
#It will combine all my params, such <1e3 and 0.001> and <1e3 and 0.005> etc(notice grid)
clf = GridSearchCV(SVC(kernel = 'rbf', class_weight = 'balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print ("done in %0.3fs" % (time() - t0))
print ("Best estimator found by grid search:")
print (clf.best_estimator_)

########################################################
#Quantitative evaluation of the model quality on the test set
print ("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print ("done in %0.3fs" % (time() - t0))

print (classification_report(y_test, y_pred, target_names = target_names))
print (confusion_matrix(y_test, y_pred, labels = range(n_classes)))


########################################################
#Qualitative evalution of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row = 3, n_col = 4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize = (1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom = 0, left = 0.01, right = 0.99, top = 0.99, hspace = 0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray)
        plt.title(titles[i], size = 12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig("./SVM-face.png")

#plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return "predicted: %s\n true: %s" % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) 
                        for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

#plot the gallery of the most significatve eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)













































