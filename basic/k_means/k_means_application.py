#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
K means algorithm using python.
"""

import numpy as np

#Function: K means
#-------------------
#K-means is an algorithm that takes in a dataset and a constant
#K and return k centroids(which defines clusters of data in dataset 
#which are similar to one another)

def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape

    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X

    #Initialize centroids randomly
    #>>> np.random.randint(10, size=6)
    #array([3, 0, 6, 1, 5, 4])  #Rand from [0, 10) and gen 6 number as an array

    #You should notice that this centroids select k whole row(k instances)
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
#   centroids = dataSet[0:2, :]
    #Randomly assign labels to initial centroids
    #[1, k)
    centroids[:, -1] = range(1, k+1)

    #Initialize book keeping vars
    iterations = 0
    oldCentroids = None

    #Run the main k-means algoithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("Iteration: \n", iterations)
        print("DataSet: \n", dataSet)
        print("Centroids: \n", centroids)
        #Save old centroids for covergence test.
        #这里不使用直接赋值等号的原因是, python是面向对象的
        #如果修改其中一个值, 那么另一个也会变
        oldCentroids = np.copy(centroids)
        iterations += 1

        #Assign labels to each datapoint based on centroids
        #这里面的centroids是dataset的一些随机行
        updateLabels(dataSet, centroids)

        #Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, k)

    #we can get the labels too by calling getLabels(dataSet, centroids)
    return dataSet


#Function: Should stop
#----------------------
#Returns True or False if k-means is done. K-means terminates either
#because it has run a maximum number of iterations OR the centroids stop
#changing.
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    #注意是比较对象还是比较值
    return np.array_equal(oldCentroids, centroids)


#Function: Get labels
#----------------------
#Update a label for each piece of data in the dataset.
def updateLabels(dataSet, centroids):
    #For each elemetn in the dataset, chose the closet centroid.
    #make that centroid the element's label
    numPoints, numDim = dataSet.shape
    #每一行进行迭代, 得到一个label
    for i in range(0, numPoints):
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)

#将每一个实例与中心点进行计算距离并比较大小
def getLabelFromClosestCentroid(dataSetRow, centroids):
    #先让label等于第一个中心点的label, 再让最小距离等于与第一个中心
    #点的距离, 反复迭代替换
    label = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        #np.linalg.norm是计算Euclidean distance的内建算法
        #最后一列是label, 不计算
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    print("minDist: ", minDist)
    return label


#Function: Get centroids
#-----------------------
#Return k random centroids, each of dimension n.
def getCentroids(dataSet, k):
    #Each centroid is the geometric mean of the points that
    #have that cetroids label.
    #Attention: If a centroid is empty(no points have that centroid's label)
    #you should randomly re-initlize it
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k+1):
        #这句代码很NB
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        #>>> a = np.array([[1,2,3], [4,5,6]])   
        #>>> np.mean(a, axis = 0)               
        #array([ 2.5,  3.5,  4.5])
        result[i-1, :-1] = np.mean(oneCluster, axis = 0)
        result[i-1,  -1] = i
    return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))

result = kmeans(testX, 2, 10)
print("Final result: ", result)











