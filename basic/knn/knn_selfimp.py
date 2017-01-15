#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 root <root@localdebian>
#
# Distributed under terms of the MIT license.

"""
Self-implement of KNN algorithm
"""

import csv
import random
import math
import operator

#split the dataset to training data and test data
#split is the boundary to split these two sets
def loadDataSet(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        #read all rows
        lines = csv.reader(csvfile)
        #translate to list type structure
        #[[...],[...],...]
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

#note that this is testInstance
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        #get the correct label
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            #This is for building the dict, cannot be `+= 0`
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.67
    #r is short for `raw`
    loadDataSet(r"./iris.data", split, trainingSet, testSet)
    print("Train set: " + repr(len(trainingSet)))
    print("Test set: " + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print("> predicted = " + repr(result) + ", actual = " + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print("Accuary: " + repr(accuracy) + "%")


if __name__ == "__main__":
    main()

















