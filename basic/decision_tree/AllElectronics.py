#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decision Tree
直接使用sklearn实现好的DT, 但是sklearn对数据类型有要求
- 对属性必须是数值型的。比如对age有三种类型, 那么就映射到数字上去
比如对于第一行的数据, 就

youth middle_age senior
1     0          0
high  medium     low
1     0          0
...

用这种方法来指定一个instance的各种属性和类别才能被sklearn处理
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

#Read in the csv file and put features in a list of an class label
allElectronicsData = open(r"./AllElectronics.csv", newline='')
#Return a reader object which will iterate over lines in the given csvfile
reader = csv.reader(allElectronicsData)
#Attention reader.next() has been renamed to reader.__next__()
headers = reader.__next__()

#print(headers)

featureList = []        #For feature, NO RID
labelList = []          #For class

for row in reader:
    #Appent the last value(class)
    labelList.append(row[len(row) - 1])
    rowDict = {}
    #--------->
    #--------->
    #Interate
    for i in range(1, len(row) - 1):
        #print (row[i])
        rowDict[headers[i]] = row[i]
        #print("rowDict: " + str(rowDict))
    featureList.append(rowDict)
print("featureList is: \n")
print(featureList)
print("\n")


#Vectorize features, because python supports make a list(content is dict)
# vectorized
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())
print("LabelList: " + str(labelList))

#Vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " +str(dummyY))


#Using decision tree for classification
#http://scikit-learn.org/stable/modules/tree.html
#Default isnot entropy
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

#Visualize model
with open("allElectronicInformationGainOri.dot", "w") as f:
    #feature_name is for drawing the origin feature name
    #dot -T pdf DOTFILENAME.dot -o output.pdf
    f = tree.export_graphviz(clf, 
            feature_names = vec.get_feature_names(), out_file = f)

#Now predict a new data from the decision tree
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX)
#output: predictedY[1], it has changed to 1, means buy
print("predictedY" + str(predictedY))





