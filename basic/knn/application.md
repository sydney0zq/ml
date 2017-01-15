###Application for KNN

####数据集介绍

**虹膜(iris)**

150个实例(行): 
萼片长度,      萼片宽度,    花瓣长度,        花瓣宽度
(sepal length, sepal width, petal length and petal width)

学习目标实现识别这些话的类别。
类别: iris setosa, iris vericolor, iris virginica


####流程

利用python的机器学习库sklearn: SkLearnExample.py

```python
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
print iris

knn.fit(iris.data, iris.target)
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print predictedLabel

```


在这里面并没有设置K的值, 这在文档中有说明。
<http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>











