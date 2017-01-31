##Notes


1. 数层数通常不计输出层。

![](../pic/do-00.png)

梯度下降核心就是减去导数值。

对于每个训练实例x, 都要计算梯度向量gradient vector: $\Delta C$

如果训练数据集过大, 会花费很长的时间, 学习过程太慢, 所以一个变种称为

**随机梯度下降算法(stochastic gradient descent)**

基本思想: 从所有训练实例中取出一个小的采样(sample): X1, X2, ..., Xm (mini-batch) 来估计$\partial C$, 大大提高学习速度。







