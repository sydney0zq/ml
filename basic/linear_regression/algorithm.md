##Simple Linear Regression

为什么需要统计量? ---> 描述数据特征。


####集中趋势衡量

1. 均值
2. 中位数(median): 将数据中的各个数值按照大小循序排列, 居于中间位置的变量。当 n 为奇数的时候, 直接取位置处于中间的数; 当 n 为偶数的时候, 取中间两个量的平均值。
3. 众数(mode): 数据中出现次数最多的数


####离散程度的衡量

1. 方差(variance)

![](../pic/lr-00.png)

2. 标准差(standard deviation)

![](../pic/lr-01.png)


###Introduction

介绍: 
- 回归(regression) Y 变量为连续数值型(continous numberical variable)。如: 房价, 人数, 降雨量
- 分类(classificaiton): Y 变量为类别型(categorical variable)。如: 颜色类别, 电脑品牌, 有无信誉


简单线性回归(Simple Linear Regression)
- 很多做决定过程通常是根据两个或者多个变量之间的关系
- 回归分析(regression analysis)用来建立方程模拟两个或者多个变量之间如何关联
- 被预测的变量叫做: 因变量(dependent variable), 即为输出 y (output)
- 被用来进行预测的变量叫做: 自变量(indenpent variable), 即为输入 x (input)


简单线性回归介绍
- 简单线性回归包含一个自变量(x)和一个因变量(y)
- 以上两个变量的关系用一个直线来模拟
- 如果包含两个以上的自变量, 则称为多元回归分析(multiple regression)


简单线性回归模型
- 被用来描述因变量(y)和自变量(X)以及偏差(error)之间关系的方程叫做归回模型
- 简单线性回归的模型是:

![](../pic/lr-02.png)


简单线性回归方程:  $E(y) = β_0+β_1x$
**这个方程对应的图像是一条直线, 称作回归线。**
其中, β0是回归线的截距, β1是回归线的斜率, E(y)是在一个给定 x 值下 y 的期望值(均值)。

> 对模型的期望值求期望和对回归方程求期望, 对偏差的期望为0。(偏差的分布为正态分布)


正向线性关系:

![](../pic/lr-03.png)


负向线性关系:

![](../pic/lr-04.png)


无关系:

![](../pic/lr-05.png)



估计的简单线性回归方程: $ŷ=b_0+b_1x$

这个方程叫做估计线性方程(estimated regression line), 其中, b0是估计线性方程的纵截距;  b1是估计线性方程的斜率;  ŷ是在自变量x等于一个给定值的时候，y的估计值


###线性回归分析流程

![](../pic/lr-06.gif)

$\beta 0$和$\beta 1$是真实的值, b0 和 b1 是对这两个值的估计。

关于偏差ε的假定
- 是一个随机的变量, 均值为0
- ε 的方差(variance)对于所有的自变量 x 是一样的
- ε 的值是独立的
- ε 满足正态分布


####简单线性回归实例

```
Number of TV ads(x)     Number of Cars sold(y)
1                       14
3                       24
2                       18
1                       17
3                       27
sum x = 10              sum y = 100
ave x = 2               ave y = 20
```

![](../pic/lr-07.png)

目标就是使得`sum of squares`最小。


![](../pic/lr-08.png)
![](../pic/lr-09.png)






















