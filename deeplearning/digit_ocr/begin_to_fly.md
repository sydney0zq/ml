##More hidden layer

<http://neuralnetworksanddeeplearning.com/chap5.html>


当层数更深的时候：可以学习到不同抽象程度的概念。例如图像中：第一层学到边角，第二层学到一些基本形状，第三层学到物体概念。


如何训练深度神经网络？

难点：神经网络的不同层学习的速率显著不同。（靠近输入层的学习比较慢，接近输出层的速度会比较快，差别很大）

Vanishing gradient problem

第一层的学习速率远远低于第二层学习速率。可以从公式中推出来。

PDF 上的例子非常鲜明，差距可以达到 100 倍之多。

还有一种相反的情况，即内层学习速率很快，外层学习很慢。(exploding gradient problem)

所以神经网络算法用 gradient 之类的算法学习存在不确定性。

看 PDF 看 PDF 看 PDF。有证明。

修正以上问题：

1. 初始化比较大的权重 100
2. 初始化 b 使得 sigmoid 的导数不太小，即想方设法使得 z=0

--->但这又可能导致 exploding 的问题。


> 从根本来说，这不是 vanishing 或者 exploding 的问题，而是后面层的梯度是前面层的累积，所以神经网络不稳定。唯一可能的情况是以上的连续乘积刚好平衡大约等于 1, 但是这种几率非常小。

总结来看，只要是 sigmoid 函数的神经网络都会造成 gradient 的时候极其不稳定或者 Vanishing or exploding 问题。


其他难点：

sigmoid 函数造成输出层的 activation 大部分饱和为 0, 建议了其他的 activation 函数。

提出了 momentum-based SGD

PDF 提出了不同的 activation 函数，看一下加深理解。注意 ReL 方程。

Sigmoid函数适合描述概率, ReL适合描述实数。

Sigmoid函数的gradient随着x增大或减小和消失。ReL不会。其优势是不会产生Vanishing gradient问题。








