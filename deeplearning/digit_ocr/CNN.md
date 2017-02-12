##CNN

<http://neuralnetworksanddeeplearning.com/chap6.html>

看 PDF。

关键词：

feature map: 从输入层到第一个隐藏层，用不同的 feature map, 可以得到多个神经网络（通常一些表现好的方法都使用更多的 feature map)

CNN 共享权重和偏向，大大减少了参数的数量。

对于每一个 feature map, 需要 5x5=25 个权重参数，加上一个偏向 b, 26 个。
如果有 20 个 feature maps, 则一共才 520 个参数。

如果之前的 NN, 两两相连，需要 28x28=784 加上隐藏层的 30 个神经元，需要 784x30+30=23550 个参数。比 CNN 多了 40 倍。


pooling layers: 浓缩神经网络的代表性，减小尺寸。

重要特征点找到之后，绝对位置并不重要，相对位置更加重要。其他的 pooling 如 L2 pooling（平方再开方）。


训练还是 backprop 和 SGD 的方法解决。

在`demo_cnn.py`中，为什么只对最后一层用 dropout?

CNN 本身的 convolution 层对于 overfitting 有防止作用：共享的权重造成 convolution filter 强迫对于图像进行学习。


为什么可以克服深度学习里面的一些困难？

用 CNN 大大减少了参数的数量。
用 dropout 减少了 overfitting。
用 Rectified linear units 代替 sigmoid, 避免了 overfitting和不同层学习率差距很大的问题。
用 GPU 计算，每次更新较少，但是可以训练很多次。



目前的深度神经网络有多深？

最多有 20 层。
















































