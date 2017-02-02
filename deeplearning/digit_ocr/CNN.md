##CNN

<http://neuralnetworksanddeeplearning.com/chap6.html>

看 PDF。

关键词：

feature map: 从输入层到第一个隐藏层，用不同的 feature map, 可以得到多个神经网络（通常一些表现好的方法都使用更多的 feature map)

CNN 共享权重和偏向，大大减少了参数的数量。

对于每一个 feature map, 需要 5x5=25 个权重参数，加上一个偏向 b, 26 个。
如果有 20 个 feature maps, 则一共才 520 个参数。

如果之前的 NN, 两两相连，需要 28x28=784 加上隐藏层的 30 个神经元，需要 784x30+30=23550 个参数。比 CNN 多了 40 倍。


pooling layers: 浓缩神经网络的代表性, 减小尺寸。

重要特征点找到之后, 绝对位置并不重要, 相对位置更加重要。其他的pooling如L2 pooling(平方再开方)。


训练还是backprop和SGD的方法解决。















































