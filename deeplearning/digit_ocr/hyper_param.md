##Hyper-parameter 的选择

<http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters>

到 CH3 为止，学到的超参数有：

- 学习率 (learning rate)
- Regularization parameter: $\lambda$


如果超参数选择不当，则神经网络的表现结果非常糟糕，几乎和随机猜测差不多。但是神经网路中可变化调整的因素很多。

总体策略：

从简单的出发：开始实验。

如 mnist 数据集，开始不知道如何设置，可以先简化使用 0, 1 两类图。减少 80% 的数据量，用两层神经网络 [784, 10]。更快的获取反馈，之前每个 epoch 来检测准确率，可以替换每 1000 张图之后，或者减少 validation_data 的量。

mini-batch 是一个比较独立的参数，不用重新尝试，选定即可。

Grid Search 网络状自动搜索各种参数组合。

SGD 其他变种：Hessian 优化，momentum-based gradient descent.

除了sigmoid还有tanh, rectified linear神经元。


目前神经网络还有很多方面理论基础需要研究, 为什么学习能力强, 现在的一些实验结果表明结果比较好, 但是发展底层理论基础还有很长的路要走。




