##常用软件包和环境配置

常用软件包

<http://deeplearning.net/software_links/>

- [theano](http://deeplearning.net/software/theano/)

它是一个强大的库，几乎能在任何情况下使用，从简单的logistic回归到建模并生成音乐和弦序列或是使用长短期记忆人工神经网络对电影收视率进行分类。Theano大部分代码是使用Cython编写，Cython是一个可编译为本地可执行代码的Python方言，与仅仅使用解释性Python语言相比，它能够使运行速度快速提升。最重要的是，很多优化程序已经集成到Theano库中，它能够优化你的计算量并让你的运行时间保持最低。它还内置支持使用CUDA在GPU上执行那些所有耗时的计算。所有的这一切仅仅只需要修改配置文件中的标志位即可。在CPU上运行一个脚本，然后切换到GPU，而对于你的代码，则不需要做任何变化。但是早期开发, 打包成module不强。


- [pylearn2](http://deeplearning.net/software/pylearn2/)

在`theano`基础上开发的软件包(针对深度学习), 依赖于theano。它把`deep learning`模块化成了三个步骤, 所以会比较方便。调用它的模块和接口, 可以将自己的数据转化为它内部的标准类型, 且它也集成了`deep learning`的算法。Pylearn2和Theano由同一个开发团队开发，Pylearn2是一个机器学习库，它把深度学习和人工智能研究许多常用的模型以及训练算法封装成一个单一的实验包，如随机梯度下降。你也可以很轻松的围绕你的类和算法编写一个封装程序，为了能让它在Pylearn2上运行，你需要在一个单独的YAML格式的配置文件中配置你整个神经网络模型的参数。


- scikit-neuralnetwork

它和scikit支持的数据类型一模一样, 所以共享性很好。


- caffe

C++写出来的。


- deeplearning4j

Java写出的。


- Torch

matlab写的。


<div><ol>
<li><a href="http://deeplearning.net/software/theano">Theano</a> – CPU/GPU symbolic expression compiler in python (from MILA lab at University of Montreal)</li>
<li><a href="http://www.torch.ch/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.torch.ch/', 'Torch']);">Torch</a> – provides a Matlab-like environment for state-of-the-art machine learning algorithms in lua (from Ronan Collobert, Clement Farabet and Koray Kavukcuoglu)</li>
<li><a href="https://github.com/lisa-lab/pylearn2" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/lisa-lab/pylearn2', 'Pylearn2']);">Pylearn2</a>&nbsp;- Pylearn2 is a library designed to make machine learning research easy.</li>
<li><a href="https://github.com/mila-udem/blocks" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/mila-udem/blocks', 'Blocks ']);">Blocks </a>- A Theano framework for training neural networks</li>
<li><a href="http://www.tensorflow.org/get_started/index.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.tensorflow.org/get_started/index.html', 'Tensorflow']);">Tensorflow</a>&nbsp;-&nbsp;TensorFlow™ is an open source software library for numerical computation using data flow graphs.</li>
<li><a href="https://github.com/dmlc/mxnet" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/dmlc/mxnet', 'MXNet']);">MXNet</a>&nbsp;- MXNet is a deep learning framework designed for both efficiency and flexibility.</li>
<li><a href="http://caffe.berkeleyvision.org/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://caffe.berkeleyvision.org/', 'Caffe']);">Caffe</a> -Caffe is a deep learning framework made with expression, speed, and modularity in mind.Caffe is a deep learning framework made with expression, speed, and modularity in mind.</li>
<li><a href="https://github.com/Lasagne/Lasagne" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/Lasagne/Lasagne', 'Lasagne ']);">Lasagne </a>- Lasagne is a lightweight library to build and train neural networks in Theano.</li>
<li><a href="http://keras.io/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://keras.io/', 'Keras']);">Keras</a>- A theano based deep learning library.</li>
<li><a href="http://deeplearning.net/tutorial">Deep Learning Tutorials</a> – examples of how to <em>do</em> Deep Learning with Theano (from LISA lab at University of Montreal)</li>
<li><a href="https://github.com/pfnet/chainer" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/pfnet/chainer', 'Chainer']);">Chainer</a>&nbsp;- A GPU based Neural Network Framework</li>
<li><a href="https://github.com/Microsoft/CNTK/wiki" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/Microsoft/CNTK/wiki', 'CNTK – ']);">CNTK – </a>Computational Network Toolkit – is a unified deep-learning toolkit by Microsoft Research.</li>
<li><a href="http://www.vlfeat.org/matconvnet/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.vlfeat.org/matconvnet/', 'MatConvNet']);">MatConvNet</a>&nbsp;- A MATLAB toolbox implementing&nbsp;Convolutional Neural Networks&nbsp;(CNNs) for computer vision applications. It is simple, efficient, and can run and learn state-of-the-art CNNs.</li>
<li><a href="https://github.com/rasmusbergpalm/DeepLearnToolbox" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/rasmusbergpalm/DeepLearnToolbox', 'DeepLearnToolbox']);">DeepLearnToolbox</a> – A Matlab toolbox for Deep Learning (from Rasmus Berg Palm)</li>
<li><a href="http://code.google.com/p/cuda-convnet/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://code.google.com/p/cuda-convnet/', 'Cuda-Convnet']);">Cuda-Convnet</a> – A fast C++/CUDA implementation of convolutional (or more generally, feed-forward) neural networks. It can model arbitrary layer connectivity and network depth. Any directed acyclic graph of layers will do. Training is done using the back-propagation algorithm.</li>
<li><a href="http://www.cs.toronto.edu/%7Ehinton/MatlabForSciencePaper.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/%7Ehinton/MatlabForSciencePaper.html', 'Deep Belief Networks']);">Deep Belief Networks</a>. Matlab code for learning Deep Belief Networks (from Ruslan Salakhutdinov).</li>
<li><a href="http://www.fit.vutbr.cz/~imikolov/rnnlm/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.fit.vutbr.cz/~imikolov/rnnlm/', 'RNNLM']);">RNNLM</a>- Tomas Mikolov’s Recurrent Neural Network based Language models Toolkit.</li>
<li><a href="http://sourceforge.net/projects/rnnl/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://sourceforge.net/projects/rnnl/', 'RNNLIB']);">RNNLIB</a>-RNNLIB is a recurrent neural network library for sequence learning problems. Applicable to most types of spatiotemporal data, it has proven particularly effective for speech and handwriting recognition.</li>
<li><a href="http://code.google.com/p/matrbm/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://code.google.com/p/matrbm/', 'matrbm']);">matrbm</a>. Simplified version of Ruslan Salakhutdinov’s code, by Andrej Karpathy (Matlab).</li>
<li><a href="https://github.com/deeplearning4j/deeplearning4j" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/deeplearning4j/deeplearning4j', 'deeplearning4j']);">deeplearning4j</a>- Deeplearning4J is an Apache 2.0-licensed, open-source, distributed neural net library written in Java and Scala.</li>
<li><a href="http://www.cs.toronto.edu/%7Ersalakhu/rbm_ais.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/%7Ersalakhu/rbm_ais.html', 'Estimating Partition Functions of RBM’s']);">Estimating Partition Functions of RBM’s</a>. Matlab code for estimating partition functions of Restricted Boltzmann Machines using Annealed Importance Sampling (from Ruslan Salakhutdinov).</li>
<li><a href="http://web.mit.edu/%7Ersalakhu/www/DBM.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://web.mit.edu/%7Ersalakhu/www/DBM.html', 'Learning Deep Boltzmann Machines ']);">Learning Deep Boltzmann Machines </a>Matlab code for training and fine-tuning Deep Boltzmann Machines (from Ruslan Salakhutdinov).</li>
<li>The <a href="http://lush.sourceforge.net/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://lush.sourceforge.net/', 'LUSH']);">LUSH</a> programming language and development environment, which is used @ NYU for deep convolutional networks</li>
<li><a href="http://cs.nyu.edu/~koray/wp/?page_id=29" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://cs.nyu.edu/~koray/wp/?page_id=29', 'Eblearn.lsh']);">Eblearn.lsh</a> is a LUSH-based machine learning library for doing Energy-Based Learning. It includes code for “Predictive Sparse Decomposition” and other sparse auto-encoder methods for unsupervised learning. <a href="http://cs.nyu.edu/~koray/wp/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://cs.nyu.edu/~koray/wp/', 'Koray Kavukcuoglu']);">Koray Kavukcuoglu</a> provides Eblearn code for several deep learning papers on this <a href="http://cs.nyu.edu/~koray/wp/?page_id=17" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://cs.nyu.edu/~koray/wp/?page_id=17', 'page']);">page</a>.</li>
<li><a href="https://github.com/kyunghyuncho/deepmat" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/kyunghyuncho/deepmat', 'deepmat']);">deepmat</a>- Deepmat, Matlab based deep learning algorithms.</li>
<li><a href="https://github.com/dmlc/mshadow" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/dmlc/mshadow', 'MShadow']);">MShadow</a>&nbsp;- MShadow is a lightweight CPU/GPU Matrix/Tensor Template Library in C++/CUDA. The goal of mshadow is to support efficient, device invariant and simple tensor library for machine learning project that aims for both simplicity and performance. Supports CPU/GPU/Multi-GPU and distributed system.</li>
<li><a href="https://github.com/dmlc/cxxnet" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/dmlc/cxxnet', 'CXXNET']);">CXXNET</a>&nbsp;- CXXNET is&nbsp;fast, concise, distributed deep learning framework based on MShadow. It is a lightweight and easy extensible C++/CUDA neural network toolkit with friendly Python/Matlab interface for training and prediction.</li>
<li><a href="http://nengo.ca/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://nengo.ca/', 'Nengo']);">Nengo</a>-Nengo is a graphical and scripting based software package for simulating large-scale neural systems.</li>
<li><a href="http://eblearn.sourceforge.net/index.shtml" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://eblearn.sourceforge.net/index.shtml', 'Eblearn']);">Eblearn</a> is a C++ machine learning library with a BSD license for energy-based learning, convolutional networks, vision/recognition applications, etc. EBLearn is primarily maintained by <a href="http://cs.nyu.edu/~sermanet/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://cs.nyu.edu/~sermanet/', 'Pierre Sermanet']);">Pierre Sermanet</a> at NYU.</li>
<li><a href="http://code.google.com/p/cudamat/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://code.google.com/p/cudamat/', 'cudamat']);">cudamat</a> is a GPU-based matrix library for Python. Example code for training Neural Networks and Restricted Boltzmann Machines is included.</li>
<li><a href="http://www.cs.toronto.edu/~tijmen/gnumpy.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/~tijmen/gnumpy.html', 'Gnumpy']);">Gnumpy</a> is a Python module that interfaces in a way almost identical to numpy, but does its computations on your computer’s GPU. It runs on top of cudamat.</li>
<li>The <a href="http://www.ais.uni-bonn.de/deep_learning/downloads.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.ais.uni-bonn.de/deep_learning/downloads.html', 'CUV Library']);">CUV Library</a> (github <a href="https://github.com/deeplearningais/CUV" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/deeplearningais/CUV', 'link']);">link</a>) is a C++ framework with python bindings for easy use of Nvidia CUDA functions on matrices. It contains an RBM implementation, as well as annealed importance sampling code and code to calculate the partition function exactly (from <a href="http://www.ais.uni-bonn.de" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.ais.uni-bonn.de', 'AIS lab']);">AIS lab</a> at University of Bonn).</li>
<li><a href="http://www.cs.toronto.edu/~ranzato/publications/factored3wayRBM/code/factored3wayBM_04May2010.zip" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/~ranzato/publications/factored3wayRBM/code/factored3wayBM_04May2010.zip', '3-way factored RBM']);">3-way factored RBM</a> and <a href="http://www.cs.toronto.edu/~ranzato/publications/mcRBM/code/mcRBM_04May2010.zip" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/~ranzato/publications/mcRBM/code/mcRBM_04May2010.zip', 'mcRBM']);">mcRBM</a> is python code calling CUDAMat to train models of natural images (from <a href="http://www.cs.toronto.edu/~ranzato" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/~ranzato', 'Marc’Aurelio Ranzato']);" title="Marc'Aurelio Ranzato">Marc’Aurelio Ranzato</a>).</li>
<li>Matlab code for training <a href="http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html', 'conditional RBMs/DBNs']);">conditional RBMs/DBNs</a> and <a href="http://www.cs.nyu.edu/~gwtaylor/publications/icml2009/code/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.nyu.edu/~gwtaylor/publications/icml2009/code/', 'factored conditional RBMs']);">factored conditional RBMs</a> (from <a href="http://www.cs.nyu.edu/~gwtaylor/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.nyu.edu/~gwtaylor/', 'Graham Taylor']);">Graham Taylor</a>).</li>
<li><a href="http://www.cs.toronto.edu/~ranzato/publications/mPoT/mPoT.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/~ranzato/publications/mPoT/mPoT.html', 'mPoT']);">mPoT</a> is python code using CUDAMat and gnumpy to train models of natural images (from <a href="http://www.cs.toronto.edu/%7Eranzato" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.cs.toronto.edu/%7Eranzato', 'Marc’Aurelio Ranzato']);" title="Marc'Aurelio Ranzato">Marc’Aurelio Ranzato</a>).</li>
<li><a href="https://github.com/ivan-vasilev/neuralnetworks" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/ivan-vasilev/neuralnetworks', 'neuralnetworks']);">neuralnetworks</a> is a java based gpu library for deep learning algorithms.</li>
<li><a href="https://github.com/sdemyanov/ConvNet" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/sdemyanov/ConvNet', 'ConvNet']);">ConvNet</a> is a matlab based convolutional neural network toolbox.</li>
<li><a href="http://elektronn.org/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://elektronn.org/', 'Elektronn']);">Elektronn</a> is a deep learning toolkit that makes powerful neural networks accessible to scientists outside the machine learning community.</li>
<li><a href="http://www.opennn.net/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://www.opennn.net/', 'OpenNN']);">OpenNN</a> is an open source class library written in C++ programming language which implements neural networks, a main area of deep learning research.</li>
<li><a href="https://neuraldesigner.com/" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://neuraldesigner.com/', 'NeuralDesigner']);">NeuralDesigner</a> &nbsp;is an innovative deep learning tool for predictive analytics.</li>
<li><a href="https://github.com/ironbar/Theano_Generalized_Hebbian_Learning" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/ironbar/Theano_Generalized_Hebbian_Learning', 'Theano Generalized Hebbian Learning.']);">Theano Generalized Hebbian Learning.</a></li>
<li><a href="http://singa.apache.org/en/index.html" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'http://singa.apache.org/en/index.html', 'Apache Singa']);" target="_blank">Apache Singa</a> is an open source deep learning library that provides a flexible architecture for scalable distributed training. It is extensible to run over a wide range of hardware, and has a focus on health-care applications.</li>
<li><a href="https://github.com/yechengxi/LightNet" onclick="_gaq.push(['_trackEvent', 'outbound-article', 'https://github.com/yechengxi/LightNet', 'Lightnet']);">Lightnet</a>&nbsp;&nbsp;is a lightweight, versatile and purely Matlab-based deep learning framework. The aim of the design is to provide an easy-to-understand, easy-to-use and efficient computational platform for deep learning research.</li>
</ol>
</div>


###环境配置

<https://en.wikipedia.org/wiki/CUDA>

<https://developer.nvidia.com/cuda-gpus>

CUDA: **CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia.** It allows software developers and software engineers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing – an approach termed GPGPU (General-Purpose computing on Graphics Processing Units). **The CUDA platform is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of compute kernels.**


scikit-learn: <http://scikit-learn.org/stable/>

scikit-neuralnetwork: <https://github.com/aigamedev/scikit-neuralnetwork>

```
#Install dependencies(I have installed Anaconda3)
> pip3 install Theano                   #Install theano
> pip3 install -e git+https://github.com/lisa-lab/pylearn2.git#egg=Package  #install pylearn2
> 
> #Install sickit-neuralnetwork
> git clone https://github.com/aigamedev/scikit-neuralnetwork.git
> cd scikit-neuralnetwork; python setup.py develop  #Only the dependencies
> pip3 install scikit-neuralnetwork
> 
> #Test
> pip install nose
> nosetests -v sknn.tests
```


Mnist little demo

```
#You had better use `proxychains python3`
>>> from sklearn.datasets import fetch_mldata
>>> mnist = fetch_mldata('MNIST original')
>>> mnist.data.shape
(70000, 784)        #60000 instances, 10000 for test. 28 * 28  for one instance(one pixel is a feature)
>>>
>>> import numpy as np
>>> np.unique(mnist.target)
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
```





















