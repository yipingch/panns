panns -- 最邻近搜索
==================

![Downloads](https://pypip.in/d/panns/badge.png "Downloads") . ![License](https://pypip.in/license/gensim/badge.png "License")

panns是Python Approximate Nearest Neighbor Search的缩写。panns是一种优化的python库。这个库用于在高维空间中进行[近似最邻近查找](http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor)。一种典型应用是在语义网络中对大量文本资料对相关字条进行搜寻。目前，panns支持两种距离度量：欧几里德距离(Eclidean)和余弦距离(cosine)。－对于角的相似性来说，数据集需要标准化。－

panns stands for "Python Approximate Nearest Neighbor Search", which is an optimized python library for searching [approximate k-nearest neighbors](http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor) in very high dimensional spaces. E.g. one typical use in semantic web is finding the most relevant documents in a big text corpus. Currently, panns supports two distance metrics: Euclidean and Angular (consine). For angular similarity, the dataset need to be normalized.

```python

from panns import *

p1 = PannsIndex(metric='angular')    # index using cosine distance metric
p2 = PannsIndex(metric='euclidean')  # index using Euclidean distance metric
...
```

从技术层面来说，panns只是我们开发项目中一个很小的模块。我们把它作为一个单独的包放出是因为我们意识到很难找到一种简单的工具能在高维空间中进行高效的K-NN搜索。－高维在这里指的是数据集具有成千上万不同的特性。－这已经超出[k-d树](http://en.wikipedia.org/wiki/K-d_tree)的处理能力。

Technically, panns is only a small function module in one of our ongoing projects. The reason we release it as a separate package is we realized it is actually very difficult to find an easy-to-use tool which can perform efficient k-NN search with satisfying accuracy in high dimensional space. High dimensionality in this context refers to those datasets having **hundreds of features**, which is already far beyond the capability of standard [k-d tree](http://en.wikipedia.org/wiki/K-d_tree).

panns是由[Liang Wang](http://cs.helsinki.fi/liang.wang) @ Helsinki University开发。若您有任何疑问，请发邮件至`liang.wang[at]helsinki.fi`或者在[panns-group](https://groups.google.com/forum/#!forum/panns)提出您的宝贵意见。

panns is developed by [Liang Wang](http://cs.helsinki.fi/liang.wang) @ Helsinki University. If you have any questions, you can either contact me via email `liang.wang[at]helsinki.fi` or post in [panns-group](https://groups.google.com/forum/#!forum/panns)。－我的联系方式－－


## 特征

* 纯python的实现。
* 对大型高维数据集的优化。(比方说，量大于500的) - 什么是高维数据集 -
* 生成很小但有很高搜寻准确率的索引文件。
* 支持欧几里德(Euclidean)和余弦(cosine)距离度量。
* 支持并行索引的生成。
* 极低的内存使用率以及索引可以被多个进程共享。
* 支持raw，csv和[HDF5](http://www.hdfgroup.org/HDF5/)数据集。

* Pure python implementation.
* Optimized for large and high-dimension dataset (e.g. > 500).
* Generate small index file with high query accuracy.
* Support Euclidean and cosine distance metrics.
* Support parallel building of indices.
* Small memory usage and index can be shared among processes.
* Support raw, csv and [HDF5](http://www.hdfgroup.org/HDF5/) datasets.



## 快速安装

在panns中大部分代数运算依赖于[Numpy](http://www.numpy.org/)包和[Scipy](http://www.scipy.org/)包。至于一些涉及到HDF5的运算，依赖的包是[h5py](http://www.h5py.org/)。值得注意的是，在这里HDF5的包是可选的。如果不需要相关的运算，您可以考虑不使用HDF5文件。在使用panns的功能之前，请确保上述包已经成功安装。您可以通过下面的shell命令来安装上述包。

Algebra operations in panns rely on both [Numpy](http://www.numpy.org/) and [Scipy](http://www.scipy.org/), and HDF5 operations rely on [h5py](http://www.h5py.org/). Note h5py is optional if you do not need operate on HDF5 files. Please make sure you have these packages properly installed before using the full features of panns. The installation can be done by the following shell commands.

```bash
sudo pip install numpy --upgrade
sudo pip install scipy --upgrade
sudo pip install h5py --upgrade
```
在安装完上述包后，您可以开始安装panns。安装panns的过程相当简单，您有两种安装方式可以选择：直接通过PyPI安装，或者下载下载panns的源代码进行手动安装。

The installation of panns is very straightforward. You can either install it directly from PyPI (probably the easiest way), or download the source code then install manually.

```bash
sudo pip install panns --upgrade
```

如果您对panns的源代码有兴趣，请加入我们。您可以从Github下载源代码。

If you are interested in the source code or even want to contribute to make it faster and better, you can clone the code from Github.

```bash
git clone git@github.com:ryanrhymes/panns.git
```



## 快速开始

panns假设数据集是由一个－基于排的矩阵（比方说， m x n）－，其中每一排代表n维的数据点。下面代码是个例子：第一行代表创建一个由1000乘以100的矩阵，然后创建50个二叉树的索引，最后将这个索引保存在一个文件中。

panns assumes that the dataset is a row-based the matrix (e.g. m x n), where each row represents a data point from an n-dimension feature space. The code snippet below first constructs a 1000 by 100 data matrix, then builds an index of 50 binary trees and saves it to a file.

```python

from panns import *

# create an index of Euclidean distance
p = PannsIndex(dimension=100, metric='euclidean')

# generate a 1000 x 100 dataset
for i in xrange(1000):
    v = gaussian_vector(100)
    p.add_vector(v)

# build an index of 50 trees and save to a file
p.build(50)
p.save('test.idx')
```

除了使用`add_vector(v)`函数去加载一个数据集，panns还提供其他的方式。对于那些相当大的数据集，[HDF5](http://www.hdfgroup.org/HDF5/)是值得推荐，但是创建的性能会极大的被降低。可是，我们可以通过并行的生成来提高起性能。具体方式列举如下：

Besides using `add_vector(v)` function, panns supports multiple ways of loading a dataset. For those extremely large datasets, [HDF5](http://www.hdfgroup.org/HDF5/) is recommended though the building performance will be significantly degraded. However, the performance can be improved by enabling parallel building as shown later.

```python
# datasets can be loaded in the following ways
p.load_matrix(A)                     # load a list of row vectors or a numpy matrix
p.load_csv(fname, sep=',')           # load a csv file with specified separator
p.load_hdf5(fname, dataset='panns')  # load a HDF5 file with specified dataset
```

存储的索引可以被未来的多个进程加载和共享。因此，并行性可以提高请求性能。下述代码加载先前生成的索引文件，然后进行一个简单的请求。请求会返回大约10个邻近的节点。

The saved index can be loaded and shared among different processes for future use. Therefore, the query performance can be further improved by parallelism. The following code loads the previously generated index file, then performs a simple query. The query returns 10 approximate nearest neighbors.

```python

from panns import *

p = PannsIndex(metric='euclidean')
p.load('test.idx')

v = gaussian_vector(100)
n = p.query(v, 10)
```

通常，在高维资料集中创建索引是一个很耗时的过程，panns通过两个方面加速这个过程：优化代码以及充分利用物理资源。在多核的环境中，并行创建更容易达到。代码如下：

Usually, building index for a high dimensional dataset can be very time-consuming. panns tries to speed up this process from two perspectives: optimizing the code and taking advantage of the physical resources. If multiple cores are available, parallel building can be easily enabled as follows:

```python

from panns import *

p = PannsIndex(metric='angular')

....

p.parallelize(True)
p.build()

```



## 原理简述

简单来说，我们通过[random projection](http://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection)来获取k-NN的近似值。索引的创建可以通过生成一个人二叉树来实现。树中的每个节点代表一定数值点，进而通过比较平均值被分成两组(左子树和右子树)。准确率可以通过下面的方法来提高：

Simply put, approximate k-NN in panns is achieved by [random projection](http://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection). The index is built by constructing a binary tree. Each node of the tree represents a scalar-projection of certain data points, which are further divided into two groups (left- and right-child) by comparing to their average. The accuracy can be improved from the following perspective:


* 合理的放置偏离值(e.g. 在示例的平均值)
* 选择合理的投影向量(随机值或者生成分)
* 使用更多的投影(但是需要更多的生成时间和更大的索引)
* 使用更多的二叉树(也要更多的生成时间和更大的索引)
* Place the offset wisely (e.g. at the sample average).
* Choose the projection vector wisely (e.g. random or principle components).
* Use more projections (but longer building time and larger index).
* Use more binary trees (also longer building time and larger index).

实现近似k-NN值高准确率是以大索引为代价的。panns希望在这两个冲突的值中寻求一个平衡点。与其他的库为每个节点生成一个全新的随机向量不同，panns重复使用不同树中的投影向量。这种办法极大降低索引的大小当维数很高或者数很多的时候。与此同时，重复使用投影向量不会降低准确性(请看评估部分)

The accuracy of approximate k-NN is usually achieved at the price of large index. panns aims to find the good trade-off of these two conflicting factors. Different from other libraries, panns reuses the projection vectors among different trees instead of generating a new random vector for each node. This can significantly reduces the index size when the dimension is high and trees are many. At the same time, reusing the projection vectors will not degrade the accuracy (see Evaluation section below).



## 评估

评估部分主要通过比较panns和Annoy. Annoy是用C++开发的，具有和panns一样的功能。它被用于Spotify 推荐系统中。在评估中，我们使用5000 x 200的数据集，命名为5000个 200维数的向量。为了公平比较，Annoy和panns各生成128个二叉树。评估通过两种距离度量(Euclidean和cosine).下述列表总结了实验结果

Evaluation in this section is simply done by comparing against Annoy. Annoy is a C++ implementation of similar functionality as panns, it is used in Spotify recommender system. In the evaluation, we used a 5000 x 200 dataset, namely 5000 200-dimension feature vectors. For fair comparison, both Annoy and panns use 128 binary trees, and evaluation was done with two distance metrics (Euclidean and cosine). The following table summarizes the results. (data type?)

|            | panns (Euclidean) | Annoy (Euclidean) | panns (cosine) | Annoy (cosine) |
|:----------:|:-----------------:|:-----------------:|:--------------:|:--------------:|
|  Accuracy  |       69.2%       |       48.8%       |      70.1%     |      50.4%     |
| Index Size |       5.4 MB      |       20 MB       |     5.4 MB     |      11 MB     |


比较Annoy, panns可以达到更高的准确率采用更小的索引文件。原因已经在原理部分简单描述。一般来说，高准确率是通过放置偏离值在示例的平均值，与此同时，实现更小的索引是通过重复使用投影向量。

Compared with Annoy, panns can achieve higher accuracy with much smaller index file. The reason was actually already briefly discussed in "Theory" section. Generally speaking, the higher accuracy is achieved by placing the offset at sample average; while the smaller index is achieved by reusing the projection vectors.

值得注意的是这里的评估远远不够，我们还需要其他方面的评估。

One thing worth pointing out is the evaluation here is far from thorough and comprehensive, other evaluations are highly welcome and we are always ready to link.



## 讨论

任何建议，问题和相关的讨论非常欢迎，您可以在[panns-group](https://groups.google.com/forum/#!forum/panns)提出意见以及找到相关的信息。

Any suggestions, questions and related discussions are warmly welcome. You can post and find relevant information in [panns-group](https://groups.google.com/forum/#!forum/panns) .



## 未来工作

* 在索引文件上实现mmap去提高索引加载速度
* 用并行性去提高请求性能
* 从更广泛的角度去评估

* Implement mmap on index file to speed up index loading.
* Improve query performance by parallelism.
* Perform more thorough evaluations.
