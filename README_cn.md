panns -- 最邻近搜索
==================

![Downloads](https://pypip.in/d/panns/badge.png "Downloads") . ![License](https://pypip.in/license/gensim/badge.png "License")

panns是Python Approximate Nearest Neighbor Search的简称。panns是一种用于在高维空间中寻求最邻近节点([approximate k-nearest neighbors](http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor))的python库。一种比较典型的panns应用是在[语义网络](http://baike.baidu.com/view/157370.htm?fr=aladdin)中对大量文本资料对有关联的字符串进行搜寻。相对其他具有相同功能的库，panns有自己性能上的优势。目前，panns支持两种距离度量：[欧式距离](http://baike.baidu.com/view/1615257.htm?fr=aladdin)和余弦距离。[余弦相似度](http://zh.wikipedia.org/wiki/余弦相似性)通常用于两个向量的夹角小于90度，因此，数据集需要标准化(值控制在0到1之间)。


```python

from panns import *

p1 = PannsIndex(metric='angular')    # index using cosine distance metric
p2 = PannsIndex(metric='euclidean')  # index using Euclidean distance metric
...
```

panns本来只是我们正在开发项目中一个很小的模块。最开始我们是想能够在高维空间的环境下找到一种简单的工具进行高效的K-NN搜索，比方说，[k-d树](http://en.wikipedia.org/wiki/K-d_tree)。但在这里，高维指的是每个数据集具有成千上万不同的属性，但这已经超过k-d树的处理能力。

panns是由[Liang Wang](http://cs.helsinki.fi/liang.wang) @ Helsinki University开发，Yiping Chen维护。若您有任何疑问，请发邮件至`liang.wang[at]helsinki.fi`或者`yiping.chen[at]helsinki.fi`。您还可以在[panns-group](https://groups.google.com/forum/#!forum/panns)提出您的宝贵意见。


## 特征

* 纯python的实现。
* 对处理大型高维数据集进行优化，比方说，大于500维。
* 生成很小但有很高搜寻准确率的索引文件。
* 支持欧几里德(Euclidean)和余弦(cosine)距离度量。
* 支持并行索引的生成。
* 极低的内存使用率以及索引可以被多个进程共享。
* 支持raw，csv和[HDF5](http://www.hdfgroup.org/HDF5/)数据集。


## 快速安装

在panns中大部分科学计算依赖于[Numpy](http://www.numpy.org/)和[Scipy](http://www.scipy.org/)。至于一些涉及到HDF5的运算，依赖的包是[h5py](http://www.h5py.org/)。值得注意的是，在这里HDF5是可选的。如果不需要相关的运算，您可以考虑不使用HDF5文件。在使用panns的功能之前，请确保上述包已经成功安装。您可以通过下面的shell命令来安装上述包。


```bash
sudo pip install numpy --upgrade
sudo pip install scipy --upgrade
sudo pip install h5py --upgrade
```
在安装完上述包后，您可以开始安装panns。安装panns的过程相当简单，您有两种安装方式可以选择：直接通过PyPI安装，或者下载下载panns的源代码进行手动安装。


```bash
sudo pip install panns --upgrade
```

如果您对panns的源代码有兴趣，请加入我们。您可以从Github下载源代码。


```bash
git clone git@github.com:ryanrhymes/panns.git
```



## 快速开始
panns假定数据集是一个基于排的矩阵，在这个矩阵中每一排代表一个n维的数据点。下面的代码就是一个例子：第一部分定义一个100维度的基于欧式距离的索引，第二部分创建一个1000x100的数据集(数据矩阵)，第三部分根据先前创建的数据集生成一个50个二叉树的索引然后将这个索引储存在idx文件中。


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

除了使用`add_vector(v)`函数，panns还提供其他多种方式去加载一个数据集。使用[HDF5](http://www.hdfgroup.org/HDF5/)去创建相当大的数据集虽然会极大的降低性能，但是这种方式是指地推荐的，因为我们可以通过并行创建去弥补其损失。具体的逻辑我们后面会解释。

```python
# datasets can be loaded in the following ways
p.load_matrix(A)                     # load a list of row vectors or a numpy matrix
p.load_csv(fname, sep=',')           # load a csv file with specified separator
p.load_hdf5(fname, dataset='panns')  # load a HDF5 file with specified dataset
```

被存储在文件中的二叉树索引未来可以被多个进程加载或者共享。因为这个办法，对索引的请求可以通过并行性进一步来提高。下面的代码就是个例子，首先加载之前创建的idx文件，然后请求其返回一个大约10最邻近节点。


```python

from panns import *

p = PannsIndex(metric='euclidean')
p.load('test.idx')

v = gaussian_vector(100)
n = p.query(v, 10)
```

通常，在高维资料集中创建索引是会很耗时，panns通过两个方面加速这个过程：优化代码以及充分利用物理资源。在多核的环境中，并行创建更容易达到。代码如下：


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

我们非常欢迎关于其的任何讨论和建议。在[panns-group](https://groups.google.com/forum/#!forum/panns)，您可以得到更多相关信息。


## 未来工作

* 在索引文件上实现mmap去提高索引加载速度
* 用并行性去提高请求性能
* 从更广泛的角度去评估
