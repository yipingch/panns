panns -- 最邻近搜索
===================


.. image:: https://pypip.in/d/panns/badge.png

 
.. image:: https://pypip.in/license/gensim/badge.png

panns是Python Approximate Nearest Neighbor Search的缩写。panns是一种优化的python库，用于在高维空间进行 `近似最邻近查找`_ 。比方说，一种典型应用是在语义网页的大量文本资料中搜寻相关的字条。目前，panns支持两种距离度量：欧几里德距离（Eclidean）和角距离（cosine）.对于角的相似性来说，数据集需要标准化。

.. _近似最邻近查找: http://zh.wikipedia.org/wiki/最邻近搜索#近似最邻近查找

::

	from panns import *

	p1 = PannsIndex(metric='angular')    # index using cosine distance metric
	p2 = PannsIndex(metric='euclidean')  # index using Euclidean distance metric
	...

从技术层面来说，panns只是我们开发项目中一个很小的模块。我们把它作为独立包放出是因为我们意识到很难找到一种简单的工具在高维空间中进行高准确率的K-NN搜索。高维在这里指的是数据集具有 **成千上万不同的特性** ，这已经超出 `k-d tree`_ 的处理范围。

.. _k-d tree: http://en.wikipedia.org/wiki/K-d_tree

panns是由 `Liang Wang`_ @ Helsinki University开发。若您有如何问题，请发邮件至 ``liang.wang[at]helsinki.fi`` 或者在 `panns-group`_ 提出意见。

.. _Liang Wang: http://cs.helsinki.fi/liang.wang
.. _panns-group: https://groups.google.com/forum/#!forum/panns

特征
----

- 纯python的实现。
- 对大型高维数据集的优化(e.g. > 500)。
- 生成很小具有很高请求准确率的索引文件。
- 支持欧几里德(Euclidean)和余弦(cosine)距离度量。
- 支持并行索引的生成。
- 极低的内存使用率以及索引可以被多个进程共享。
- 支持raw, csv和 `HDF5`_ 数据集。

.. _HDF5: http://www.hdfgroup.org/HDF5/

安装
----

在panns中代数计算依赖雨 `Numpy`_ 包和 `Scipy`_ 包，HDF5运行依赖 `h5py`_ 包。值得注意的是h5py包是可选的，如果你不需要使用HDF5文件可以不用安装。在使用panns的功能之前，请确保上述包已经安装。用户可以通过下面的shell命令来进行安装上述包。

.. _Numpy: http://www.numpy.org/
.. _Scipy: http://www.scipy.org/
.. _h5py: http://www.h5py.org/

::

	sudo pip install numpy --upgrade
	sudo pip install scipy --upgrade
	sudo pip install h5py --upgrade

panns的安装过程很直接。你可以选择通过PyPI来直接安装或者下载源代码进行手动安装。

::

	sudo pip install panns --upgrade

如果您对源代码有兴趣或者愿意帮助提高panns的性能，您可以从Github下载源代码。

::

	git clone git@github.com:ryanrhymes/panns.git

快速开始
--------

panns假设资料集是一个基于排的矩阵(e.g. m x n)，每一排代表多维空间的数据点。下面第二行开始的代码创建一个1000乘以100的数据矩阵，然后创建一个由50个二叉树组成的索引，最后把这个索引储存在文件里。

::

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

除了使用``add_vector(v)``函数，panns还支持其他多种方式加载数据集。对于那些极大的数据集，`HDF5`_ 的使用会导致创建性能极大的降低，但可以通过并行创建来提高其性能。

.. _HDF5: http://www.hdfgroup.org/HDF5/


::

	# datasets can be loaded in the following ways
	p.load_matrix(A)                     # load a list of row vectors or a numpy matrix
	p.load_csv(fname, sep=',')           # load a csv file with specified separator
	p.load_hdf5(fname, dataset='panns')  # load a HDF5 file with specified dataset

存储的索引可以被未来的多个进程加载和共享。因此，并行性可以提高请求性能。下述代码加载先前生成的索引文件，然后进行一个简单的请求。请求会返回大约10个邻近的节点。

::

	from panns import *

	p = PannsIndex(metric='euclidean')
	p.load('test.idx')

	v = gaussian_vector(100)
	n = p.query(v, 10)

通常，在高维资料集中创建索引是一个很耗时的过程，panns通过两个方面加速这个过程：优化代码以及充分利用物理资源。在多核的环境中，并行创建更容易达到。代码如下：

::

	from panns import *

	p = PannsIndex(metric='angular')

	....

	p.parallelize(True)
	p.build()

原理简述
-------

简单来说，我们通过 `random projection`_ 来获取k-NN的近似值。索引的创建可以通过生成一个人二叉树来实现。树中的每个节点代表一定数值点，进而通过比较平均值被分成两组(左子树和右子树)。准确率可以通过下面的方法来提高：

.. _random projection: http://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection

Simply put, approximate k-NN in panns is achieved by `random projection`_. The index is built by constructing a binary tree. Each node of the tree represents a scalar-projection of certain data points, which are further divided into two groups (left- and right-child) by comparing to their average. The accuracy can be improved from the following perspective:

.. _random projection: http://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection

- 合理的放置偏离值(e.g. 在示例的平均值)
- 选择合理的投影向量(随机值或者生成分)
- 使用更多的投影(但是需要更多的生成时间和更大的索引)
- 使用更多的二叉树(也要更多的生成时间和更大的索引)

- Place the offset wisely (e.g. at the sample average).
- Choose the projection vector wisely (e.g. random or principle components).
- Use more projections (but longer building time and larger index).
- Use more binary trees (also longer building time and larger index).

实现近似k-NN值高准确率是以大索引为代价的。panns希望在这两个冲突的值中寻求一个平衡点。与其他的库为每个节点生成一个全新的随机向量不同，panns重复使用不同树中的投影向量。这种办法极大降低索引的大小当维数很高或者数很多的时候。与此同时，重复使用投影向量不会降低准确性(请看评估部分)

The accuracy of approximate k-NN is usually achieved at the price of large index. panns aims to find the good trade-off of these two conflicting factors. Different from other libraries, panns reuses the projection vectors among different trees instead of generating a new random vector for each node. This can significantly reduces the index size when the dimension is high and trees are many. At the same time, reusing the projection vectors will not degrade the accuracy (see Evaluation section below).

评估
----

评估部分主要通过比较panns和Annoy. Annoy是用C++开发的，具有和panns一样的功能。它被用于Spotify 推荐系统中。在评估中，我们使用5000 x 200的数据集，命名为5000个 200维数的向量。为了公平比较，Annoy和panns各生成128个二叉树。评估通过两种距离度量(Euclidean和cosine).下述列表总结了实验结果。(data type?)

+------------+-------------------+-------------------+----------------+----------------+
|	     | panns (Euclidean) | Annoy (Euclidean) | panns (cosine) | Annoy (cosine) |
+------------+-------------------+-------------------+----------------+----------------+
|  Accuracy  | 	   69.2%         |     48.8%         |    70.1%       |     50.4%      |
+------------+-------------------+-------------------+----------------+----------------+
| Index Size |     5.4 MB        |     20 MB         |    5.4 MB      |     11 MB      |
+------------+-------------------+-------------------+----------------+----------------+

比较Annoy, panns可以达到更高的准确率采用更小的索引文件。原因已经在原理部分简单描述。一般来说，高准确率是通过放置偏离值在示例的平均值，与此同时，实现更小的索引是通过重复使用投影向量。

Compared with Annoy, panns can achieve higher accuracy with much smaller index file. The reason was actually already briefly discussed in "Theory" section. Generally speaking, the higher accuracy is achieved by placing the offset at sample average; while the smaller index is achieved by reusing the projection vectors.

值得注意的是这里的评估远远不够，我们还需要其他方面的评估。

讨论
----

任何建议，问题和相关的讨论非常欢迎，您可以在 `panns-group`_ 提出意见以及找到相关的信息。

.. _panns-group: https://groups.google.com/forum/#!forum/panns

未来工作
--------

- 在索引文件上实现mmap去提高索引加载速度
- 用并行性去提高请求性能
- 从更广泛的角度去评估
