#!/usr/bin/env python
#coding:utf-8

"""
1、SVD初探
- https://yanyiwu.com/work/2012/09/10/SVD-application-in-recsys.html

2、SVD属于基于协同过滤进行推荐的派系，即“希望预测目标用户对其他未评分物品的评分，进而将评分高的物品推荐给目标用户。”
- 概览总结的还可以：https://www.cnblogs.com/pinard/p/6351319.html
- 需要弄懂以下SVD算法：
-- 传统SVD 
	1. 借助其数学理论，给ML指出了一条不错的路子！~
	2. **PS：在ML中，SVD被称为是Latent Factor Model。
	3. **ML中玩SVD进行推荐的本质：通过矩阵分解把评分矩阵通过分解，用一个低秩的矩阵来逼近原来的评分矩阵，逼近的loss function让预测的矩阵和原来的矩阵之间的误差平方最小。
	4. （矩阵分解一般不用数学上直接分解的办法，尽管分解出来的精度很高，但是效率实在太低！矩阵分解往往会转化为一个优化问题，通过迭代求局部最优解。）
	5. SVD的作用在于可以通过先对缺失评分进行随机填充，然后在迭代后完成准确填充。

-- FunkSVD【就是将原来矩阵降维，分解后矩阵与原来矩阵是近似而不是等价】
	1. FunkSVD是在传统SVD面临计算效率问题时提出来的，既然将一个矩阵做SVD分解成3个矩阵很耗时，同时还面临稀疏的问题，那么我们能不能避开稀疏问题，同时只分解成两个矩阵呢。
	1.5 ****FunkSVD 的最大优势在于：****
		- 不用考虑缺失值, 即计算时只需要考虑有评分的地方；理论来源：http://nicolas-hug.com/blog/matrix_facto_3
		- Loss Function使用非常常见的MSE，这样在寻找最优解的时候有很多好处；
		- user 和 item各自有一个K维度的特征，这个特征可以理解为是 latent factor，即user和item各有K个latent factor
	1.8 
	2. 思路很简单：用MSE作为Loss Function，来寻找最终的分解矩阵P和Q。求解方法选用梯度下降，
	3. 为了达到更好的效果，一般会加入L2正则项
	4. ** FunkSVD是一般在ML中使用的SVD。因为传统数学意义的SVD的分解形式意义不大，ML中的SVD只不过借鉴了SVD分解形式：R=U*S*R，通过最优化方法进行模型拟合，求得R=U*V
	5. FunkSVD属于Latent Factor Model，本质上是加入特征这一概念，然后通过某种方式计算出用户u和每个特征的关系以及物品i和每个特征的关系【在这里，我们并不能知道特征是什么】。
	6. 这样，就可以通过“用户u和每个特征的关系以及物品i和每个特征的关系”来预测用户u对物品i的得分了。LFM的预测用户u对物品i的评分公式如下：r_ui=..... 【http://jyd.me/resys/rating-predict-svd/】

-- BiasSVD
	- 就是在FunkSVD的基础上加入了Baseline Predictors.[为什么要加入：user_i对item_j的评分是3分，如果mean_rate是1.2，那么3分就很高。如果mean_rate是4.7，那么3分就很低。]
	- http://www.cnblogs.com/Xnice/p/4522671.html

-- SVD++
	1. https://medium.com/@danjtchen/svd-%E6%8E%A8%E8%96%A6%E7%B3%BB%E7%B5%B1-%E5%8E%9F%E7%90%86-c72c2e35af9c
	2. 考虑隐式反馈。隐式反馈的形式很多。一般而言，会选择数据比较稠密且容易收集的特征作为隐式反馈的特征，比如浏览行为、时间咨询，或者其它一些看似无关的特征。
	3. 使用方式：
		- 用户i对物品j的兴趣程度 = 用户i对物品j的显式反馈(大概率缺失) + 用户i的一系列隐式反馈在物品j上表现；
		- 定义一组关于item的隐式反馈特征：比如用户u对每一个item的浏览时间；因为是关于item的特征，所以保证了长度和item一样。
		- |N(u)|表示用户u在隐式反馈特征中有行为的个数。
		- |y_i|表示用户u在隐式反馈上的表对将要评分item的隐性影响的权重数。可以理解为一种行为的数值化，如果没有就用0填充，保证长度一致；
	4. 举例说明：
		- 隐式反馈特征定义为：用户浏览过电影的电影介绍页的停留时间。
		- 用户u在这个隐式反馈特征上的表现必定不一样。故|N(u)|长度不一样。使用1/(|N(u)|)^-(1/2) 来平衡个数不同带来的影响。这种标准化的方法纯属是经验，无理论根据。
		- 举个例子：如果用户u对电影i有评分，且已知其在电影介绍页的停留时间，那么这种“修正方式”更能体现用户u对电影i的兴趣程度。
		- **可以认为，svd++是一种SVD 基于特征组合的扩展，也是对SVD 原始的latent factor的一种修正。

-- timeSVD++ ：http://10tiao.com/html/284/201509/207943847/1.html
	1. 玩法很简单，使用户过去的行为成为一个随时间衰减的变量，比如可以用Adam中用过的指数衰减法“EWMA(Exponentially Weighted Moving Average，指数加权移动平均) ”
	2. **timeSVD++其实是衰减过往用户行为的权重，提高就近用户隐式反馈行为的权重。

-- TSVD（Truncated SVD，截断SVD）
	1. 资料：https://blog.csdn.net/zhangweiguo_717/article/details/71778470
	2. 如果奇异值矩阵的奇异值发生严重倾斜：TOP N之后的奇异值全部趋于0。这种情况称之为病态矩阵。
	3. 病态程度越高，无用的特征越多，通常会截取前p个最大的奇异值；
	4. 截断参数p的选取是TSVD方法的一个难点。
	5. 在此处列出TSVD，是因为TSVD的sk-learn中实现PCA的一种svd_solver；
	6. 关于TSVD的算法逻辑参见：https://zhuanlan.zhihu.com/p/32903540  【算法还是有点意思的】
	7. TSVD的目的是为了得到稀疏解。

3、SVD中奇异值的物理意义：奇异值往往对应着矩阵中隐含的重要信息，且重要性和奇异值大小正相关。回应了“机器学习的一个最根本也是最有趣的特性是数据压缩概念的相关性。”
- https://www.zhihu.com/question/22237507
- https://yanyiwu.com/work/2012/09/10/SVD-application-in-recsys.html


4、SVD主要是用于数据压缩和降噪，但也可以用于推荐系统，即将用户和喜好对应的矩阵做特征分解，进而得到隐含的用户需求来做推荐。
- http://www.cnblogs.com/pinard/p/6251584.html
- 数据压缩便于存储；
- 通过SVD的矩阵压缩，便于存储；
- 通过SVD的矩阵分解，可以填充数据缺失的地方。
- 本质上，在用SVD之前，原始矩阵不允许出现缺失，如果有必须先用一些缺失值填充方法（均值填充或者随机数填充）进行填充后才能使用。这也就决定了传统的SVD的局限性。
- missing value被随机填充后的值，会在SVD完成后变成SVD对它的预测值。


5、FunkSVD的参数依赖的问题：
- 如果是用传统的SVD来进行dataSet的压缩，那么根据Stanford course中的建议是：TOP K 个的奇异值的和为c倍的剩下奇异值的和，c一般取10。
- 如果是用funk SVD或其他变种SVD进行协同过滤，那么没有一个合理的办法来确定K值。毕竟没有计算奇异值矩阵。

6、NMF，全称为non-negative matrix factorization，中文呢为“非负矩阵分解”的玩法？
- SVD是否产生负值，和初始化的矩阵P和矩阵Q非常有关系；
- funk SVD 和 SVD++，都是服从高斯分布的随机数初始化P和Q，默认mean=0, sigma=0.1。故大概率会在最后的P和Q中存在负数；
- NMF的loss function和funk SVD很像，只是加了限制条件：P和Q中任意元素必须非负。
- 为了实现上述限制条件，初始化P和Q时，使用服从均匀分布的随机数初始化P和Q，且强制均匀分布的最小值非负。
- NMF基于上述方法初始化P和Q，则能保证最后的结果均非负。
- NMF一般容易overfitting，所以K值一般选的较小。
- NMF的loss function可以是MSE【MSE的几何意义就是欧几里德距离】或者是KL散度

7、一个很重要的point：NMF和K-means的区别
- 资料：http://maider.blog.sohu.com/303848412.html
- Kmeans中的W同样是存放r个类的基向量的矩阵，只不过每个基向量必须是那个类的中心（centroid），
- Kmeans中的H同样表示各个数据点分配到各个类的权重，只不过H的每一列只有一个非0元素（即每个数据点被hard label到唯一的一个类）。
- 而Non-negative Matrix Factorization则放宽了这些要求，W中的每一列向量虽为每个类的basis vector，但不必是centroid；
- H中的每一列向量并不只能有一个非0的元素，即*****NMF为soft label形式的聚类算法。*****
- 由paper《Sparse Nonnegative Matrix Factorization for Clustering》可知，K-means其实可以认为是NMF的一种特例？
-- In K-means, the factor W = AF is the centroid matrix with its columns rescaled and the factor H has exactly one nonzero element for each row. 
-- i.e., its row represents a hard clustering result of the corresponding data point.
-- But in NMF, NMF formulation does not have these constraints, and this means that basis vectors (the columns in W) of NMF are not necessarily the centroids of the clusters, 
-- and NMF does not force hard clustering of each data point. （NMF 给予的是当前record属于某个类别的概率）
-- When a basis vector is close to a cluster center, data points in that cluster are easily approximated by the basis vector only (or with a small deviation of another basis vector). 
-- As a result, we can safely determine clustering assignment by the largest element of each row in H.
-- NMF中的H，一般而言是存在超过一个非零元素。
-- NMF本质上为soft label形式的聚类算法。
-- K-means可以认为是NMF的hard label的特殊形式；
- 如果想让每个数据点不只属于一个类，但也不想让每个数据点由太多类“组成”，Sparse NMF应运而生：
-- 数据特点：H中的每一列由尽可能少的非0元素构成。
-- 数学表达：SNMF的构成就是给W和H各自加上L2正则；（个人觉得应该加上L1正则才能达到稀疏效果呀！！）

8、SVD和PCA的区别及联系
- 我们一般说的PCA是侠义的基于特征值分解的矩阵分解方法，进行基于特征值进行数据压缩；
- 但其实我们用的更多的是基于奇异值分解的矩阵分解方法，同样基于奇异值也能进行数据压缩，而且压缩效果、计算速度和应用性方面更广。
- 高斯归一化是数值减去均值，再除以标准差。而中心化，是指变量减去它的均值。
- 为什么PCA一开始要做中心化：因为这样就能用简便方法计算协方差矩阵：
-- 假设原dataSet去中心化后为X，则原dataSet的协方差矩阵为X*X^T，再除以1/(m-1)，假设为m维；（实际应用中并没有除以1/(m-1)）
- PCA只是对列进行压缩，即对相关性很强的特征进行压缩。推导如下：
-- X为dataSet，那么X^T * X  为近似协方差矩阵（和真正的协方差矩阵只是相差1/(m-1)）
-- X^T * X得到方阵的维数和X的列数一样。
-- 在此基础上进行基于特征值的分解，等价于只对X使用右奇异矩阵，而未使用左奇异矩阵；
-- 左奇异矩阵的作用在于对行数进行压缩。
-- 详细的推导过程 和 推导奇异值矩阵的特征值矩阵的开方一样。需要注意的是一定是X^T * X。
-- 注意到我们的SVD也可以得到协方差矩阵X^T * X最大的d个特征向量张成的矩阵，
-- 但是SVD有个好处，SVD的实现算法可以不求先求出协方差矩阵X^T * X，也能求出我们的右奇异矩阵V。
-- 也就是说，我们的PCA算法可以不用做特征分解，而是做SVD来完成。
-- 这个方法在样本量很大的时候很有效。实际上，scikit-learn的PCA算法的背后真正的实现就是用的SVD，而不是我们我们认为的暴力特征分解
-- scikit-learn的PCA算法是基于以下svd_solver实现的：
	1. full SVD
	2. TSVD
	3. randomized SVD
- 为什么用SVD而不用基于特征值的PCA？
-- PCA的所有的缺点都来自于需要计算X^T * X这一步：
	1. X^T * X 会使矩阵一些很小的值在平方中损失，或者失去精度（比如e）
	2. X^T * X 占用的空间不X要更大，计算效率也更低；

9、为什么MF可以用user的latent factor 与 item的latent factor的内积作为user对item的喜好程度判断？
- 内积本质上反应的是两个向量的相似度到底有多高？如果user对物品某些特质很喜爱，某item在这些特质上表现明显，那么这个user就会喜欢这个item;
- 在数学上的表现就是内积的值大（本质就是夹角够小）
- 同理，不同user对同一个item喜好的偏差，可以用不同user的latent factor与该item的latent factor的内积来量化比较。


**1** PCA的本质是什么？线性变换？还是通过线性变化进行数据降维？
**2** PCA的优缺点还需要详细？必须强撸一遍，把本质思想撸清楚。比如如何基于特征值的变化做到数据压缩等等。
**3** SVD的优缺点？
"""






























