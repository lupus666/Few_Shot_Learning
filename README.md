# Few_Shot_Learning
学习孪生网络进行图像验证*Image Verification*以及尝试着进行改进并进行一些学习总结，其中网络的实现参考[*https://github.com/akshaysharma096/Siamese-Networks*](https://github.com/akshaysharma096/Siamese-Networks)，相关的改进方法参考原论文[*Siamese Neural Networks for One-Shot Image Recognition*](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
## N类别单样本学习（*One-shot Learning*）
- 单样本学习，区别于传统观点认为的基于大量标记样本训练的深度神经网络，只需要少量的样本进行训练便能获得较好的效果（更贴近于人类的学习能力）  
- N类别单样本学习，即要解决，给定一个*Test Image*，以及N个样本的*Support Set*
，如何正确地预测其所属类别

### 1-邻近基准
1-邻近(*1-nearest neighbour*)可以作为单样本学习的一个简单的*baseline*，即通过计算筛选出测试样本与训练集中每个样本的最小欧几里得距离，即可获得一个预测结果。虽然在[*Omniglot*](https://github.com/brendenlake/omniglot)数据集中的20分类上，1-邻近只获得了约28%的精度，但是也比随机猜测的5%要好上不少

### 孪生网络*Siamese networks*
基本思想是通过同时向神经网络输入两张图片，获得两张图片同属一个类别的概率的输出，而又因为**对称性**，即输入的两张图片的顺序颠倒不影响其输出的结果，因此不能简单地将两张图片拼接起来。孪生网络正是依赖于对称性而设计的。

#### 网络架构
如图所示
![image](https://camo.githubusercontent.com/b27757e11d8687dc846b016e0fac80a544e7b645/68747470733a2f2f736f72656e626f756d612e6769746875622e696f2f696d616765732f5369616d6573655f6469616772616d5f322e706e67)

#### 图像输入处理
在训练的每一个epoch中，输入两组图片inputs以及一组目标数据targets，其中一组全为同一类别，另一组为一半其他类别，一半该类别，而targets为一半为0一半为1的一维数组，因为每一个样本与另一个样本的配对是完全随机的，因此需要较大的迭代训练次数

#### 可选改进方法
在[原论文](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)中，该孪生网络运用了许多提高性能的方法，而[该实现](https://github.com/akshaysharma096/Siamese-Networks)中并没有使用这些方法
- 逐层的学习率和冲量*momentum* ，虽然用*keras*实现逐层的学习率和冲量较为麻烦，但依然是一个可选的改进方向
- 数据失真的数据增强方法
- 贝叶斯超参数优化（[*Bayesian Optimization*](https://en.m.wikipedia.org/wiki/Bayesian_optimization)）
- 很大的epochs次数  

实际上，贝叶斯超参数优化可以实现很大一部分的改进

#### 所尝试的改进方法
因为[该实现](https://github.com/akshaysharma096/Siamese-Networks)已经给出较为适合的超参数，以及对贝叶斯优化方法不了解，所以就没有针对超参数进行改进，而只进行了较为简单的改进方法
- 增加batch_size（增加迭代次数）
- 采用数据增强的方法

#### 结果
- 原始结果  
  ![base](https://github.com/lupus666/Few_Shot_Learning/raw/master/image/base.png)
- 增加batch_size（32 to 64）  
  ![batch_size](https://github.com/lupus666/Few_Shot_Learning/raw/master/image/batch_size.png)
- 在增加batch_size基础上采用数据增强  
  ![data_aug](https://github.com/lupus666/Few_Shot_Learning/raw/master/image/data_aug.png)  

可以看到增加batch_size稍有提升，但是三者几乎没有多少差别，从某种程度上来看，数据增强甚至起到了反作用，但有可能也是因为epochs不够多。而且三者都存在着过拟合的情况，虽然已经使用了l2正则化的方法，但依然可以尝试着使用*Dropout*的方法

