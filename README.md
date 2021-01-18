# Normalization-Layers
**Numpy：**完成batch normalization的基层实现：前向传播和反向传播

题目要求：

1.  题目1：实现训练过程中batch normalization 层的前向传播；实现测试过程中batch
    normalization层的前向传播[1]（20分）

2.  题目2：实现batch normalization层的反向传播，将结果保存在dx, dgamma,
    dbeta三个变量中（20分）

**PyTorch：**利用PyTorch中常用的normalization函数实现BN[5]，LN[6]，IN[8]，GN[7]及改进ResNet[3]在CIFAR数据集上的表现[2]

题目要求：

1.  题目1：根据对BN，LN，IN，GN的理解，完成BLOCK
    1.1和1.2两处的程序填空并正确运行（20分）

2.  题目2：完成去掉normalization的程序版本并比较实验结果，同时自行查阅资料分析normalization的作用
    （20分）

3.  题目3：查阅资料了解CIFAR10，CIFAR100数据集上:ResNet表现，改进该网络中不适用于CIFAR数据集的结构（20分）

    //斜线内正式采用时删掉 

    其中一些可改进的问题：

    1.  第一层卷积kernel_size过大，不适用于CIFAR10/100小尺寸图片

    2.  同理，第一层卷积后maxpool过早
    
    //

附加题bonus：基本复现ResNet在CIFAR上的最好评分
    
    

**数据集：**

1.  CIFAR-10数据集：包含60000张32\*32大小的彩色图片（一共十类，即每类6000张），分为50000张训练图片和10000张测试图片。数据集被分为五个训练batch，一个测试batch，每个batch包含10000张图片，测试集包含从每一类随机选取的1000张图片；训练集是从剩下的图片中随机选取的，有些训练batch可能每一类选取的图片数量不平均。
    10类间是相互独立的，如truck（仅包含big trucks，没有pickup
    trucks）和automobile（包含sedans、SUVs等）之间没有重叠。

2.  CIFAR-100数据集：和CIFAR-10类似，不过该数据集共包含100类，每类600张图片；每类500张训练图片，100张测试图片，其中100类又被分组为20个超类，每张图片带有一个“fine”标签（表示该图片属于哪一类）和一个“coarse”标签（表示该图片属于哪一超类）。

**参考文献：**

1.  Thakkar V, Tewary S, Chakraborty C. Batch Normalization in Convolutional
    Neural Networks—A comparative study with CIFAR-10 data[C]//2018 Fifth
    International Conference on Emerging Applications of Information Technology
    (EAIT). IEEE, 2018: 1-5.

2.  Krizhevsky A, Hinton G. Convolutional deep belief networks on cifar-10[J].
    Unpublished manuscript, 2010, 40(7): 1-9.

3.  Srivastava R K, Greff K, Schmidhuber J. Training very deep networks[J].
    Advances in neural information processing systems, 2015, 28: 2377-2385.

4.  Abouelnaga Y, Ali O S, Rady H, et al. CIFAR-10: KNN-based Ensemble of
    Classifiers[C]//2016 International Conference on Computational Science and
    Computational Intelligence (CSCI). IEEE, 2016: 1192-1195.

5.  Santurkar S, Tsipras D, Ilyas A, et al. How does batch normalization help
    optimization?[C]//Advances in neural information processing systems. 2018:
    2483-2493.

6.  Ba J L, Kiros J R, Hinton G E. Layer normalization[J]. arXiv preprint
    arXiv:1607.06450, 2016.

7.  Wu Y, He K. Group normalization[C]//Proceedings of the European conference
    on computer vision (ECCV). 2018: 3-19.

8.  Ulyanov D, Vedaldi A, Lempitsky V. Instance normalization: The missing
    ingredient for fast stylization[J]. arXiv preprint arXiv:1607.08022, 2016.
