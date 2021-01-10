---
layout: post
title:  "Notes on contrastive learning"
date:   2021-01-04 21:03:36 +0530
categories: ComputerVersion models
---




<div align=center><img src="https://alwaysbyx.github.io/assets/cnn.png" style="zoom:50%"></div>

# CNN
1. feature map (aka. avtivation map)  
input 32\*32\*3 filter 5\*5\*3 -> output 28\*28\*1  
if we have two filters, then output is 28\*28\*2  
output_size = 1+ (input_size+2\*padding-kernel_size)/stride
2. Sub-sampling layer (aka. pooling)  
功能：缩聚特征图谱矩阵，同时保留特征图谱矩阵内的关键信息。  
methods: max, average, sum; exp. max pool with 2\*2 filters and stride 2  
3. CNN construction  
> input -> Conv -> ReLU -> Conv -> ReLU -> Pool -> ReLU -> Conv -> ReLU -> Pool -> FullyConnected  

第一层Conv得到activation map, 提取低层特征（边缘曲线etc); 用ReLU激活后，进入第二层Conv, 从低层次特征提取高层次特征  

# Resnet, Deep Residual Learning
Adding more layers (deep learning) leads to a higher training error, and to solve this problem, Kaiming He,etc proposed Resnet.  
> There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart.  

![Resnet construction](https://alwaysbyx.github.io/assets/resnet01.png)  

Intuition: improve the ability of identity mapping：一个block的输入可以等于这个block的输出，则更深层次的网络不应该出现比浅层次网络更差的退化问题  
F(x) = 0 只需要weight为0即可

### vanishing/exploding gradients  
原因：
1. 网络长度过长
2. 反向传播链式求导
3. 激活函数的最大值为0.25  

**梯度消失**是因为反向传播中对梯度的求解会产生sigmoid导数和参数的连乘，而sigmoid导数最大为0.25，初始权重小于1，导致靠近输入层的梯度几乎为0，得不到更新。  
如果初始权重大于1，多个大于1的数相乘，**梯度爆炸**。  

There are 5 types of Resnet:  
* Res18
* Res34
* Res50
* Res101
* Res152  
<div align=center><img src="https://alwaysbyx.github.io/assets/resnet.png" style="zoom:35%"></div>

ResNet50起，采用Bottleneck结构，引入1\*1卷积核，可以：  
* 对通道数进行升维和降维（跨通道信息整合），实现了多个特征图的线性组合，同时保持了原有的特征图大小；
* 相比于其他尺寸的卷积核，可以极大地降低运算复杂度；
* 如果使用两个3x3卷积堆叠，只有一个relu，但使用1x1卷积就会有两个relu，引入了更多的非线性映射；

# Exploring SimCLR
<img src="https://alwaysbyx.github.io/assets/cl.png" style="zoom:80%" />  

The core of contrastive learning is the Noise Contrastive Estimator (NCE) loss.
<img src="https://alwaysbyx.github.io/assets/nceloss.png" style="zoom:50%" />  

> SimCLR uses ResNet-50 as the main ConvNet backbone. The ResNet receives an augmented image of shape (224,224,3) and outputs a 2048-dimensional embedding vector **h**. Then, a projection head **g(.)** is applied to the embedding vector **h** which produces a final representation z=g(h). The projection head **g(.)** is a Multilayer Perceptron (MLP) with 2 dense layers. Both layers have 2048 units and the hidden layer has a non-linearity (ReLU) activation function.  

<div align=center><img src="https://alwaysbyx.github.io/assets/cl01.png" style="zoom:30%" ></div>  

> To train SimCLR, I took the **train + unlabeled** portions of the dataset – that gives a total of 105000 images.  
After training, we need a way to evaluate the quality of the representations learned by SimCLR. One standard way is to use a linear evaluation protocol.  
The idea is to train linear classifiers on fixed representations from the SimCLR encoder. To do that, we take the training data, pass it through the pre-trained SimCLR model, and store the output representations. Note that at this point, we do not need the projection head g(.) anymore.  
These fixed representations are then used to train a Logistic Regression model using the training labels as targets. Then, we can measure the testing accuracy, and use it as a measure of feature quality.  
This Jupyter Notebook shows the evaluation protocol. Using the SimCLR fixed representations as training signals, we reach a test accuracy of 64%. To have an idea, performing PCA on the training data and keeping the most important principal components, we get a test accuracy of only 36%. This emphasizes the quality of the features leaned by SimCLR.  
From [simple self-supervised learning](https://sthalles.github.io/simple-self-supervised-learning/)




