---
layout: post
title:  "notes on contrastive learning"
date:   2021-01-04 21:03:36 +0530
categories: computer version, models
---



### CNN
![ijiangtao](https://alwaysbyx.github.io/assets/cnn.png)
1. feature map (aka. avtivation map)
input 32\*32\*3 filter 5\*5\*3 -> output 28\*28\*1  
if we have two filters, then output is 28\*28\*2
2. Sub-sampling layer (aka. pooling)
功能：缩聚特征图谱矩阵，同时保留特征图谱矩阵内的关键信息。
methods: max, average, sum; exp. max pool with 2\*2 filters and stride 2
3. CNN construction
> input -> Conv -> ReLU -> Conv -> ReLU -> Pool -> ReLU -> Conv -> ReLU -> Pool -> FullyConnected
第一层Conv得到activation map, 提取低层特征（边缘曲线etc); 用ReLU激活后，进入第二层Conv, 从低层次特征提取高层次特征  


### Resnet
Adding more layers (deep learning) leads to a higher training error, and to solve this problem, Kaiming He,etc proposed Resnet.  
> There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart.  
#### vanishing/exploding gradients

There are 5 types of Resnet:  
1. Res18
2. Res34
3. Res50
4. Res101
5. Res152  



