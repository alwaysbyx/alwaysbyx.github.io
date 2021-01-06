---
layout: post
title:  "Image Style Trasfer Review"
date:   2021-01-06 22:03:36 +0530
categories: CNN
---

Many thanks to *Image Style Transfer Using Convolutional Neural Networks*  

# Theory
## Content representation
$$L_{content(\vec{p},\vec{x},l)} = \frac{1}{2}\sum_{i,j}(F_{ij}^l-P_{ij}^l)^2$$  
define the squared-error loss between the two feature representations, l means layer  
## Style representation
To obtain a representation of the style of an input image, we use a feature space designed to capture texture information. This feature space can be built on **top of the filter responses in any layer** of the network. It consists of the correlations between the different filter responses, where the expectation is taken over the spatial extent of the feature map.
These feature **correlations** are given by the **Gram matrix** $G^l \in R^{N_l \times N_l}$
, where $G^l_{ij}$ is the inner product between the vectorised feature maps i and j in layer l:  
$$G_{ij}^l = \sum_kF_{ik}^lF_{jk}^l$$  

By including the feature correlations of multiple layers, we obtain a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangemen.  
理解：可以看作feature的偏心协方差矩阵（不减去均值），feature map指的是滤波器在特定位置得到的滤波值，gram计算的是feature不同位置作用的不同强度。  
### cross-covariance matrix
$$ cov(X,Y) = E[(X-E[X])(Y-E[Y])^T] $$  
指的是变量同向程度  
## Style transfer
A random white noise image $\vec{x}$ is passed through the network and its style features $G_l$ and content features $F_l$ are computed. Then try to minimize the total loss.
$$L_{total}(\vec{p},\vec{a},\vec{x}) = aL_{content}(\vec{p},\vec{x})+bL_{style}(\vec{a},\vec{x})$$  
# Methods
## VGG16
proposed by *Very Deep Convolutional Networks for Large-Scale Image Recognition*.  
Main contribution is a thorough evaluation of networks of increasing depth using an architecture with very *small (3 × 3) convolution filters*, which shows that a significant improvement on the prior-art configurations can be achieved by *pushing the depth* to 16–19 weight layers.
## using LBFGS to minimize the loss
an optmization algorithm in the family of quasi-Newton methods that approxicates the BFGS using a limited of computer memory  
 
# Implementation
basic functions to compute loss
```python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 我们从用于动态计算梯度的树中“分离”目标内容：
        # 这是一个声明的值，而不是变量。 
        # 否则标准的正向方法将引发错误。
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # 特征映射 b=number
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # 我们通过除以每个特征映射中的元素数来“标准化”gram矩阵的值.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
```
generate models
```python
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# VGG网络通过使用mean=[0.485, 0.456, 0.406]和std=[0.229, 0.224, 0.225]参数来标准化图片的每一个通道，并在图片上进行训练
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # 规范化模块
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # 只是为了拥有可迭代的访问权限或列出内容/系统损失
    content_losses = []
    style_losses = []

    # 假设cnn是一个`nn.Sequential`，
    # 所以我们创建一个新的`nn.Sequential`来放入应该按顺序激活的模块
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # 对于我们在下面插入的`ContentLoss`和`StyleLoss`，
            # 本地版本不能很好地发挥作用。所以我们在这里替换不合适的
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 加入内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 加入风格损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
        # 现在我们在最后的内容和风格损失之后剪掉了图层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
```
start to perform image style transfer
```python
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # 更正更新的输入图像的值
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # 收敛到[0,1]
    input_img.data.clamp_(0, 1)

    return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,content_img, style_img, input_img)

```