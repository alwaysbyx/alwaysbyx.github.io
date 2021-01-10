---
layout: post
title:  "System Control via Neural Network"
date:   2021-01-09 22:03:36 +0530
categories: Control System NeuralNetwork
---

Many thanks to *OPTIMAL CONTROL VIA NEURAL NETWORKS: A CONVEX APPROACH*  

# Main contribution
To strike a balance between requiring painstaking **manual construction of physics based models** and the risk of not capturing rich and complex system dynamics through **models that are too simplistic**.  
### why using linear models rather than neural network
Based on *Deep Learning without Poor Local Minimal by Kenji Kawaguchi*, squared loss function of deep linear neural networks with any depth and any widths is **non-convex** and **non-concave**. Thus It is difficult to directly train a deep model in theory. Also, there is  nonexistence of poor local minimal.   
- convex function
$$g(\frac{1}{k}\sum_{i=1}^kx_i) <= \frac{1}{k}\sum_ig(x_i))$$
if g() is a convex function.  
Since there exists $\prod_{i=1}^kn_k!$ points in the parameter space that can achieve that local optima, where $n_k$ is the number of nodes in the $k^{th}$ layer.   
The function evaluation at the average is less than or equal to the average of these optimal points, which are all equal and hence what we get is that the evaluation at the average point is also optimal.  
In a words, Neural nets also have many symmetric configurations. This symmetry means they can’t be convex. (http://www.cs.cornell.edu/courses/cs6787/2017fa/Lecture7.pdf)
- What makes non-convex optimization hard  
(1) Saddle points  
(2) Very flat regions  
(3) Widely varying curvature  
<div align=center><img src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Saddle_point.svg" style="zoom:50%"></div>  

- Why are neural networks non-convex  
Composition of convex functions is not convex, so deep neural networks also aren’t convex

# Method
## Using convex neural network
Tackle **modeling accuracy and control tractability tradeoff** by building on the **input convex neural networks (ICNN)** in (Amos et al., 2017) to both represent system dynamics and to find optimal control policies.