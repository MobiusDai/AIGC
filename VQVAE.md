### 总览

**出发点**：在无监督的情况下学习图像的表征

**与VAE的不同：**

- 编码器端出输出离散的代码本
- 先验信息也是学习得到的而非事先指定

**优点：**

- 避免VAE的 ‘后验崩溃’ 问题

```
后验坍塌：在传统的VAE（变分自编码器）中，我们尝试通过将输入数据编码为潜在空间的低维表示，并通过解码器将其还原回原始数据。然而，当解码器非常强大时，它有能力几乎完美地重建输入数据，这导致了一个问题。该问题被称为“后验坍塌”，它指的是潜在变量被忽视，因为解码器足够强大，可以独立地生成所有信息，而无需利用潜在变量。
```

VQ-VAE引入了向量量化（VQ）方法来解决这个问题。在VQ-VAE中，编码器将输入数据映射到离散的潜在向量空间。这个向量空间由一组称为“代码本”的向量表示。代码本是由训练数据自动学习得到的。

```
VQ：当输入数据被映射到潜在向量空间时，它们被强制与最接近的代码本向量进行匹配。也就是说，每个潜在向量都被替换为其最接近的代码本向量。这种替换操作称为“向量量化”。
```

通过向量量化，VQ-VAE可以强制编码器利用潜在向量空间中的代码本向量表示输入数据。解码器则负责将量化后的向量映射回原始数据。由于解码器不能直接生成输入数据，它必须依赖于潜在向量空间中的代码本向量，从而避免了后验坍塌的问题。

这样做可以增强模型对潜在变量的利用，从而提高模型的表达能力和生成能力。

### Method

```
VAE:
	img -> encoder -> latent variables z
	encoder: q(z|x) 
	decoder: p(x|z)
	prior  : p(z)
posteriors and priors in VAEs are assumed normally distributed with diagonal covariance
```

**VQVAE**

![arch](/Users/jerry/Blog/AIGC/imgs/vqvae.png)

**1. 离散隐变量**

*latent embedding space*: $e \in R^{K\times D}$, K表示离散空间向量的数量，D表示隐向量的维度

$input:x \rightarrow encoder \rightarrow z_e(x)$

$z_e(x) \rightarrow nearset neightbor \rightarrow e_k$

$q(z=k|x) = \begin{cases}1, & for\space k ==argmin_j\left \| z_e(x)-e_j \right \| \\ 0, &otherwise\end{cases}$

$p(z) \sim Uniform([0, 1, \dots , K])$ 

$KL(q(x=k|x)\|p(z)) = \mathit{logK}$



**2. 参数学习**

VQVAE使用类似于直通估计器（straight-through estimator）的方法来近似梯度，将梯度从解码器的输入$Z_q(x)$复制到编码器的输出$Z_e(x)$.

在前向计算过程中，最相似的的嵌入$Z_q(x)$被传递给解码器，而在反向传播过程中梯度$\nabla_z L$ 以原样传递给编码器。由于编码器的输出表示和解码器的输入共享相同的D维空间，梯度包含了有关编码器如何改变其输出以降低重构损失的有用信息。

*loss function:* $\mathbb{L} = log \mathit{p(x|z_q(x))} + \left \| sg[z_e(x)] -e\right \|^2_2 +\beta \left \| z_e(x) -sg[e] \right \|$

***sg** stands for the stopgradient operator*

```
第一部分：log p(x|zq(x))
这一项表示给定潜在变量zq(x)，通过解码器生成的重构数据与原始输入数据x之间的对数似然。它衡量了解码器在给定潜在变量的情况下，对输入数据进行重建的能力。如果这一项的值越大，说明重构的数据越接近原始输入数据。

第二部分：||sg[ze(x)] - e||^2_2
这一项表示编码器的输出ze(x)与预先定义的目标向量e之间的欧氏距离的平方。向量量化的目标是通过最小化L2误差来调整代码本向量，使其尽可能接近编码器的输出。或使用exponential moving averages(EMA)来更新。

第三部分：β||ze(x) - sg[e]||
这一项是一个正则化项，其中β是一个调节系数。它衡量了编码器的输出ze(x)与目标向量sg[e]之间的欧氏距离。通过这一项，我们鼓励编码器的输出在潜在空间中保持与目标向量的一致性，以促进更好的表示学习。

综合起来，该损失函数的目标是最大化重构数据的对数似然，同时使编码器的输出尽可能接近目标向量，并在潜在空间中保持一致性。调节系数β可以控制第二部分和第三部分之间的权衡关系，从而平衡重建性能和表示学习的优化过程。通过最小化这个损失函数，我们可以训练VQ-VAE模型来获得更好的数据重建和表示学习性能。
```

