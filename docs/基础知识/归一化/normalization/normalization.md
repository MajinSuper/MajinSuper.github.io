---
title: 归一化（Normalization）
tags:
  - batch norm
  - layer norm
  - RMS norm
createTime: 2025-09-05 09:44:31
permalink: /article/normalization/
---
## 归一化

### 什么是归一化？

对==神经元的输入==进行处理，使其符合==标准分布==

::: center
$\bar{z} = \frac{z-\mu}{\sigma}$

$\hat{z} = w * \bar{z} + b$

其中，$\mu$、$\sigma$分别为均值、标准差，$w$、$b$为可学习参数
:::

### 为什么需要归一化？

- **角度一**：==无量纲化==。==不同维度的值，可以统一使用==。比如：175cm和60kg特征取值返回、含义均不一致，无法直接运算
- **角度二**：==提高训练收敛效率==。
  - loss图上，正圆(归一化) 相对于 扁圆 收敛更快
  - 解决==梯度消失==。在使用sigmoid激活时，如果输入离0特别远，会出现梯度消失。落在0的附近，可有效缓解。
  - 解决==梯度爆炸==。对神经元参数($z=w*x+b$),求导数为$\frac{\partial z}{\partial w}  = x$，x的值如果特别大，累乘下去很容易出现梯度爆炸。
  - ==减小参数初始化的影响==。使模型不过分受参数初始化方法的影响
  - ==内部协向量偏移==。每一层的输入是上一层的输出，如果上一层的输出分布不固定，那么基于这个数据分布而训练往往会“不稳定”。层数越多，这种内部偏移会累加。
- **角度三**: ==训练和测试能对齐==
  - 能够使训练和测试阶段，各个神经元输入分布一致

::: note 回看
   ==角度一==并不是神经网络归一化的原因，这是==数据标准化==的操作
:::

### 归一化方法：

|          |                                                          BatchNorm                                                          |                  LayerNorm                  |                              RMSNorm                              |
| :------: | :--------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------: | :---------------------------------------------------------------: |
|   位置   |                                                   pre-norm(输出后，激活前)                                                   |                  pre-norm                  |                             pre-norm                             |
|   公式   |                                        $\frac{z_i-\mu}{\sigma} * \gamma_i+ \beta_i$                                        | $\frac{z_i-\mu}{\sigma}*\gamma_i+\beta_i$ | $\frac{z_i}{\sqrt{\frac{1}{n}\sum_{i=0}^{n} z_i^2}} * \gamma_i$ |
|   输入   |                                                      [B,N] 或者 [B,N,L]                                                      |             [B,N] 或者 [B,L,N]             |                        [B,N] 或者 [B,L,N]                        |
|   输出   |                                                              -                                                              |                      -                      |                                 -                                 |
|   内存   |                                                             2*N                                                             |                     2*N                     |                                 N                                 |
| 均值形状 |                                                    [1, N] 或者 [1, N, 1]                                                    |            [B, 1] 或者 [B, L, 1]            |                                 -                                 |
|   心得   |                                               “某个特征，所有batch内的样本”                                               |       “某个样本，所有样本内的特征”       |                  “某个样本，所有样本内的特征”                  |
|   备注   |                                        记录特征的全局均值和方差<br />使用滑动平均方式                                        |                                            |              提升计算效率<br />主流大模型多数都是用              |
|   优劣   | ICS、收敛更快、梯度消失/爆炸<br />抹掉特征间的差距（不同特征的大小关系被破坏）<br />训练时Batch Size不能太小，推理时速度最快 | + 适应序列数据<br />抹掉了样本间的特征差距 |                                                                  |

::: note 注释

“样本”含义，

- 2d时，每个样本即为batch中每个样本（有B个）；
- 3d时，每个样本为batch中==每条数据==的==每个词==（有B*L个样本）

“特征”含义，

- nlp中，特征i 为 hidden_dim的第i个值（有N个）；
- cv中，特征i 为  channek的第i个值（有C个）。

“seq_len”与“H，w”，

- 在CV的==H，W==合起来 对标 NLP中的 ==seq_len==。

:::

::: tip 常见面试题

:::

::: details 问题1. 为什么RNN或者transformer使用的是LN？而不是BN？

回答：

有些场景上，不适合使用BN。比如：batch很小、序列问题。

序列问题上，样本的序列长度不一致，会存在padding。在padding后计算特征的BN，结果失真

:::

### 1. BatchNorm

```python

class MyBatchNorm1d(torch.nn.Module):
    def __init__(self, feature_num: int, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.feature_num = feature_num
        self.eps = eps  # 防止除0

        # moving average
        self.momentum = momentum
        self.running_mean = torch.zeros(feature_num)  # [N]
        self.running_var = torch.ones(feature_num)  # [N]

        # 仿射变换
        self.affine = affine
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(feature_num))  # [N]
            self.beta = torch.nn.Parameter(torch.zeros(feature_num))  # [N]

    def forward(self, x):
        """
        x: [B，N] or [B, N, L]
        """

        if x.dim() == 2:
            reduce_dims = [0]
            new_shape = [1, x.shape[1]]
        elif x.dim() == 3:
            reduce_dims = [0, 2]
            new_shape = [1, x.shape[1], 1]
        else:
            raise ValueError(f"Excepted 2D or 3D tensor , but get {x.dim()}")

        if self.training:
            mean_value = x.mean(dim=reduce_dims, keepdims=True)  # [1,N] 、[1,N,1]
            var_value = x.var(dim=reduce_dims, unbiased=False, keepdims=True)  # [1,N] 、[1,N,1]

            x_norm = (x - mean_value) / torch.sqrt(var_value + self.eps)  # [B,N]、 [B,N,L]

            # moving mean and var
            self.running_mean = self.running_mean * (1 - self.momentum) + mean_value.view(-1) * self.momentum  # [N]   
            self.running_var =  self.running_var * (1 - self.momentum)  + var_value.view(-1) * self.momentum # [N]
        else:
            x_norm = ( x - self.running_mean.reshape(new_shape) ) / self.running_var.reshape(new_shape)

        if self.affine:
            return x_norm * self.gamma.reshape(new_shape) + self.beta.reshape(new_shape)
        else:
            return x_norm

```

### 2. LayerNorm：

```python
class MyLayerNorm(torch.nn.Module):
    def __init__(self,batch_size,eps=1e-5,affine=True) -> None:
        super().__init__()
        self.eps = eps

        # 仿射变换
        self.affine =  affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(feature_num)) #[N]
            self.bias = torch.nn.Parameter(torch.zeros(feature_num)) #[N]

    def forward(self,x):
        '''
            x : [B, L, N] or [B, N]
        '''
        mean_value = x.mean(dim=-1,keepdim=True) # [B, 1] [B, L, 1]
        var_value = x.var(dim=-1,unbiased=False,keepdim=True) # [B, 1] [B, L, 1]

        x_norm = (x - mean_value) / torch.sqrt(var_value+self.eps) # [B, N] [B, N, L]

        if self.affine:
            return x_norm * self.weight + self.bias
        else:
            return x_norm
```

### 3. RMSNorm:

```python
class RmsNorm(torch.nn.Module):
    def __init__(self, norm_shape, eps=1e-6,affine=True):
        super().__init__()
        self.eps = eps
  
        # 仿射变换
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(norm_shape)) # [N]

    def forward(self, x):
        '''
            x: [B, L, N]
        '''
        var = torch.mean(x.pow(2),dim=-1,keepdim=True)  # [B, L, 1]
        x = x / torch.sqrt(var + self.eps) # [B, L, N]
        if self.affine:
            return x * self.weight
        else:
            return x
```

### 应用（CV与NLP）

![BN_LN.png](/images/BN_LN.png)

CV中一般使用BN，

NLP中一般使用LN
