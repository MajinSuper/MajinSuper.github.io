---
title: 逐层分解transformer
tags:
  - transformer
  - 基础
---

## 1.介绍

### 1.1 整体架构

::: center
![transformer介绍.png](/images/transformer介绍.png)
:::

**transformer架构**如上如图所示，是==seq2seq的网络结构（序列转录、sequence transduction）==，是由==Encoder==和==Decoder==两部分组成。

::: tip 常见面试题
:::

::: details 问题1. transformer优势是什么？RNN遇到了什么问题？
回答：
RNN存在并行计算差、遗忘现象严重两个主要问题
- ==并行计算差==：处理t时刻的输入，需要先完成t-1及其之前的所有输入。
- ==遗忘现象严重==: 当输入长度很长时，仅靠$h_t$来存储t时刻及之前的信息，很容易出现遗忘。即便$h_t$用了很大的size。

tansformer又快又好：
- ==并行计算==：Encoder可并行化计算所有attention，Decoder在训练阶段也是并行化的。
- ==效果上更好==：在原文的机器翻译任务上，是取得了非常好的效果。目前在更多工作上，也验证了transformer的优越性。
:::

::: details 问题2. transformer为什么可以并行化？
回答：
这里，并行化是指某条==样本内==所有输入，并行化计算。
Encoder部分：
- QKV：QKV矩阵的获取，是同时获取的，一次针对（样本内的）所有输入
- Attn：attention矩阵计算是并行的，一次针对所有输入
- MHA：多头之间也是并行的
- FNN：也是一次针对所有输入
Decoder部分：
- ==训练阶段==：引入了"teacher force"的概念，每个时刻==不依赖上一时刻的输出==，而==依赖正确样本==。
- ==测试阶段==：在测试阶段，不存在真实label，采用==自回归==的方式，还是==依赖上一时刻的输出==。
:::

  
### 1.2 基本流程
1. **分词**：对输入文本进行分词 `string -> [token_1,token_2 ... token_n]`
2. **词嵌入**：每个词使用向量表达 `[token_1,token_2 ... token_n] -> [x_1,x_2 ... x_n]`
3. **位置编码**：每个词的词嵌入 + 其位置编码 `[x_1,x_2 ... x_n] + [pe_1, pe_2 ... pe_n] -> [\hat{x}_1, \hat{x}_2... \hat{x}_n]`
4. **Encoder**: 输入给Encoder进行计算 `[\hat{x}_1, \hat{x}_2... \hat{x}_n] -> [z_1, z_2 ... z_n]`
5. **Decoder**: Encoder计算结果再传给Decoder `[z_1, z_2 ... z_n] -> y_1`
6. **Decoder**: Encoder+Decoder结果继续传给Decoder `[z_1, z_2 ... z_n,y_1] -> y_2`
7. 重复直至结束

## 2. Transformer的输入

### 2.1 词嵌入
词嵌入的方式有很多，可以是word2vec、Glove等算法训练得到，也可以是**learnable**，与transformer一同训练得到

### 2.2 位置编码 💖
标准的位置编码是

::: center
$PE_{(pos,2i)}=sin(pos / 1000^{2i/d})$

$PE_{(pos,2i+1)}=cos(pos / 1000^{2i/d})$
:::

其中,$pos$为==句中的位置==，$d$表示==位置嵌入的维度==（等于==词嵌入的维度==），
$2i$表示位置嵌入的==偶数维度==，$2i+1$表示==奇数维度==，且$2i<d,2i+1<d$


::: tip 常见面试题
:::

::: details 问题1. transformer为什么要使用位置编码？
:::

::: details 问题2. 设计使用sin、cos的原因是什么？
:::

::: details 问题3. 这种位置编码的好处和坏处分别有什么？
:::


## 3. Transformer的架构 

### 3.1 Scaled Dot-Product Attetion
query和key是同一大小的维度$d_k$，value的维度是$d_v$

::: center
![scaled_dot_product_attetion.png](/images/scaled_dot_product_attetion.png)

**计算公式**：

$Attention(Q,K,V) = softmax(\frac{Q*K^T}{\sqrt{d_k}})*V$

:::


::: tip 常见面试题
:::

::: details 问题1. 为什么要除以$\sqrt{d_k}$
回答：
当$d_k$的维度特别大时，$q*k$的值会变得特别大，导致softmax输入趋近于两端，会出现==梯度消失==
:::

### 3.2 多头注意力机制-MHA
**动机**：类比CNN的多通道，能够==捕获多个不一样的模式==，实现多个输出通道

::: center
![img.png](/images/MHA.png)

进行h次不同的投射，产生Q_i,K_i,V_i(i=1...h)。计算自注意力之后的结果，进行==拼接==。


**计算公式**：
$Multi-Head(Q,K,V) = concat(head_1,head_2,……,head_h)*W^O$

$head_i = Attention(Q*W^Q_i,K*W^K_i,V*W^V_i)$

Q的维度$n,d_k$,K的维度$m,d_k$,V的维度是$m,d_v$

$W^Q_i \in R^{\{d_k,d_k/h\}}$，$W^K_i \in R^{\{d_k,d_k/h\}}$，$W^V_i \in R^{\{d_v,d_v/h\}}$
$W^O_i$ \in R^{\{d_v,d_v\}}$
:::

### 3.3 Point-wise FeedForward Networks

单独作用在每个position的输入上。比如：输入是一个$n \times d_v$，作用在每个$ 1 \times d_v$。

::: center
![FFN.png](/images/FFN.png)

计算公式：
$FFN(\hat{x}) = Linear(ReLU(Linear(x)))$

第一个$Linear \in R{\{d_model,d_model * 4\}}$
第二个$Linear \in R{\{d_model * 4,d_model\}}$
:::

::: tip 常见面试问题
:::

::: details 问题1：FFN作用是什么？可以去掉吗？
作用：
- ==特征增强==：经过attention后的结果，每个位置上的内容都是对整个序列的aggregation，使用FFN来完成表征的增强
- ==引入非线性==：V的投影是线性的。attention计算是对V的线性加权和，也是线性的。需要依赖FFN来引入非线性，拟合更复杂的模式。
- ==知识存储==: [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
:::

::: details 问题2：FFN有什么改进工作？
激活函数：

Linear：

:::

### 计算复杂度


::: tip 常见面试题目
:::

::: details 问题1：《Attention is All Your Nedd》中的一些超参数设置
Encoder和Decoder是多少层？ ---- 均为==6层==
模型使用的维度是多少？ ---- ==$d_model$512维==
使用的是Pre-Norm还是Post-Norm? ---- ==Post-Norm==, $LayerNorm(x+SubLayer(x))$
模型使用的头数是多少？ ---- 8个头
:::