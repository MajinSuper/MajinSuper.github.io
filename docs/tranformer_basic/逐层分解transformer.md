---
title: 逐层分解transformer
tags:
  - transformer
  - 基础
---

## 1.介绍

### 1.1 整体架构

::: center
![img.png](/images/transformer介绍.png)
:::

**transformer架构**如上如图所示，是由==Encoder==和==Decoder==两部分组成，在原论文中每个部分包含==6个block==。

### 1.2 基本流程
1. **分词**：对输入文本进行分词 `string -> [token_1,token_2 ... token_n]`
2. **词嵌入**：每个词使用向量表达 `[token_1,token_2 ... token_n] -> [emb_1,emb_2 ... emb_n]`
3. **位置编码**：每个词的词嵌入 + 其位置编码 `[emb_1,emb_2 ... emb_n] + [pe_1, pe_2 ... pe_n] -> [input_1, input_2... input_n]`
4. **Encoder**: 输入给Encoder进行计算 `[input_1, input_2 ... input_n] -> [enc_emb_1, enc_emb_2 ... enc_emb_n]`
5. **Decoder**: Encoder计算结果再传给Decoder `[enc_emb_1, enc_emb_2 ... enc_emb_n] -> out_1`
6. **Decoder**: Encoder+Decoder结果继续传给Decoder `[enc_emb_1, enc_emb_2 ... enc_emb_n,out_1] -> out_2`
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




