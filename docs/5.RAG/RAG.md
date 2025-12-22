---
title: RAG
tags:
  - RAG
createTime: 2025-12-21 10:44:31
permalink: /article/introduce_RAG/
---

### RAG基本流程

- ==离线==：切分、索引
- ==在线==：召回、重排、生成

#### （离线）切分
- 按句子分：太细太碎
- 按段落分：不均衡，可能会很长，可能会很多
- 按字数分：语义不连贯

#### （离线）索引
- 将每个chunk进行向量化表达，放入向量库中以备后续查询

::: tip 常见面试题
:::

::: deatials 1. 如何向量化？模型怎么选的？有没有自己训练过？
:::

::: detaials 2. 向量库有了解吗？向量库是怎么选择的？
:::

#### （在线）召回
纯向量的泛化性好，但对精确符号、专属名词不敏感；
文本召回能解决精准匹配、且召回可控，稳定性好；
两者互补

- 向量召回：
    - 用户query进行emb后，跟库中的每个索引向量（矩阵运算） 计算相似度；
    - 计算相似度的方法：余弦、欧式距离、点积
    - **核心**：“能解决用户==想表达什么==”
    - **优势**：==泛化性==，近似语义理解（谷歌/必应/搜索引擎）、跨语言理解（谷歌浏览器/chrome）、容错性（拼写错误、模糊表述）
    - **不足**：缩写词和短语（忍3、“楠”得一见）、搜索具体专有名词（歌名+歌名的remix）和人名可能会飘逸（马丁->马丁内斯）
- 文本&图谱召回（倒排）：
    - 提取领域实体 -> 实体文本多路召回：原词，n-gram，拼音，拼音首字母
    - 实体 -> 文本：实体属性，周围一跳的信息描述（模板）
    - 文本检索引擎：ElasticSearch

#### （在线）重排

1. 重排可以==融合多路召回的结果==

2. 在单路召回情况下，依然能==提升准确率==。相关≠可用（RAG需要的是能不能解决问题，而不是是否相关）

    具体来说，召回返回的分数是$P(doc \in 同一主题|query)$ ，重排返回的分数是$P(doc 解决 query|query)$

    再比如，原始query：`transformer为什么需要PE？`，

    召回1：`transformer是一种基于self-attention结构的模型，广泛应用于XXX`

    召回2：`由于self-attention对输入顺序不敏感，必须通过引入PE来注入位置信息`

    召回1的召回分数很高（主题匹配高）、但没法回答问题（无关问题）

    召回2的召回分数较低（主题匹配低）、但可以回答问题（命中问题）



::: center
 <img src="/images/RAG/Embedding_bi_encoder.png" style="zoom:30%" alt="Point-wise FeedForward Networks"  />
:::
- Embedding模型本质上是双编码器，文本内部没有任何交互，只有最后输出结果时，两个结果才会唯一一次交互。


::: center
 <img src="/images/RAG/rerank_cross_encode.png" style="zoom:30%" alt="Point-wise FeedForward Networks"  />
:::
- ReRank模型是Cross-encoder的模型，一开始就通过transformer进行交互。

|        | 召回                                     | 重排                                   |
| ------ | ---------------------------------------- | -------------------------------------- |
| 关注点 | 刻画相似度，”跟query是不是相关“          | 刻画能用性，”能不能解决query“          |
| 原理   | embedding模型+向量相似度                 | cross-encoder => 分数                  |
| 优点   | 成本低；快（预计算好一半，在线计算很少） | 准确                                   |
| 缺点   | 不准确                                   | 成本高；慢（相比召回，没有预计算内容） |
| 场景   | 在大规模后选中，初步筛选                 | 在部分候选上，进一步精挑细选           |



#### （在线）生成

将召回+重排后的片段，拼接prompt+原始问题，输入给大模型，输出最终答案。

---



::: tip 常见面试题
:::

::: detials 1. 为什么需要重排？直接用召回的相似度不行吗？
片段相似 ≠ 语义相关。
比如：transformer为什么需要PE？ 检索到：transformer是一种基于self-attention的模型结构，被广泛应用于NLP。
两者片段相似，但是语义是不相关的

:::

::: details 2.怎么提高召回准确率和覆盖率？
使用多路召回、过滤、rerank
:::

::: details 3.怎么改进embedding模型？怎么训练？
:::

::: details 4.怎么改进rerank模型？怎么训练？
:::


### 评估RAG系统

- 召回率（recall）：能不能从库中尽可能的把相关的召回来。“召回十个，里面有多少个是相关的 ” 除以 “库中有多少个相关的”
- 精确率（precision）：召回的东西，有多少是相关的。“召回十个，里面有多少个是相关的 ” 除以 “召回了多少个（十个）”
- 响应时延（Latency）：提问到生成答案的时间



### 参考：

- [Bilibili: RAG 工作机制详解——一个高质量知识库背后的技术全流程](https://www.bilibili.com/video/BV1JLN2z4EZQ/?spm_id_from=333.337.search-card.all.click&vd_source=748cb51f7cdac32f173ae1c569bfb80d)

- [Github: RAG系统：数据越多效果越好吗？](https://github.com/netease-youdao/QAnything/wiki/RAG%E7%B3%BB%E7%BB%9F%EF%BC%9A%E6%95%B0%E6%8D%AE%E8%B6%8A%E5%A4%9A%E6%95%88%E6%9E%9C%E8%B6%8A%E5%A5%BD%E5%90%97%EF%BC%9F)
- [知乎：embedding那些事](https://zhuanlan.zhihu.com/p/29949362142)
- [知乎：rerank那些事](https://zhuanlan.zhihu.com/p/29977179977)
- [飞书：混合检索和重排序改进RAG](https://xuqiwei1986.feishu.cn/wiki/Mmt0wpQo6iDHAyky5fzcZbvBnih)