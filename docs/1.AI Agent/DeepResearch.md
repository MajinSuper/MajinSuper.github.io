---
title: Deep Research
tags:
  - AI Agent
  - LLM
  - Deep Research
---
content is coming

## 基本架构

## Planner

## Answer





step1: 记忆如何设计的？

最直接的两种方法：

- N个任务结束后，做一次摘要。任务多，context会超长；且信息丢失率高。
- 每个任务结束后，都做一次摘要。调用api次数增加，耗时会增多；信息丢失



Q：为什么要用结构化输出？
A：
- 便于提取结果：从自然语言中提取 -> 字段解析
- 便于校验、提升鲁棒性： 字段值缺失、不合法 -> 重跑

Q: 怎么保证模型输出结构化内容？
A：
- prompt中要求最终结果使用json表达，给出json的schema，明确说明每个字段的内容
- 当出现

Q：幻觉是如何消除的？
A：使用分章节编写，而不是通篇写完。

Q: 为什么要用MCP，解决了什么问题？MCP是怎么用的？
A: 

Q: Memory是怎么管理的？
A：

Q: 多模态是怎么做的？
A：
