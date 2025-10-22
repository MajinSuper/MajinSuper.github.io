---
title: MCP
tags:
  - MCP
  - LLM
  - agent
createTime: 2025-10-21 23:44:31
permalink: /article/introduce_mcp/
---

## MCP
### 一、介绍

- 全称Model Context Protocol，模型上下文协议
- 起源于2024年，由Anthropic提出。将==AI应用==链接到==外部数据/工具/服务==
- **目的**：
    - 解决工具的标准不统一（不同厂商、不同语言、不同平台）,==开发、维护成本高==
        - 开发成本：==服用难、集成爆炸==（集成次数 = N个模型 x M个工具）
        - 维护成本：==高度耦合==（修改、维护成本高）
        - 于是，用设计统一的出口（MCP server），模型集成时只需要让模型适配MCP server即可
    - 解决==安全性差==（凭证、密钥等 配到工具/AI应用里）
        - 传统function call下：AI应用 -> API密钥 -> 使用密钥调用外部服务。**AI服务内汇集了大量密钥**
        - MCP模式下：用户 -> API密钥 -> MCP服务； AI应用(无密钥) -> MCP服务(持有密钥) -> 调用外部服务。**密钥被隔离在MCP服务器内**


::: tip 常见面试题
:::

::: details 问题1. MCP与Function Call有什么区别？分别在什么情况下用MCP和Funciton Call？
回答：


|  | MCP | Function Call |
|------|------|-----|
| 耦合度 | 低耦合 | 高耦合|
| 复用性 | 好 | 差 |
| 维护成本 | 低 | 高 |
| 面向任务 | 工具生态 | 确定的工具调用 |
| 使用场景 | 工具多，一次编写到处运行；<br>安全要求高 | 工具少，需要确定的工具；<br>无专门的安全要求 | 
:::

### 二、MCP系统架构
![MCP系统架构](/images/mcp/mcp_structure.png)

#### 2.1 三个概念

- MCP Host：实际的AI应用、AI Agent
- MCP Client：AI Agent==内部==的一个功能模块，负责连接管理、错误处理。与MCP Server建立一对一的链接。
- MCP Server：==外部==服务程序，提供外部数据、工具、服务的访问

#### 2.2 工作流程
<Badge type="tip" text="STEP 1:" /> user任务，Host借助MCP Client，要求MCP server列出所有的工具；MCP server返回所有的可用工具

![MCP](/images/mcp/MCP_workflow_part1.png)

<Badge type="tip" text="STEP 2:" /> Host请求LLM，判断使用哪个工具；借助MCP Client，要求调用MCP server的对应工具；MCP server使用工具，返回结果
![MCP](/images/mcp/MCP_workflow_part2.png)

<Badge type="tip" text="STEP 3:" />Host拿MCP server的结果，请求LLM；返回最终的结果，给用户
![MCP](/images/mcp/MCP_workflow_part3.png)

#### 2.3 数据传输
1. Stdio：标准输入输出，借助OS/kernel的stdin、stdout、stderror
2. HTTP over SSE（已被替代）：借助网络远程连接，持久连接
3. Streamable HTTP：text/event-stream

### 三、MCP实践

#### 3.1 MCP server开发

```python
# mcp_server.py
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

mcp = FastMCP("Custom MCP server") # [!code highlight]

@mcp.tool() # [!code highlight]
async def add_number(a:int, b: int) -> TextContent:
    return TextContent(type="text",text=f"结果：{a+b}") # [!code highlight]

@mcp.tool() # [!code highlight]
async def get_weather(city:str) -> TextContent:
    url, API_KEY = "", "sk-"
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    # 访问服务的url
    async with httpx.AsyncClient() as client:
        resp = client.get(url, params=params)
        data = resp.json()
        return TextContent(type="text", text=f"{city}的天气是：{data["weather"]}") # [!code highlight]

if __name__ == "__main__":
    mcp.run(transport="stdio") # [!code highlight]
```

#### 3.2 MCP server调试

- 执行`mcp dev mcp_server.py`
- 访问给出的链接，会进入调试页面
- 点击"List Tools"，会列出该server下所有的tool
- 点击某个tool，输出合适的参数即可调试tool效果
![dev mcp](/images/mcp/dev_mcp.png)

#### 3.3 MCP server调试