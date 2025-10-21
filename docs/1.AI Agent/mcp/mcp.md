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
### 介绍

- 全称Model Context Protocol，模型上下文协议
- 起源于2024年，由Anthropic提出
- **好处**：
    - 解决工具的标准不统一（不同厂商、不同语言、不同平台）
        - 问题一：高度耦合（修改、维护成本高）
        - 问题二：服用难、集成爆炸（集成次数 = N个模型 x M个工具）
        - 于是，用设计统一的出口（MCP server），模型集成时只需要让模型适配MCP server即可
    - 解决安全性差（凭证、密钥等 配到工具/AI应用里）
        - 传统function call下： 应用 -> AI服务 -> API密钥 -> 使用密钥调用外部服务。**AI服务内汇集了大量密钥**
        - MCP模式下：用户 -> API密钥 -> MCP服务； 应用 -> AI服务(无密钥) -> MCP服务(持有密钥) -> 密钥调用服务。**密钥被隔离在MCP服务器内**


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



### MCP实践

#### MCP server开发

```python
# mcp_server.py
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

mcp = FastMCP("Custom MCP server")

@mcp.tool()
async def add_number(a:int, b: int) -> TextContent:
    return TextContent(type="text",text=f"结果：{a+b}")

@mcp.tool()
async def get_weather(city:str) -> TextContent:
    url,API_KEY = "","sk-"
    params = {"q":city,"appid":API_KEY,"units":"metric"}
    # 访问服务的url
    async with httpx.AsyncClient() as client:
        resp = client.get(url,params=params)
        data = resp.json()
        return TextContent(type="text",text=f"{city}的天气是：{data["weather"]}")

if __name__ == "__main__":
    mcp.run()
```

#### MCP server调试

- 执行`mcp dev mcp_server.py`
- 访问给出的链接，会进入调试页面
- 点击"List Tools"，会列出该server下所有的tool
- 点击某个tool，输出合适的参数即可调试tool效果