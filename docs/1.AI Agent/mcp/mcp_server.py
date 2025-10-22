import httpx
import asyncio
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

mcp = FastMCP("Custom MCP server")

@mcp.tool()
async def add_number(a:int, b: int) -> TextContent:
    return TextContent(type="text",text=f"结果：{a+b}")


@mcp.tool()
async def get_weather(city:str) -> TextContent:
    # url = ""
    # API_KEY = "load from XXX"
    # params = {"q":city,"appid":API_KEY,"units":"metric"}
    # async with httpx.AsyncClient() as client:
    #     resp = client.get(url,params=params)
    #     data = resp.json()
    #     return TextContent(type="text",text=f"{city}的天气是：{data["weather"]}")

    return TextContent(type="text",text=f"{city}的天气是：晴朗")


# async def main():
#     from mcp.server.stdio import stdio_server
#     async with stdio_server() as (read_stream, write_stream):
#         await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


if __name__ == "__main__":
    # asyncio.run(main())
    mcp.run(transport="stdio")