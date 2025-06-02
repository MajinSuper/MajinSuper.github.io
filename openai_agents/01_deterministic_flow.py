import asyncio
from turtledemo.sorting_animate import instructions1

from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, ModelSettings, Model, set_default_openai_client
from openai import AsyncOpenAI, models
from pydantic import BaseModel
from openai.types.graders.score_model_grader_param import Input

from openai_agents.utils.my_file_util import save_story_to_file
from utils.instructions_manager import instructions_dict

config = {
    "model_name": "qwq:latest",
    # "model_name": "deepseek-r1:7b",
    "api_url": "http://localhost:11434/v1",
    "api_key": "no key",
    "temperature": 0.5,  # 默认温度
    "output_dir": "deterministic_output",  # 输出目录
    "time_out": 1200,  # 超时时间
}

#
external_client = AsyncOpenAI(
    api_key=config["api_key"],
    base_url=config["api_url"],
    timeout=config["time_out"],
)
set_default_openai_client(external_client, use_for_tracing=False)

outline_agent = Agent(
    name="outline writer agent",
    instructions=instructions_dict.get("outline_instruction"),
    model=OpenAIChatCompletionsModel(  # 使用OpenAI兼容的聊天完成模型
        model=config["model_name"],  # 模型名称
        openai_client=external_client,  # 使用之前配置的Ollama客户端
    ),
    model_settings=ModelSettings(temperature=config.get("temperature")),
    output_type=str
)

class OutlineCheckResult(BaseModel):
    good_quality: bool  # 质量好坏
    is_scifi: bool  # 是否科幻题材

outline_check_agent = Agent(
    name="outline checker agent",
    instructions=instructions_dict.get("outline_checker"),
    model=OpenAIChatCompletionsModel(  # 使用OpenAI兼容的聊天完成模型
        model=config["model_name"],  # 模型名称
        openai_client=external_client,  # 使用之前配置的Ollama客户端
    ),
    model_settings=ModelSettings(temperature=config.get("temperature")),
    output_type=OutlineCheckResult
)

detail_writer = Agent(
    name="outline checker agent",
    instructions=instructions_dict.get("detail_writer"),
    model=OpenAIChatCompletionsModel(  # 使用OpenAI兼容的聊天完成模型
        model=config["model_name"],  # 模型名称
        openai_client=external_client,  # 使用之前配置的Ollama客户端
    ),
    model_settings=ModelSettings(temperature=config.get("temperature")),
    output_type=str
)

async def main():
    user_input= input("请输入想要创作的类型:\n")
    while not user_input.strip() :
        user_input = input("输入为空，请重新输入\n")

    print(f"开始生成大纲...\n")
    outline = await Runner.run(outline_agent, user_input)
    print(f"已生成大纲:\n\n{outline}")

    print(f"开始检查大纲...")
    check_result = await Runner.run(outline_check_agent, outline.final_output)
    print(f"大纲检查结果为{check_result}")

    if not check_result.final_output.is_scifi:
        print("不是科幻题材.退出")
        return
    elif not check_result.final_output.good_quality:
        print("质量不高，推出")
        return

    print("开始生成故事细节...")
    detail =await Runner.run(detail_writer, outline.final_output)
    print("完成故事细节")

    result_path = save_story_to_file(detail.final_output,user_input,config["output_dir"])
    print(f"保存到{result_path}")



if __name__ == '__main__':
    asyncio.run(main())
