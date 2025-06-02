from agents import Agent, Runner, RunConfig, OpenAIProvider

# 创建 AI 助手实例
# - name: 设置助手名称为 "Assistant"
# - instructions: 设置助手的基本指令
# - model: 使用 glm-4-flash 模型
agent = Agent(name="Assistant", instructions="You are a helpful assistant", model="deepseek-r1:7b")

# 配置运行环境
# - api_key:  API 密钥
# - base_url: API 接口地址
# - use_responses: 禁用 responses 库
run_config = RunConfig(model_provider = OpenAIProvider(
    api_key="your api key", #
    base_url="http://localhost:11434/v1",
    use_responses=False)
)

# 导入并应用 nest_asyncio 来解决 Jupyter 中的异步运行问题
import nest_asyncio
nest_asyncio.apply()

# 同步运行 AI 助手
# - 让助手创作一首关于编程中递归的俳句
# - 使用之前配置的 run_config
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.", run_config=run_config)
# 打印助手生成的俳句
print(result.final_output)
