
# 定义function tools库
import yfinance as yf
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period='1d')['close'].iloc[-1]
    except Exception as e:
        return 0.0
availabl_tools = {'get_stock_price':get_stock_price}

# 调用llm时，传递可以使用的工具
import ollama
respone = ollama.chat(
    'llama3.1:8b',#'deepseek-r1:7b',
    messages = [{'role':'user','content':'what is the stock price of Apple?'}],
    tools = [get_stock_price]
)

print(respone)
# model='llama3.1:8b' created_at='2025-05-11T06:31:45.8852474Z' done=True done_reason='stop' total_duration=273894600 
# load_duration=12327900 prompt_eval_count=158 prompt_eval_duration=2614700 eval_count=19 eval_duration=258952000 
# message=Message(role='assistant', content='', images=None, 
# tool_calls=[ToolCall(function=Function(name='get_stock_price', arguments={'ticker': 'Apple'}))])


# 对返回的tool_calls进行处理
for tool_call in respone.message.tool_calls:
    if tool_call.function.name in availabl_tools.keys():
        tool_function = availabl_tools[tool_call.function.name]
        stock_price = tool_function(tool_call.function.arguments)
        print(f"The stock price of {tool_call.function.arguments['ticker']} is {stock_price}")

