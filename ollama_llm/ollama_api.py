from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="your api key",  # needed but not used

)

response = client.chat.completions.create(
    model="deepseek-r1:7b",
    # model="llama3.1:8b",
    messages=[
        {"role": "user", "content": "天空为什么是蓝色的？"}
    ]
)

print(response.choices[0].message.content)
