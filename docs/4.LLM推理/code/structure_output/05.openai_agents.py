import asyncio
from typing import Any


import json
from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import Agent, ModelSettings,Runner, set_default_openai_api, set_default_openai_client


class QuestionAnswer(BaseModel):
    question: str
    answer: str

client = AsyncOpenAI(
    api_key="nokey",
    base_url="http://localhost:11434/v1",
)

set_default_openai_client(client=client,use_for_tracing=False)
set_default_openai_api("chat_completions")

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

agent = Agent (
    name="QuestionAnswerAgent",
    #description="A agent that can answer questions",
    instructions=system_prompt,
    model="deepseek-r1:1.5b",
    model_settings=ModelSettings(
        temperature=0.1,
    ),
    output_type=QuestionAnswer,
)

async def main():
    res = await Runner.run(agent,user_prompt)
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
# RunResult:
# - Last agent: Agent(name="QuestionAnswerAgent", ...)
# - Final output (QuestionAnswer):
#     {
#       "question": "Which is the longest river in the world?",
#       "answer": "The Nile River"
#     }
# - 1 new item(s)
# - 1 raw response(s)
# - 0 input guardrail result(s)
# - 0 output guardrail result(s)
# (See `RunResult` for more details)