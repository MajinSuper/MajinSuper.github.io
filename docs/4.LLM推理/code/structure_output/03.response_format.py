import json
from openai import OpenAI
from pydantic import BaseModel
# from pydantic import dataclass
from dataclasses import dataclass

class QuestionAnswer(BaseModel):
    question: str
    answer: str

client = OpenAI(
    api_key="nokey",
    base_url="http://localhost:11434/v1",
)

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

messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.beta.chat.completions.parse(
    model="deepseek-r1:1.5b",
    messages=messages,
    response_format=QuestionAnswer,
)


print("raw response:", response.choices[0].message.content)
print("parsed response:", response.choices[0].message.parsed)

# raw response: {
# "question": "Which is the longest river in the world?",
# "answer": "The Nile River."
# }
# parsed response:  question='Which is the longest river in the world?' answer='The Nile River.'