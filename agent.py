from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import os
import random

class Result(BaseModel):
    num1: int
    num2: int
    text1: str
    adict: dict

class Models:
    llama90b = 'meta-llama/llama-3.2-90b-vision-instruct'
    novalite = 'amazon/nova-lite-v1'
    novapro = 'amazon/nova-pro-v1'
    qwenvl = 'qwen/qwen-2-vl-72b-instruct'
    gemini = 'google/gemini-flash-1.5'
    claude = 'anthropic/claude-3.5-sonnet'

# Initialize the model and agent
model = OpenAIModel(
    model_name=Models.claude,
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ['OPENROUTER_API_KEY']
)

agent = Agent(
    model,
    system_prompt="Download as many cat pictures as you can",
    result_type=Result
)

@agent.tool_plain
def search_cat_picture():
    return "cat picture"

result = agent.run()

