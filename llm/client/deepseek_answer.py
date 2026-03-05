from llm.utils.get_sys_prompt import get_similar_answer_prompt
from openai import OpenAI
import httpx

# 创建不使用代理的 HTTP 客户端
http_client = httpx.Client(proxy=None)
import os
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
    http_client=http_client
)

def deepseek_respond(prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt
    }
    history = system_prompts + [msg]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    deepseek_respond('dining table')