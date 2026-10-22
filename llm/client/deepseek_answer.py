from llm.utils.get_sys_prompt import get_similar_answer_prompt
from openai import OpenAI
import os

# Temporarily disable proxy environment variables to avoid SOCKS proxy issues
old_proxies = {}
proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
for var in proxy_vars:
    if var in os.environ:
        old_proxies[var] = os.environ[var]
        del os.environ[var]

try:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com"
    )
finally:
    # Restore proxy environment variables
    for var, value in old_proxies.items():
        os.environ[var] = value

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