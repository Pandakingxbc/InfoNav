import os

# Temporarily disable proxy environment variables to avoid SOCKS proxy issues
old_proxies = {}
proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
for var in proxy_vars:
    if var in os.environ:
        old_proxies[var] = os.environ[var]
        del os.environ[var]

try:
    from ollama import chat
    from ollama import ChatResponse
finally:
    # Restore proxy environment variables
    for var, value in old_proxies.items():
        os.environ[var] = value

from llm.utils.get_sys_prompt import get_similar_answer_prompt

def ollama_respond(model, prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt,
    }
    history = system_prompts + [msg]
    
    response = chat(model=model, messages=history, stream=False),
    return response[0].message.content
    
