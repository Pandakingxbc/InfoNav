#!/usr/bin/env python3
"""
测试 VLM 和 LLM API 服务是否正常工作
"""

import os
import sys
import time

def test_dashscope_vlm():
    """测试 DashScope VLM API"""
    print("\n" + "="*60)
    print("测试 DashScope VLM API")
    print("="*60)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] DASHSCOPE_API_KEY 环境变量未设置")
        print("请运行: export DASHSCOPE_API_KEY='your-api-key'")
        return False

    print(f"[OK] DASHSCOPE_API_KEY 已设置 (长度: {len(api_key)})")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        print("[INFO] 发送测试请求到 DashScope VLM...")
        t0 = time.time()

        # 简单的文本测试（不需要图片）
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请回复 'VLM服务正常' 这四个字"}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=50
        )

        t1 = time.time()
        response = completion.choices[0].message.content

        print(f"[OK] VLM 响应成功!")
        print(f"  耗时: {t1-t0:.2f}s")
        print(f"  响应: {response[:100]}")
        return True

    except Exception as e:
        print(f"[ERROR] VLM 测试失败: {e}")
        return False


def test_deepseek_llm():
    """测试 DeepSeek LLM API"""
    print("\n" + "="*60)
    print("测试 DeepSeek LLM API")
    print("="*60)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY 环境变量未设置")
        print("请运行: export DEEPSEEK_API_KEY='your-api-key'")
        return False

    print(f"[OK] DEEPSEEK_API_KEY 已设置 (长度: {len(api_key)})")

    try:
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": "请回复 'LLM服务正常' 这四个字"}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }

        print("[INFO] 发送测试请求到 DeepSeek LLM...")
        t0 = time.time()

        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        t1 = time.time()

        if resp.status_code != 200:
            print(f"[ERROR] HTTP {resp.status_code}: {resp.text}")
            return False

        result = resp.json()
        response = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        print(f"[OK] LLM 响应成功!")
        print(f"  耗时: {t1-t0:.2f}s")
        print(f"  响应: {response[:100]}")
        return True

    except Exception as e:
        print(f"[ERROR] LLM 测试失败: {e}")
        return False


def test_ollama_vlm(api_url="http://localhost:20004/api/chat"):
    """测试 Ollama VLM API (可选)"""
    print("\n" + "="*60)
    print(f"测试 Ollama VLM API @ {api_url}")
    print("="*60)

    try:
        import requests

        payload = {
            "model": "qwen3-vl:32b",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": False
        }

        print("[INFO] 发送测试请求到 Ollama...")
        t0 = time.time()

        resp = requests.post(api_url, json=payload, timeout=10)

        t1 = time.time()

        if resp.status_code != 200:
            print(f"[ERROR] HTTP {resp.status_code}")
            return False

        print(f"[OK] Ollama 响应成功! 耗时: {t1-t0:.2f}s")
        return True

    except requests.exceptions.ConnectionError:
        print("[WARNING] Ollama 服务未运行 (这是可选的)")
        return False
    except Exception as e:
        print(f"[WARNING] Ollama 测试失败: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# API 服务连通性测试")
    print("#"*60)

    results = {}

    # 测试 DashScope VLM
    results['dashscope_vlm'] = test_dashscope_vlm()

    # 测试 DeepSeek LLM
    results['deepseek_llm'] = test_deepseek_llm()

    # 测试 Ollama (可选)
    results['ollama_vlm'] = test_ollama_vlm()

    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)

    for service, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {service}: {status}")

    # 检查必需的服务
    required_pass = results['dashscope_vlm'] and results['deepseek_llm']

    if required_pass:
        print("\n[SUCCESS] 必需的 API 服务 (DashScope VLM + DeepSeek LLM) 均正常!")
        print("可以使用 --realtime-hypothesis 模式运行 habitat_evaluation.py")
    else:
        print("\n[FAILED] 部分必需服务不可用，请检查 API Key 配置:")
        if not results['dashscope_vlm']:
            print("  - 设置 DASHSCOPE_API_KEY 环境变量")
        if not results['deepseek_llm']:
            print("  - 设置 DEEPSEEK_API_KEY 环境变量")

    return 0 if required_pass else 1


if __name__ == "__main__":
    sys.exit(main())
