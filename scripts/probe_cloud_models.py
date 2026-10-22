#!/usr/bin/env python3
"""Make minimal authenticated calls to the cloud models used by InfoNav."""

from __future__ import annotations

import argparse
import os
import time

import requests
from openai import OpenAI


def probe_qwen(model: str) -> None:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is unset")
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    started = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with only: OK",
            }
        ],
        temperature=0,
        max_tokens=8,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("empty response")
    print(f"[probe] qwen model={model} status=available latency={time.monotonic() - started:.2f}s")


def probe_deepseek(model: str) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is unset")
    api_url = os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"
    )
    if not api_url.rstrip("/").endswith("chat/completions"):
        api_url = f"{api_url.rstrip('/')}/chat/completions"
    started = time.monotonic()
    response = requests.post(
        api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with only: OK"}],
            "temperature": 0,
            "max_tokens": 8,
            "stream": False,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        raise RuntimeError("empty response")
    print(f"[probe] deepseek model={model} status=available latency={time.monotonic() - started:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model", default="qwen3-vl-flash")
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    args = parser.parse_args()

    failures = 0
    for name, probe in (
        ("qwen", lambda: probe_qwen(args.qwen_model)),
        ("deepseek", lambda: probe_deepseek(args.deepseek_model)),
    ):
        try:
            probe()
        except Exception as exc:
            failures += 1
            print(f"[probe] {name} status=unavailable error={type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
