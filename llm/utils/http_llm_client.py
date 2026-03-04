"""
HTTP-based LLM Client using OpenAI-compatible API

Connects to local LLM service at http://localhost:20006
Compatible with OpenAI chat completion format.

Author: Zager-Zhang
"""

import requests
import json
from typing import List, Dict, Optional


class HTTPLLMClient:
    """
    Client for OpenAI-compatible LLM API.

    Default endpoint: http://localhost:20006/v1/chat/completions
    """

    def __init__(self, base_url: str = "http://localhost:20006", timeout: int = 30, model: Optional[str] = None):
        """
        Initialize HTTP LLM client.

        Args:
            base_url: Base URL of LLM service (default: http://localhost:20006)
            timeout: Request timeout in seconds
            model: Model name (auto-detect if None)
        """
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        self.models_endpoint = f"{self.base_url}/v1/models"
        self.timeout = timeout
        self.model = model

        print(f"[HTTPLLMClient] Initialized with endpoint: {self.api_endpoint}")
        
        # Auto-detect model if not specified
        if self.model is None:
            self._auto_detect_model()
    
    def _auto_detect_model(self):
        """Auto-detect available model from service."""
        try:
            response = requests.get(self.models_endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                self.model = data["data"][0]["id"]
                print(f"[HTTPLLMClient] Auto-detected model: {self.model}")
            else:
                self.model = "gpt-3.5-turbo"  # Fallback
                print(f"[HTTPLLMClient] No models found, using fallback: {self.model}")
        except Exception as e:
            self.model = "gpt-3.5-turbo"  # Fallback
            print(f"[HTTPLLMClient] Model auto-detection failed: {e}, using fallback: {self.model}")

    def chat(self, messages: List[Dict[str, str]],
             model: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: Optional[int] = None) -> str:
        """
        Send chat completion request to LLM service.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (uses auto-detected model if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response text from LLM
        """
        # Use auto-detected model if not specified
        if model is None:
            model = self.model
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            # Send POST request
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )

            # Check response status
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Extract message content (OpenAI format)
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                return content
            else:
                print(f"[HTTPLLMClient] Unexpected response format: {data}")
                return ""

        except requests.exceptions.RequestException as e:
            print(f"[HTTPLLMClient] Request error: {e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"[HTTPLLMClient] JSON decode error: {e}")
            return ""
        except Exception as e:
            print(f"[HTTPLLMClient] Unexpected error: {e}")
            return ""

    def get_answer(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simplified interface: get answer for a single prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Response text from LLM
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages)


if __name__ == "__main__":
    # Test the HTTP LLM client
    print("Testing HTTP LLM Client...")

    client = HTTPLLMClient()

    # Test simple query
    print("\n=== Test 1: Simple query ===")
    response = client.get_answer("Hello! Can you hear me?")
    print(f"Response: {response}")

    # Test chat with system prompt
    print("\n=== Test 2: Chat with system prompt ===")
    messages = [
        {"role": "system", "content": "You are a helpful navigation assistant."},
        {"role": "user", "content": "Where would I find a dining table?"}
    ]
    response = client.chat(messages)
    print(f"Response: {response}")

    # Test semantic expansion
    print("\n=== Test 3: Semantic expansion ===")
    from semantic_prompt_expansion import build_semantic_expansion_messages

    messages = build_semantic_expansion_messages("refrigerator")
    response = client.chat(messages, temperature=0.7)
    print(f"Response: {response}")

    # Try to parse
    from semantic_prompt_expansion import parse_semantic_expansion_response
    parsed = parse_semantic_expansion_response(response)
    print(f"\nParsed {len(parsed)} hypotheses:")
    for prompt, weight in parsed:
        print(f"  - [{weight:.2f}] {prompt}")
