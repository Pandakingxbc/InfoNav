"""
HTTP-based VLM Client using OpenAI-compatible API

Connects to local VLM service at http://localhost:20004
Supports vision-language tasks with OpenAI format.

Author: Zager-Zhang
"""

import requests
import json
import base64
import numpy as np
import cv2
from typing import Optional, Union
from io import BytesIO
from PIL import Image


class HTTPVLMClient:
    """
    Client for OpenAI-compatible VLM (Vision-Language Model) API.

    Default endpoint: http://localhost:20004/v1/chat/completions
    Supports image-text matching and visual question answering.
    """

    def __init__(self, base_url: str = "http://localhost:20004", timeout: int = 30, model: Optional[str] = None):
        """
        Initialize HTTP VLM client.

        Args:
            base_url: Base URL of VLM service (default: http://localhost:20004)
            timeout: Request timeout in seconds
            model: Model name (auto-detect if None)
        """
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        self.models_endpoint = f"{self.base_url}/v1/models"
        self.timeout = timeout
        self.model = model

        print(f"[HTTPVLMClient] Initialized with endpoint: {self.api_endpoint}")
        
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
                # Prefer vision models (llama3.2-vision, gpt-4-vision, etc.)
                models = [m["id"] for m in data["data"]]
                vision_models = [m for m in models if "vision" in m.lower()]
                self.model = vision_models[0] if vision_models else models[0]
                print(f"[HTTPVLMClient] Auto-detected model: {self.model}")
            else:
                self.model = "gpt-4-vision-preview"  # Fallback
                print(f"[HTTPVLMClient] No models found, using fallback: {self.model}")
        except Exception as e:
            self.model = "gpt-4-vision-preview"  # Fallback
            print(f"[HTTPVLMClient] Model auto-detection failed: {e}, using fallback: {self.model}")

    def encode_image(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        Encode image to base64 string for API transmission.

        Args:
            image: Image as numpy array (H, W, 3) or PIL Image

        Returns:
            Base64 encoded image string
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Encode to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str

    def query_vlm(self, image: Union[np.ndarray, Image.Image],
                  text_prompt: str,
                  model: Optional[str] = None,
                  max_tokens: int = 300) -> str:
        """
        Query VLM with image and text prompt.

        Args:
            image: Input image (numpy array or PIL Image)
            text_prompt: Text prompt/question
            model: Model name (uses auto-detected model if None)
            max_tokens: Maximum tokens in response

        Returns:
            Response text from VLM
        """
        # Use auto-detected model if not specified
        if model is None:
            model = self.model
        
        # Encode image
        image_base64 = self.encode_image(image)

        # Build message in OpenAI vision format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

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

            # Extract message content
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                return content
            else:
                print(f"[HTTPVLMClient] Unexpected response format: {data}")
                return ""

        except requests.exceptions.RequestException as e:
            print(f"[HTTPVLMClient] Request error: {e}")
            return ""
        except Exception as e:
            print(f"[HTTPVLMClient] Unexpected error: {e}")
            return ""

    def get_itm_score(self, image: Union[np.ndarray, Image.Image],
                      text_prompt: str) -> float:
        """
        Get image-text matching score.

        This simulates ITM scoring by asking VLM to rate the match.

        Args:
            image: Input image
            text_prompt: Text to match (e.g., "Is there a kitchen ahead?")

        Returns:
            ITM score in [0, 1] (0 = no match, 1 = strong match)
        """
        # Construct prompt for ITM scoring
        scoring_prompt = f"""You are evaluating image-text matching.

Question: {text_prompt}

Please rate how well the image matches this question on a scale from 0.0 to 1.0:
- 0.0: The image does not match the question at all
- 0.5: The image partially matches the question
- 1.0: The image strongly matches the question

Respond with ONLY a number between 0.0 and 1.0, nothing else.
"""

        response = self.query_vlm(image, scoring_prompt, max_tokens=10)

        # Parse numerical score
        try:
            score = float(response.strip())
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            return score
        except ValueError:
            print(f"[HTTPVLMClient] Failed to parse score from: {response}")
            # Try to extract first number
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    pass
            return 0.5  # Default neutral score

    def cosine_similarity(self, image: Union[np.ndarray, Image.Image],
                          text_prompt: str) -> float:
        """
        Compute cosine similarity between image and text (simulated via ITM).

        Args:
            image: Input image
            text_prompt: Text prompt

        Returns:
            Similarity score in [0, 1]
        """
        # For HTTP API, we use ITM score as similarity
        return self.get_itm_score(image, text_prompt)


# Backward compatibility alias
class HTTPITMClient(HTTPVLMClient):
    """Alias for HTTPVLMClient to maintain compatibility with existing code."""

    def __init__(self, base_url: str = "http://localhost:20004", timeout: int = 30):
        super().__init__(base_url, timeout)

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """Compatibility method matching BLIP2ITMClient interface."""
        return self.cosine_similarity(image, txt)

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        """Compatibility method matching BLIP2ITMClient interface."""
        return self.get_itm_score(image, txt)


if __name__ == "__main__":
    # Test the HTTP VLM client
    print("Testing HTTP VLM Client...")

    client = HTTPVLMClient()

    # Create a test image
    print("\n=== Creating test image ===")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some simple shapes
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(test_image, (400, 200), 50, (255, 0, 0), -1)
    cv2.putText(test_image, "TEST", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Test VQA
    print("\n=== Test 1: Visual Question Answering ===")
    response = client.query_vlm(test_image, "What shapes and colors do you see in this image?")
    print(f"Response: {response}")

    # Test ITM scoring
    print("\n=== Test 2: Image-Text Matching Score ===")
    prompts = [
        "Is there a green rectangle in the image?",
        "Is there a blue circle in the image?",
        "Is there text in the image?",
        "Is this a photograph of nature?",
    ]

    for prompt in prompts:
        score = client.get_itm_score(test_image, prompt)
        print(f"ITM Score for '{prompt}': {score:.3f}")

    # Test compatibility interface
    print("\n=== Test 3: Compatibility Interface ===")
    itm_client = HTTPITMClient()
    cosine = itm_client.cosine(test_image, "Is there a green rectangle ahead?")
    itm = itm_client.itm_score(test_image, "Is there a green rectangle ahead?")
    print(f"Cosine similarity: {cosine:.3f}")
    print(f"ITM score: {itm:.3f}")
