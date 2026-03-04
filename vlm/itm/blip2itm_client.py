"""
BLIP2 ITM Client - HTTP client for BLIP2 Image-Text Matching service.

This is a lightweight client that doesn't require torch/lavis/flask dependencies.
Only requires: numpy, cv2, requests (which are part of ROS Python environment).
"""

import base64
import numpy as np
import cv2
import requests
import time


def _image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    """Convert numpy image to base64 string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def _send_request(url: str, **kwargs) -> dict:
    """Send HTTP POST request with image and text."""
    payload = {}

    # Encode image if provided
    if "image" in kwargs:
        payload["image"] = _image_to_str(kwargs["image"])

    # Add other parameters
    for key, value in kwargs.items():
        if key != "image":
            payload[key] = value

    # Send request
    for attempt in range(10):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 9:
                raise Exception(f"Failed to send request after 10 attempts: {e}")
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + 10 * (attempt / 9))


class BLIP2ITMClient:
    """Client for BLIP2 ITM HTTP service."""

    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute cosine similarity between image and text.

        Args:
            image: Input image as numpy array
            txt: Text prompt to compare

        Returns:
            Cosine similarity score
        """
        response = _send_request(self.url, image=image, txt=txt)
        return float(response["response"])

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        """
        Compute ITM (Image-Text Matching) score.

        Args:
            image: Input image as numpy array
            txt: Text prompt to match

        Returns:
            ITM score (probability that image matches text)
        """
        print(f"Question of blip2 is: {txt}")
        response = _send_request(self.url, image=image, txt=txt)
        return float(response["itm score"])
