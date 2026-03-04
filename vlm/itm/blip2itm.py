from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
import cv2
from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")


class BLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    def __init__(
        self,
        name: str = "blip2_image_text_matching",
        model_type: str = "pretrain",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.device = device

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            cosine = self.model(
                {"image": img, "text_input": txt}, match_head="itc"
            ).item()

        # Explicitly delete tensor and clear GPU cache to prevent memory buildup
        del img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cosine

    def itm_scores(self, image: np.ndarray, txt: str) -> np.ndarray:
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

        itm_score = itm_scores[:, 1].item()

        # Explicitly delete tensors and clear GPU cache to prevent memory buildup
        del img, itm_output, itm_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return itm_score


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])

    def cosine_batch(self, image: np.ndarray, txt_list: list) -> list:
        """
        Batch inference for multiple prompts with the same image.
        More efficient than calling cosine() multiple times.

        Args:
            image: RGB image array
            txt_list: List of text prompts

        Returns:
            List of cosine similarity scores
        """
        response = send_request(self.url, image=image, txt_list=txt_list)
        return response["response_list"]

    def ig_score_weighted(self, image: np.ndarray, weights: list = None) -> dict:
        """
        计算 IG Score（在 server 端加权相加）。

        IG prompts 固定为三个连通性描述：
        1. "This view shows a corridor or hallway leading to other areas"
        2. "There is a doorway or opening that leads to another room"
        3. "This passage connects to multiple rooms or spaces"

        Args:
            image: RGB image array
            weights: 三个 prompt 的权重列表，默认 [1/3, 1/3, 1/3] (等权平均)

        Returns:
            dict: {
                "ig_score": float,       # 加权后的 IG 分数
                "corridor_score": float, # corridor prompt 的原始分数
                "doorway_score": float,  # doorway prompt 的原始分数
                "passage_score": float,  # passage prompt 的原始分数
            }
        """
        if weights is None:
            weights = [1.0/3, 1.0/3, 1.0/3]
        response = send_request(self.url, image=image, ig_weights=weights)
        return response["ig_data"]

    def itm_score(self, image: np.ndarray, txt: str) -> np.ndarray:
        print(f"Question of blip2 is:{txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["itm score"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    # IG connectivity prompts (固定，与 main.tex Section III-D 一致)
    IG_CONNECTIVITY_PROMPTS = [
        "This view shows a corridor or hallway leading to other areas",
        "There is a doorway or opening that leads to another room",
        "This passage connects to multiple rooms or spaces"
    ]

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            import time
            t_total_start = time.time()

            # Decode image
            t0 = time.time()
            image = str_to_image(payload["image"])
            t_image_decode = time.time() - t0

            # IG Score 加权计算（server 端直接完成）
            if "ig_weights" in payload:
                weights = payload["ig_weights"]
                if len(weights) != 3:
                    weights = [1.0/3, 1.0/3, 1.0/3]

                t0 = time.time()
                ig_scores = []
                for prompt in IG_CONNECTIVITY_PROMPTS:
                    ig_scores.append(self.cosine(image, prompt))
                t_inference = time.time() - t0

                # 加权计算 IG score
                ig_score = sum(w * s for w, s in zip(weights, ig_scores))

                t_total = time.time() - t_total_start
                print(f"[BLIP2 Server] IG Score request | "
                      f"Image decode: {t_image_decode:.4f}s | "
                      f"Inference: {t_inference:.3f}s | "
                      f"IG: {ig_score:.3f} (weights: {weights}) | "
                      f"Total: {t_total:.3f}s")

                return {
                    "ig_data": {
                        "ig_score": ig_score,
                        "corridor_score": ig_scores[0],
                        "doorway_score": ig_scores[1],
                        "passage_score": ig_scores[2],
                    }
                }

            # Support batch inference for multiple prompts
            elif "txt_list" in payload:
                txt_list = payload["txt_list"]
                num_prompts = len(txt_list)

                t0 = time.time()
                cosine_list = []
                for txt in txt_list:
                    cosine_list.append(self.cosine(image, txt))
                t_inference = time.time() - t0

                t_total = time.time() - t_total_start
                print(f"[BLIP2 Server] Batch request: {num_prompts} prompts | "
                      f"Image decode: {t_image_decode:.4f}s | "
                      f"Inference: {t_inference:.3f}s ({t_inference/num_prompts:.3f}s/prompt) | "
                      f"Total: {t_total:.3f}s")

                return {
                    "response_list": cosine_list,
                }
            else:
                # Original single-text inference for backward compatibility
                t0 = time.time()
                cosine_result = self.cosine(image, payload["txt"])
                itm_result = self.itm_scores(image, payload["txt"])
                t_inference = time.time() - t0

                t_total = time.time() - t_total_start
                print(f"[BLIP2 Server] Single request | "
                      f"Image decode: {t_image_decode:.4f}s | "
                      f"Inference: {t_inference:.3f}s | "
                      f"Total: {t_total:.3f}s")

                return {
                    "response": cosine_result,
                    "itm score": itm_result,
                }

    blip = BLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
