"""
D-FINE Object Detector Module for ApexNav.
Replaces YOLOv7 with the state-of-the-art D-FINE detector.

D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
https://github.com/Peterande/D-FINE
"""

import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .detections import ObjectDetections
from ..coco_classes import COCO_CLASSES
from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

# Add D-FINE to path
DFINE_PATH = os.path.join(os.path.dirname(__file__), "D-FINE")

# Global variable to store YAMLConfig
YAMLConfig = None

def _import_dfine():
    """Import D-FINE when needed"""
    global YAMLConfig
    if YAMLConfig is None:
        sys.path.insert(0, DFINE_PATH)
        try:
            from src.core import YAMLConfig as _YAMLConfig
            YAMLConfig = _YAMLConfig
        except ImportError as e:
            raise ImportError(f"Could not import D-FINE: {e}")
    return YAMLConfig


class DFine(nn.Module):
    """
    D-FINE object detector wrapper compatible with ApexNav's detection interface.

    D-FINE is a powerful real-time object detector based on DETR architecture
    with fine-grained distribution refinement for improved accuracy.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        image_size: int = 640,
    ):
        """
        Initialize D-FINE detector.

        Args:
            config_path: Path to D-FINE YAML configuration file
            checkpoint_path: Path to D-FINE checkpoint file
            device: Device to run inference on ('cuda' or 'cpu')
            image_size: Input image size for the model
        """
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self._load_model(config_path, checkpoint_path)
        self.model.eval()

        # Warmup
        if self.device.type == "cuda":
            dummy_img = torch.rand(1, 3, image_size, image_size).to(self.device)
            orig_size = torch.tensor([[image_size, image_size]]).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_img, orig_size)
            print("D-FINE warmup complete!")

    def _load_model(self, config_path: str, checkpoint_path: str):
        """Load and configure the D-FINE model."""
        # Import D-FINE modules
        _YAMLConfig = _import_dfine()
        sys.path.insert(0, DFINE_PATH)
        try:
            cfg = _YAMLConfig(config_path)

            # Handle special cases in config
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

            # Load checkpoint
            print(f"Loading D-FINE checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Get state dict from checkpoint
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint.get("model", checkpoint)

            # Build and convert model to deploy mode
            cfg.model.load_state_dict(state)

            # Create deployment model
            class DeployModel(nn.Module):
                def __init__(self, model, postprocessor):
                    super().__init__()
                    self.model = model.deploy()
                    self.postprocessor = postprocessor.deploy()

                def forward(self, images, orig_target_sizes):
                    outputs = self.model(images)
                    outputs = self.postprocessor(outputs, orig_target_sizes)
                    return outputs

            self.model = DeployModel(cfg.model, cfg.postprocessor).to(self.device)
            print("D-FINE model loaded successfully!")

        finally:
            if DFINE_PATH in sys.path:
                sys.path.remove(DFINE_PATH)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.4,
        classes: Optional[List[str]] = None,
    ) -> ObjectDetections:
        """
        Perform object detection on the input image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            conf_thres: Confidence threshold for filtering detections
            classes: Optional list of classes to filter by

        Returns:
            ObjectDetections object with detection results
        """
        print("D-FINE is detecting...")

        # Get original image size
        h, w = image.shape[:2]

        # Convert numpy array to PIL Image
        im_pil = Image.fromarray(image).convert("RGB")
        orig_size = torch.tensor([[w, h]]).to(self.device)

        # Preprocess
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])
        img_tensor = transform(im_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            labels, boxes, scores = self.model(img_tensor, orig_size)

        # Move to CPU and convert to numpy
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        # Filter by confidence threshold
        mask = scores > conf_thres
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        # Convert to normalized coordinates [x1, y1, x2, y2]
        normalized_boxes = boxes.copy()
        normalized_boxes[:, [0, 2]] /= w
        normalized_boxes[:, [1, 3]] /= h

        # Convert labels to class names
        phrases = [COCO_CLASSES[int(idx)] for idx in labels]

        # Filter by class if specified
        if classes is not None:
            keep_mask = np.array([p in classes for p in phrases])
            normalized_boxes = normalized_boxes[keep_mask]
            scores = scores[keep_mask]
            phrases = [p for i, p in enumerate(phrases) if keep_mask[i]]

        detections = ObjectDetections(
            boxes=torch.from_numpy(normalized_boxes),
            logits=torch.from_numpy(scores),
            phrases=phrases,
            image_source=image,
            fmt="xyxy",
        )

        return detections


class DFineClient:
    """Client for remote D-FINE detection service."""

    def __init__(self, port: int = 12185):
        """
        Initialize D-FINE client.

        Args:
            port: Port number where D-FINE server is running
        """
        self.url = f"http://localhost:{port}/dfine"

    def predict(
        self,
        image_numpy: np.ndarray,
        conf_thres: float = 0.4,
    ) -> ObjectDetections:
        """
        Send detection request to D-FINE server.

        Args:
            image_numpy: Input image as numpy array (H, W, 3) in RGB format
            conf_thres: Confidence threshold for filtering detections

        Returns:
            ObjectDetections object with detection results
        """
        response = send_request(
            self.url,
            image=image_numpy,
            conf_thres=conf_thres,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


# Alias for backward compatibility with YOLOv7
class ObjectDetector(DFine):
    """Alias for DFine, provides backward compatibility."""
    pass


class ObjectDetectorClient(DFineClient):
    """Alias for DFineClient, provides backward compatibility."""
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="D-FINE Detection Server")
    parser.add_argument("--port", type=int, default=12185,
                        help="Port to host the server on")
    parser.add_argument(
        "--config",
        type=str,
        default="vlm/detector/D-FINE/configs/dfine/dfine_hgnetv2_x_coco.yml",
        help="Path to D-FINE config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_weights/dfine_x_coco.pth",
        help="Path to D-FINE checkpoint file"
    )
    args = parser.parse_args()

    print("Loading D-FINE model...")

    class DFineServer(ServerMixin, DFine):
        def process_payload(self, payload: dict) -> dict:
            # Handle GET request (health check)
            if payload is None:
                return {"status": "ok", "message": "D-FINE service is running"}

            # Handle POST request (detection)
            image = str_to_image(payload["image"])
            conf_thres = payload.get("conf_thres", 0.4)
            return self.predict(image, conf_thres=conf_thres).to_json()

    dfine = DFineServer(args.config, args.checkpoint)
    print("D-FINE loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(dfine, name="dfine", port=args.port)
