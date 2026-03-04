"""
Manual Habitat ObjectNav Runner with VLM Verification (HM3D/MP3D)

This manual runner lets you control the agent with keyboard in Habitat,
with added VLM verification for detected objects.

New Features:
- Automatic VLM verification when detector confidence >= 0.5
- Manual VLM verification by pressing 'p' key
- VLM asks: "Did we really find {target_object}? Reply yes or no"

Usage:
    # HM3D-v2 (Default)
    python habitat_manual_control_vlm_test.py --dataset hm3dv2

    # With DashScope API
    python habitat_manual_control_vlm_test.py --dataset hm3dv2 --api-type dashscope --model qwen3-vl-flash

    # Test specific episode
    python habitat_manual_control_vlm_test.py --dataset hm3dv2 test_epi_num=10

Author: Zhiyang
"""

# Standard library imports
import argparse
import base64
import gzip
import json
import os
import signal
import time
from copy import deepcopy
from pathlib import Path

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import cv2
import rospy
from omegaconf import DictConfig
from std_msgs.msg import Float64
import requests
from openai import OpenAI

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    observations_to_image,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from habitat2ros import habitat_publisher
from vlm.utils.get_object_utils import get_object
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from vlm.Labels import MP3D_ID_TO_NAME


# Keyboard controls
FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
LOOK_UP_KEY = "q"
LOOK_DOWN_KEY = "e"
FINISH = "f"
VLM_VERIFY_KEY = "p"  # New key for manual VLM verification

# VLM verification threshold
VLM_CONFIDENCE_THRESHOLD = 0.5


def signal_handler(sig, frame):
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def publish_float64(publisher, data: float):
    """Publish a Float64 value to the given ROS publisher."""
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def print_manual_controls():
    """Print manual control key bindings for the player."""
    print("\nManual controls:")
    print(f"  {FORWARD_KEY} - Move forward")
    print(f"  {LEFT_KEY} - Turn left")
    print(f"  {RIGHT_KEY} - Turn right")
    print(f"  {LOOK_UP_KEY} - Look up")
    print(f"  {LOOK_DOWN_KEY} - Look down")
    print(f"  {FINISH} - Stop (end episode)")
    print(f"  {VLM_VERIFY_KEY} - Manual VLM verification")
    print("  Ctrl+C - Quit (graceful shutdown)")
    print("Note: Focus the 'Observations' window before pressing keys.\n")


def publish_observations(event):
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Habitat Manual Runner with VLM", add_help=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["ollama", "dashscope"],
        default="dashscope",
        help="VLM API type: ollama (local) or dashscope (Alibaba Cloud)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="VLM API URL (default: auto based on api-type)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="VLM model name (default: auto based on api-type)",
    )
    args, unknown = parser.parse_known_args()
    return args.dataset, args.api_type, args.api_url, args.model, unknown


# ==================== VLM Verification Class ====================

class VLMObjectVerifier:
    """VLM-based object verification for detected objects."""

    API_OLLAMA = "ollama"
    API_DASHSCOPE = "dashscope"

    def __init__(self, api_type: str = "dashscope", api_url: str = None, model: str = None, timeout: int = 60):
        """
        Initialize VLM verifier.

        Args:
            api_type: "ollama" or "dashscope"
            api_url: API endpoint URL
            model: Model name
            timeout: Request timeout in seconds
        """
        self.api_type = api_type
        self.timeout = timeout

        # Set defaults based on api_type
        if api_type == self.API_DASHSCOPE:
            self.api_url = api_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model = model or "qwen3-vl-flash"
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY environment variable is required for DashScope API")
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=self.api_url,
            )
        else:  # ollama
            self.api_url = api_url or "http://localhost:20004/api/chat"
            self.model = model or "qwen3-vl:32b"
            self.openai_client = None

    def encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        # Convert BGR to RGB if needed, then encode as PNG
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')

    def verify_object(self, image: np.ndarray, target_object: str) -> tuple:
        """
        Verify if the target object is really detected in the image.

        Args:
            image: RGB image (numpy array)
            target_object: Target object name

        Returns:
            (result: str, is_found: bool, duration: float)
        """
        prompt = f"""Look at this image carefully. Did we really find a "{target_object}" in this image?
Please answer with ONLY "yes" or "no", followed by a brief reason (one sentence).
Format: [yes/no]: reason"""

        if self.api_type == self.API_DASHSCOPE:
            return self._verify_dashscope(image, prompt)
        else:
            return self._verify_ollama(image, prompt)

    def _verify_dashscope(self, image: np.ndarray, prompt: str) -> tuple:
        """Call DashScope API for verification."""
        img_base64 = self.encode_image(image)

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        t0 = time.time()
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
            )
            duration = time.time() - t0
            response = completion.choices[0].message.content.strip()

            # Parse yes/no from response
            is_found = response.lower().startswith("yes")

            return response, is_found, duration

        except Exception as e:
            duration = time.time() - t0
            return f"Error: {str(e)}", False, duration

    def _verify_ollama(self, image: np.ndarray, prompt: str) -> tuple:
        """Call Ollama API for verification."""
        img_base64 = self.encode_image(image)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_base64]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        t0 = time.time()
        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            duration = time.time() - t0

            if resp.status_code != 200:
                return f"Error: HTTP {resp.status_code}", False, duration

            result = resp.json()
            response = result.get("message", {}).get("content", "").strip()

            # Parse yes/no from response
            is_found = response.lower().startswith("yes")

            return response, is_found, duration

        except Exception as e:
            duration = time.time() - t0
            return f"Error: {str(e)}", False, duration


def main(cfg: DictConfig, vlm_verifier: VLMObjectVerifier) -> None:
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub

    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    score_list = []
    object_masks_list = []
    label_list = []
    llm_answer = []

    cfg = patch_config(cfg)
    env_count = 0 if cfg.test_epi_num == -1 else cfg.test_epi_num
    detector_cfg = cfg.detector

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    # Initialize Habitat environment
    env = habitat.Env(cfg)
    print("Environment creation successful")

    # Skip episodes to reach the desired starting index
    while env_count:
        env.current_episode = next(env.episode_iterator)
        env_count -= 1
    observations = env.reset()
    observations["rgb"] = transform_rgb_bgr(observations["rgb"])

    # Display first observation frame
    info = env.get_metrics()
    frame = observations_to_image(observations, info)
    cv2.imshow("Observations", frame)

    camera_pitch = 0.0
    observations["camera_pitch"] = camera_pitch
    msg_observations = deepcopy(observations)

    # Initialize ROS publishers and timer for periodic observation publishing
    ros_pub = habitat_publisher.ROSPublisher()
    timer = rospy.Timer(rospy.Duration(0.1), publish_observations)
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )

    print("Agent stepping around inside environment.")
    print_manual_controls()

    label = env.current_episode.object_category

    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    # Default values (LLM disabled)
    llm_answer = [label]
    room = "unknown"
    fusion_threshold = 0.5

    cld_with_score_msg = MultipleMasksWithConfidence()
    count_steps = 0

    # Store current RGB for VLM verification
    current_rgb = observations["rgb"].copy()

    # Manual control loop
    while not rospy.is_shutdown() and not env.episode_over:
        print(f"\n-------------Step: {count_steps}-------------")
        keystroke = cv2.waitKey(0)

        # Check for VLM verification key first
        if keystroke == ord(VLM_VERIFY_KEY):
            print("\n" + "=" * 50)
            print(f"[VLM Manual Verification] Target: {label}")
            print("=" * 50)
            print("Calling VLM API...")

            vlm_response, is_found, duration = vlm_verifier.verify_object(current_rgb, label)

            print(f"\n[VLM Result] (took {duration:.2f}s)")
            print(f"  Response: {vlm_response}")
            print(f"  Is Found: {'YES' if is_found else 'NO'}")
            print("=" * 50 + "\n")
            continue

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LOOK_UP_KEY):
            action = HabitatSimActions.look_up
            camera_pitch = camera_pitch + np.pi / 6.0
            print("action: LOOK_UP")
        elif keystroke == ord(LOOK_DOWN_KEY):
            action = HabitatSimActions.look_down
            camera_pitch = camera_pitch - np.pi / 6.0
            print("action: LOOK_DOWN")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        timer.shutdown()
        print(f"I'm finding {label}")
        observations = env.step(action)
        count_steps += 1

        info = env.get_metrics()

        # Detect objects in the current observation
        detect_img, score_list, object_masks_list, label_list = get_object(
            label, observations["rgb"], detector_cfg, llm_answer
        )

        # Store current RGB for VLM verification (before BGR conversion)
        current_rgb = observations["rgb"].copy()

        observations["rgb"] = detect_img
        observations["camera_pitch"] = camera_pitch
        ros_pub.habitat_publish_ros_topic(observations)
        observations["rgb"] = transform_rgb_bgr(detect_img)
        del observations["camera_pitch"]
        frame = observations_to_image(observations, info)

        # Generate and publish object point clouds
        obj_point_cloud_list = get_object_point_cloud(
            cfg, observations, object_masks_list
        )

        cld_with_score_msg.point_clouds = obj_point_cloud_list
        cld_with_score_msg.confidence_scores = score_list
        cld_with_score_msg.label_indices = label_list
        cld_with_score_pub.publish(cld_with_score_msg)

        # ==================== Automatic VLM Verification ====================
        # Check if any detection score >= threshold (target object, label_list == 0)
        target_detected = False
        max_target_score = 0.0

        for i, (score, lbl_idx) in enumerate(zip(score_list, label_list)):
            if lbl_idx == 0 and score >= VLM_CONFIDENCE_THRESHOLD:
                target_detected = True
                max_target_score = max(max_target_score, score)

        if target_detected:
            print("\n" + "=" * 50)
            print(f"[Auto VLM Verification] Detector confidence: {max_target_score:.3f} >= {VLM_CONFIDENCE_THRESHOLD}")
            print(f"Target: {label}")
            print("=" * 50)
            print("Calling VLM API for verification...")

            vlm_response, is_found, duration = vlm_verifier.verify_object(current_rgb, label)

            print(f"\n[VLM Result] (took {duration:.2f}s)")
            print(f"  Response: {vlm_response}")
            print(f"  Is Found: {'YES - Target Confirmed!' if is_found else 'NO - False Positive'}")
            print("=" * 50 + "\n")

        # Show updated visualization frame
        cv2.imshow("Observations", frame)

    env.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_ros_publisher", anonymous=True)

    try:
        dataset, api_type, api_url, model, overrides = _parse_dataset_arg()

        # Initialize VLM verifier
        print("\n" + "=" * 50)
        print("Initializing VLM Object Verifier")
        print(f"  API Type: {api_type}")
        print(f"  API URL: {api_url or 'default'}")
        print(f"  Model: {model or 'default'}")
        print("=" * 50 + "\n")

        vlm_verifier = VLMObjectVerifier(
            api_type=api_type,
            api_url=api_url,
            model=model
        )

        cfg_name = f"habitat_eval_{dataset}"
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg, vlm_verifier)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
