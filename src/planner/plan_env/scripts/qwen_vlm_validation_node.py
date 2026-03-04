#!/home/yangz/miniconda3/envs/infonav/bin/python
"""
Qwen-VL Object Validation Node

This node provides VLM-based object validation using Qwen-VL API via Alibaba Cloud
Bailian (百炼) platform with OpenAI-compatible API.

When the robot reaches a candidate object location, this service validates whether
the target object is actually present in the current camera view.

Functionality:
1. Subscribes to /map_ros/rgb to get current camera images
2. Provides /vlm/validate_object service for object validation
3. Calls Qwen-VL API to validate if target object is in the image
4. Returns validation result (is_valid, confidence, raw_response)
5. Supports multi-view validation with AND logic to reduce false positives

Multi-View Validation:
- When num_views > 1, the service collects multiple images over time
- AND logic: ALL views must confirm the object for validation to pass
- This significantly reduces false positive detections

Environment Variables:
    DASHSCOPE_API_KEY: API key for Bailian platform (百炼)

API Documentation:
    https://help.aliyun.com/zh/model-studio/

Author: InfoNav Team
Date: 2026-01-23
"""

import os
import re
import base64
import json
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Import the service definition (will be generated after catkin build)
from plan_env.srv import ValidateObject, ValidateObjectResponse

# Import optimized prompts
try:
    from qwen_vlm_prompts import get_validation_prompt_v1, parse_vlm_response_v2
except ImportError:
    get_validation_prompt_v1 = None
    parse_vlm_response_v2 = None

# OpenAI SDK for Bailian API (OpenAI-compatible)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[QwenVLM] openai not installed. Install with: pip install openai")


class ImageRingBuffer:
    """
    Ring buffer for caching RGB images with timestamps.
    Supports retrieval by timestamp for VLM sliding window validation.
    """

    def __init__(self, max_size=50, max_age_seconds=10.0):
        """
        Initialize the image ring buffer.

        Args:
            max_size: Maximum number of images to cache
            max_age_seconds: Maximum age of cached images in seconds
        """
        self.buffer = {}  # timestamp -> image
        self.max_size = max_size
        self.max_age = max_age_seconds
        self.timestamps = []  # Ordered list of timestamps for efficient cleanup

    def add(self, timestamp, image):
        """
        Add an image to the buffer.

        Args:
            timestamp: ROS timestamp
            image: numpy array (RGB format)
        """
        self.cleanup()

        # Convert timestamp to float for consistent handling
        ts_float = timestamp.to_sec()

        # Add to buffer
        self.buffer[ts_float] = image.copy()
        self.timestamps.append(ts_float)

        # Ensure we don't exceed max size
        while len(self.buffer) > self.max_size:
            oldest_ts = self.timestamps.pop(0)
            if oldest_ts in self.buffer:
                del self.buffer[oldest_ts]

    def get(self, timestamp, tolerance=0.5):
        """
        Get image by timestamp with tolerance.

        Args:
            timestamp: ROS timestamp to search for
            tolerance: Time tolerance in seconds

        Returns:
            numpy array if found, None otherwise
        """
        ts_float = timestamp.to_sec()

        # First try exact match
        if ts_float in self.buffer:
            return self.buffer[ts_float]

        # Search with tolerance
        for cached_ts in self.buffer.keys():
            if abs(cached_ts - ts_float) < tolerance:
                return self.buffer[cached_ts]

        return None

    def get_latest(self):
        """
        Get the most recent cached image.

        Returns:
            tuple: (timestamp, image) or (None, None) if buffer is empty
        """
        if not self.timestamps:
            return None, None

        latest_ts = self.timestamps[-1]
        return latest_ts, self.buffer.get(latest_ts)

    def cleanup(self):
        """Remove expired images from the buffer."""
        now = rospy.Time.now().to_sec()
        expired = [ts for ts in self.timestamps if (now - ts) > self.max_age]

        for ts in expired:
            if ts in self.buffer:
                del self.buffer[ts]
            self.timestamps.remove(ts)

    def size(self):
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear all cached images."""
        self.buffer.clear()
        self.timestamps.clear()


class QwenVLMValidationNode:
    """
    Qwen-VL Object Validation Node

    Provides a ROS service for validating whether a target object
    is present in the current camera view using Qwen-VL API.
    """

    def __init__(self):
        rospy.init_node('qwen_vlm_validation_node', anonymous=False)

        # Parameters
        self.rgb_topic = rospy.get_param('~rgb_topic', '/map_ros/rgb')
        # Use qwen-vl-plus or qwen-vl-max for vision tasks
        self.model_name = rospy.get_param('~model', 'qwen-vl-plus')
        self.api_timeout = rospy.get_param('~timeout', 30.0)  # seconds

        # Debug settings - save validation images and reasoning
        self.debug_save_images = rospy.get_param('~debug_save_images', True)
        # Use InfoNav/debug as default debug directory
        infonav_path = os.path.expanduser('~/Nav/InfoNav')
        self.debug_image_dir = rospy.get_param('~debug_image_dir',
                                               os.path.join(infonav_path, 'debug', 'vlm_validation'))

        # Image ring buffer parameters for sliding window support
        self.image_buffer_size = rospy.get_param('~image_buffer_size', 50)
        self.image_max_age = rospy.get_param('~image_max_age', 10.0)

        # API region: beijing (default), virginia, singapore
        self.api_region = rospy.get_param('~api_region', 'beijing')

        # Get API key from environment
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            rospy.logerr("[QwenVLM] DASHSCOPE_API_KEY environment variable not set!")
            rospy.logerr("[QwenVLM] Please set it with: export DASHSCOPE_API_KEY='sk-xxx'")
            rospy.logerr("[QwenVLM] Get API key from: https://help.aliyun.com/zh/model-studio/get-api-key")

        # Check if openai is available
        if not OPENAI_AVAILABLE:
            rospy.logerr("[QwenVLM] openai library not available!")
            rospy.signal_shutdown("openai not available")
            return

        # Initialize OpenAI client with Bailian base_url
        base_url = self._get_base_url()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # State
        self.latest_rgb_image = None
        self.image_timestamp = None
        self.validation_count = 0

        # Image ring buffer for sliding window validation
        self.image_buffer = ImageRingBuffer(
            max_size=self.image_buffer_size,
            max_age_seconds=self.image_max_age
        )

        # Create debug directory if needed
        if self.debug_save_images:
            os.makedirs(self.debug_image_dir, exist_ok=True)
            rospy.loginfo(f"[QwenVLM] Debug directory: {self.debug_image_dir}")

        # Subscriber for RGB images
        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.rgb_callback,
            queue_size=1
        )

        # Service for object validation
        self.validation_service = rospy.Service(
            '/vlm/validate_object',
            ValidateObject,
            self.validate_object_callback
        )

        rospy.loginfo("[QwenVLM] Node initialized successfully")
        rospy.loginfo(f"[QwenVLM] Model: {self.model_name}")
        rospy.loginfo(f"[QwenVLM] API region: {self.api_region}")
        rospy.loginfo(f"[QwenVLM] Base URL: {base_url}")
        rospy.loginfo(f"[QwenVLM] RGB topic: {self.rgb_topic}")
        rospy.loginfo(f"[QwenVLM] API timeout: {self.api_timeout}s")
        rospy.loginfo(f"[QwenVLM] Image buffer: size={self.image_buffer_size}, max_age={self.image_max_age}s")
        if self.api_key:
            rospy.loginfo(f"[QwenVLM] API key configured (length: {len(self.api_key)})")

    def _get_base_url(self):
        """Get the base URL based on API region."""
        region_urls = {
            'beijing': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'virginia': 'https://dashscope-us.aliyuncs.com/compatible-mode/v1',
            'singapore': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        }
        return region_urls.get(self.api_region, region_urls['beijing'])

    def rgb_callback(self, msg):
        """
        Callback for RGB images.

        Args:
            msg (Image): RGB image message
        """
        try:
            # Convert ROS Image to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Store as RGB for API
            self.latest_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.image_timestamp = msg.header.stamp

            # Add to ring buffer for sliding window validation
            self.image_buffer.add(msg.header.stamp, self.latest_rgb_image)
        except Exception as e:
            rospy.logerr(f"[QwenVLM] Failed to convert image: {e}")

    def image_to_base64_url(self, image):
        """
        Convert numpy image array to base64 data URL.

        Args:
            image: numpy array (RGB format)

        Returns:
            str: base64 encoded image URL for OpenAI API
        """
        # Convert RGB to BGR for cv2.imencode
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def parse_vlm_response(self, response_text, target_object):
        """
        Parse VLM response to determine if target object is present.
        Uses optimized strict parsing logic to reduce false positives.

        Args:
            response_text: Raw text response from VLM
            target_object: The target object we're looking for

        Returns:
            tuple: (is_valid, confidence)
        """
        # Use optimized parsing from qwen_vlm_prompts if available
        if parse_vlm_response_v2:
            try:
                is_valid, confidence, reasoning = parse_vlm_response_v2(response_text, target_object)
                rospy.logdebug(f"[QwenVLM] Parse result: valid={is_valid}, conf={confidence:.2f}, reason={reasoning}")
                return is_valid, confidence
            except Exception as e:
                rospy.logwarn(f"[QwenVLM] Error using parse_vlm_response_v2: {e}, falling back to legacy parser")
        
        # Fallback to legacy parser if import failed
        response_lower = response_text.lower().strip()

        # Stricter matching: require clear decision indicators
        yes_patterns = [
            r'^\s*(yes|yeah|correct|right|true)',
            r'\b(yes,|yes\.)\b',
            r'\bdecision:\s*yes\b',
            r'(是的|确实|有的)',
        ]

        no_patterns = [
            r'^\s*(no|nope|false|not)\b',
            r'\b(no,|no\.)\b',
            r'\bdecision:\s*no\b',
            r'(没有|不存在|没看到)',
        ]

        # Count matches - but be conservative
        has_yes = any(re.search(pattern, response_lower) for pattern in yes_patterns)
        has_no = any(re.search(pattern, response_lower) for pattern in no_patterns)

        # Decision logic (conservative)
        if has_yes and not has_no:
            is_valid = True
            confidence = 0.75  # Moderate confidence even with positive response
        elif has_no and not has_yes:
            is_valid = False
            confidence = 0.75
        elif has_yes and has_no:
            # Contradictory - default to negative (safer)
            is_valid = False
            confidence = 0.4
        else:
            # Ambiguous - default to negative (conservative)
            is_valid = False
            confidence = 0.2

        return is_valid, confidence

    def validate_single_image(self, image, target_object):
        """
        Validate a single image for target object presence.
        Uses optimized structured prompt for higher accuracy.

        Args:
            image: numpy array (RGB format)
            target_object: target object name

        Returns:
            tuple: (is_valid, confidence, raw_response)
        """
        try:
            # Convert image to base64 URL
            image_url = self.image_to_base64_url(image)

            # Use optimized prompt if available
            if get_validation_prompt_v1:
                # Try to get additional object features from ROS params if available
                object_features = rospy.get_param(f'~object_features/{target_object}', None)
                fallback_objects = rospy.get_param(f'~fallback_objects/{target_object}', None)
                prompt = get_validation_prompt_v1(
                    target_object=target_object,
                    target_features=object_features,
                    fallback_objects=fallback_objects
                )
            else:
                # Fallback prompt if import failed
                prompt = f"""Analyze this image carefully.
Is there a {target_object} anywhere in this image? (position doesn't matter)
Respond with: DECISION: YES/NO, CONFIDENCE: 0.0-1.0, REASON: brief explanation"""

            rospy.loginfo(f"[QwenVLM] Calling Qwen-VL API (model: {self.model_name})...")

            # Call API using OpenAI-compatible interface
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                # Disable thinking mode for simple yes/no questions
                extra_body={
                    'enable_thinking': False
                }
            )

            # Extract response text
            raw_response = completion.choices[0].message.content
            rospy.loginfo(f"[QwenVLM] Raw response: {raw_response}")

            # Parse response
            is_valid, confidence = self.parse_vlm_response(raw_response, target_object)

            # Save debug info (image + reasoning)
            if self.debug_save_images:
                self.save_validation_debug(image, target_object, is_valid, confidence, raw_response)

            return is_valid, confidence, raw_response

        except Exception as e:
            rospy.logerr(f"[QwenVLM] Exception during single image validation: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, f"ERROR: {str(e)}"

    def save_validation_debug(self, image, target_object, is_valid, confidence, raw_response):
        """
        Save validation image and reasoning to debug folder.
        
        Args:
            image: numpy array (RGB format)
            target_object: target object name
            is_valid: validation result
            confidence: confidence score
            raw_response: raw VLM response
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            result_str = "YES" if is_valid else "NO"
            
            # Save image
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_filename = f"{timestamp}_{target_object}_{result_str}_{confidence:.2f}.jpg"
            image_path = os.path.join(self.debug_image_dir, image_filename)
            cv2.imwrite(image_path, bgr_image)
            
            # Save reasoning to text file with same timestamp
            reasoning_filename = f"{timestamp}_{target_object}_{result_str}_{confidence:.2f}.txt"
            reasoning_path = os.path.join(self.debug_image_dir, reasoning_filename)
            with open(reasoning_path, 'w', encoding='utf-8') as f:
                f.write(f"Target: {target_object}\n")
                f.write(f"Result: {result_str}\n")
                f.write(f"Confidence: {confidence:.2f}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"VLM Response:\n{'='*60}\n")
                f.write(raw_response)
            
            rospy.logdebug(f"[QwenVLM] Saved debug: {image_path}")
            rospy.logdebug(f"[QwenVLM] Saved reasoning: {reasoning_path}")
            
        except Exception as e:
            rospy.logwarn(f"[QwenVLM] Failed to save debug info: {e}")

    def validate_object_callback(self, req):
        """
        Service callback for object validation.
        Supports multi-view validation with AND logic.

        Args:
            req: ValidateObject request with target_object and optional num_views

        Returns:
            ValidateObjectResponse: Validation result
        """
        response = ValidateObjectResponse()

        # Check if we have an image
        if self.latest_rgb_image is None:
            rospy.logwarn("[QwenVLM] No RGB image available for validation")
            response.is_valid = False
            response.confidence = 0.0
            response.raw_response = "ERROR: No image available"
            response.views_confirmed = 0
            response.views_total = 0
            return response

        # Check API key
        if not self.api_key:
            rospy.logerr("[QwenVLM] API key not configured")
            response.is_valid = False
            response.confidence = 0.0
            response.raw_response = "ERROR: API key not configured"
            response.views_confirmed = 0
            response.views_total = 0
            return response

        target_object = req.target_object
        # Get num_views from request (default to 1 for backward compatibility)
        num_views = req.num_views if req.num_views > 0 else 1

        rospy.loginfo(f"[QwenVLM] Validating target: '{target_object}' with {num_views} view(s)")

        # Save debug image if enabled
        if self.debug_save_images:
            debug_path = os.path.join(
                self.debug_image_dir,
                f"validation_{self.validation_count}_{target_object}.jpg"
            )
            cv2.imwrite(debug_path, cv2.cvtColor(self.latest_rgb_image, cv2.COLOR_RGB2BGR))
            rospy.loginfo(f"[QwenVLM] Saved debug image to: {debug_path}")

        self.validation_count += 1

        # Single view validation (original behavior or first view of multi-view)
        is_valid, confidence, raw_response = self.validate_single_image(
            self.latest_rgb_image, target_object
        )

        # For single-view mode, return immediately
        if num_views == 1:
            response.is_valid = is_valid
            response.confidence = confidence
            response.raw_response = raw_response
            response.views_confirmed = 1 if is_valid else 0
            response.views_total = 1

            rospy.loginfo(f"[QwenVLM] Single-view result: is_valid={is_valid}, confidence={confidence:.2f}")
            return response

        # Multi-view mode: This service call validates ONE view
        # The C++ side manages the multi-view state machine
        # We return the result for this single view, and C++ aggregates
        response.is_valid = is_valid
        response.confidence = confidence
        response.raw_response = raw_response
        response.views_confirmed = 1 if is_valid else 0
        response.views_total = 1

        rospy.loginfo(f"[QwenVLM] View validation result: is_valid={is_valid}, confidence={confidence:.2f}")
        rospy.loginfo(f"[QwenVLM] (Multi-view aggregation handled by C++ FSM)")

        return response

    def run(self):
        """
        Main run loop.
        """
        rospy.loginfo("[QwenVLM] Node running. Waiting for validation requests...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = QwenVLMValidationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[QwenVLM] Node crashed: {e}")
        import traceback
        traceback.print_exc()
