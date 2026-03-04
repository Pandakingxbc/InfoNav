#!/home/yangz/miniconda3/envs/infonav/bin/python
"""
VLM Approach Verifier Node

This node handles VLM-based verification when the robot approaches a target object.
It implements a sliding window mechanism to collect high-confidence detection frames
and triggers VLM verification to confirm true positives.

Key Features:
1. Subscribes to ObjectUpdateInfo to collect frames that update the target object
2. Subscribes to VLMVerificationRequest when FSM triggers verification
3. Uses a sliding window to select top-k high-confidence frames for VLM
4. Calls DashScope Qwen VLM API (qwen3vl-flash) for verification
5. Publishes VLMVerificationResult to C++ for confidence adjustment

Synchronous Mode:
- When VLM verification is triggered, FSM stops outputting actions
- VLM result determines whether to boost or penalize object confidence
- 30 second timeout disables VLM for the rest of the episode

Author: InfoNav Team
Date: 2026-02
"""

import os
import re
import time
import base64
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

# Import message types (generated after catkin build)
from plan_env.msg import (
    ObjectUpdateInfo,
    VLMVerificationRequest,
    VLMVerificationResult,
)
from std_msgs.msg import Int32

# OpenAI SDK for DashScope API (OpenAI-compatible)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[VLMApproach] openai not installed. Install with: pip install openai")


@dataclass
class CandidateFrame:
    """A candidate frame collected during approach."""
    timestamp: float           # ROS timestamp in seconds
    object_id: int             # Object ID that was updated
    confidence: float          # Detection confidence
    distance: float            # Distance to object at detection time
    image: Optional[np.ndarray] = None      # RGB image with detection boxes (if captured)
    raw_image: Optional[np.ndarray] = None  # Original RGB image without detection boxes


class VLMApproachVerifier:
    """
    VLM Approach Verification Handler.

    Collects high-confidence detection frames during approach and
    performs VLM verification when triggered by C++ FSM.
    """

    # Sliding window parameters
    WINDOW_SIZE = 10            # Maximum frames to keep in navigation cache
    MIN_CONFIDENCE = 0.2       # Minimum confidence to include in window (lowered to capture more frames)
    TOP_K_FRAMES = 5          # Number of top frames to send to VLM
    MIN_FRAMES_FOR_VLM = 1     # Minimum frames required (dynamic top-k)

    # Global history cache parameters
    GLOBAL_HISTORY_SIZE = 10    # Max frames per object in global history
    MAX_OBJECTS_IN_HISTORY = 20  # Max number of objects to track in history

    # Approach frames parameters (frames collected during approach regardless of detection)
    APPROACH_FRAMES_SIZE = 15  # Maximum frames to keep in approach cache
    MIN_APPROACH_FRAMES_FOR_VLM = 5  # Minimum approach frames to use for supplementary VLM verification

    # VLM parameters
    VLM_TIMEOUT = 90.0         # Timeout in seconds (increased for qwen3-vl-plus)
    VLM_MODEL = "qwen3-vl-plus"  # Use qwen3-vl-plus for better accuracy

    def __init__(self):
        """Initialize the VLM Approach Verifier node."""
        rospy.init_node('vlm_approach_verifier', anonymous=False)

        # Parameters
        self.rgb_topic = rospy.get_param('~rgb_topic', '/habitat/camera_rgb')
        self.rgb_raw_topic = rospy.get_param('~rgb_raw_topic', '/habitat/camera_rgb_raw')
        self.model_name = rospy.get_param('~model', self.VLM_MODEL)
        self.api_timeout = rospy.get_param('~timeout', self.VLM_TIMEOUT)
        self.window_size = rospy.get_param('~window_size', self.WINDOW_SIZE)
        self.min_confidence = rospy.get_param('~min_confidence', self.MIN_CONFIDENCE)
        self.top_k = rospy.get_param('~top_k', self.TOP_K_FRAMES)

        # Debug settings
        self.debug_save_images = rospy.get_param('~debug_save_images', True)
        infonav_path = os.path.expanduser('~/Nav/InfoNav')
        self.debug_dir = rospy.get_param(
            '~debug_dir',
            os.path.join(infonav_path, 'debug', 'vlm_approach')
        )

        # API configuration
        self.api_region = rospy.get_param('~api_region', 'beijing')
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            rospy.logerr("[VLMApproach] DASHSCOPE_API_KEY not set!")

        # Initialize OpenAI client
        if OPENAI_AVAILABLE and self.api_key:
            base_url = self._get_base_url()
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = None

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # State - Triple-layer cache design
        # Layer 1: Navigation cache - frames collected during current navigation to target
        # Only frames with high-confidence detection of target object
        # Cleared when target switches or entering frontier exploration
        self.navigation_frames: deque = deque(maxlen=self.window_size)
        self.current_nav_target_id = -1  # Current navigation target being approached

        # Layer 2: Global history cache - high-confidence frames for all objects in episode
        # Key: object_id, Value: list of top-k highest confidence frames
        # Persists throughout episode, only cleared on episode reset
        self.global_history: Dict[int, List[CandidateFrame]] = defaultdict(list)
        self.global_history_size = rospy.get_param('~global_history_size', self.GLOBAL_HISTORY_SIZE)

        # Layer 3: Approach frames cache - ALL frames collected during approach (regardless of detection)
        # Used as fallback when no high-confidence detection frames available (hallucination scenario)
        # Cleared when target switches or entering frontier exploration
        self.approach_frames: deque = deque(maxlen=self.APPROACH_FRAMES_SIZE)
        self.approach_target_id = -1  # Target ID being approached

        # Legacy alias for backward compatibility
        self.candidate_frames = self.navigation_frames
        self.current_target_id = -1

        self.latest_rgb_image: Optional[np.ndarray] = None           # Image with detection boxes
        self.latest_rgb_raw_image: Optional[np.ndarray] = None      # Raw image without detection boxes
        self.latest_image_timestamp: Optional[float] = None
        self.latest_raw_image_timestamp: Optional[float] = None
        self.verification_lock = threading.Lock()
        self.verification_count = 0

        # Episode tracking
        self.vlm_disabled_this_episode = False
        self.vlm_used_this_episode = False
        self.vlm_verify_count = 0
        self.episode_id = 0  # Incremented on each episode reset
        self.verification_in_progress = False  # Track if verification is running

        # Create debug directory
        if self.debug_save_images:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Subscribers
        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic, Image, self.rgb_callback, queue_size=1
        )
        self.rgb_raw_sub = rospy.Subscriber(
            self.rgb_raw_topic, Image, self.rgb_raw_callback, queue_size=1
        )
        self.object_update_sub = rospy.Subscriber(
            '/vlm/object_update_info', ObjectUpdateInfo,
            self.object_update_callback, queue_size=10
        )
        self.verification_request_sub = rospy.Subscriber(
            '/vlm/verification_request', VLMVerificationRequest,
            self.verification_request_callback, queue_size=1
        )
        self.episode_reset_sub = rospy.Subscriber(
            '/habitat/episode_reset', Empty,
            self.episode_reset_callback, queue_size=1
        )
        # Subscribe to target switch notification from C++ FSM
        # When target switches or enters frontier exploration, clear navigation cache
        self.target_switch_sub = rospy.Subscriber(
            '/vlm/target_switch', Int32,
            self.target_switch_callback, queue_size=1
        )

        # Publisher
        self.result_pub = rospy.Publisher(
            '/vlm/verification_result', VLMVerificationResult, queue_size=1
        )

        rospy.loginfo("[VLMApproach] Node initialized")
        rospy.loginfo(f"[VLMApproach] Model: {self.model_name}")
        rospy.loginfo(f"[VLMApproach] Navigation cache size: {self.window_size}, Global history size: {self.global_history_size}")
        rospy.loginfo(f"[VLMApproach] Top-K: {self.top_k}, Min confidence: {self.min_confidence}")

    def _get_base_url(self) -> str:
        """Get DashScope API base URL based on region."""
        region_urls = {
            'beijing': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'virginia': 'https://dashscope-us.aliyuncs.com/compatible-mode/v1',
            'singapore': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        }
        return region_urls.get(self.api_region, region_urls['beijing'])

    def rgb_callback(self, msg: Image):
        """Cache the latest RGB image (with detection boxes)."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.latest_image_timestamp = msg.header.stamp.to_sec()
        except Exception as e:
            rospy.logerr(f"[VLMApproach] Failed to convert image: {e}")

    def rgb_raw_callback(self, msg: Image):
        """
        Cache the latest raw RGB image (without detection boxes).
        Also collects approach frames when navigating to a target object.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_raw_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.latest_raw_image_timestamp = msg.header.stamp.to_sec()

            # Collect approach frame if we have an active target (regardless of detection)
            # This provides fallback images when target is not detected (hallucination scenario)
            if self.approach_target_id >= 0 and self.latest_rgb_raw_image is not None:
                approach_frame = CandidateFrame(
                    timestamp=self.latest_raw_image_timestamp,
                    object_id=self.approach_target_id,
                    confidence=0.0,  # No detection confidence for approach frames
                    distance=-1.0,   # Distance unknown without detection
                    image=None,      # No annotated image
                    raw_image=self.latest_rgb_raw_image.copy()
                )
                with self.verification_lock:
                    self.approach_frames.append(approach_frame)

        except Exception as e:
            rospy.logerr(f"[VLMApproach] Failed to convert raw image: {e}")

    def object_update_callback(self, msg: ObjectUpdateInfo):
        """
        Collect frames for ALL detected objects (not just current target).
        Updates navigation cache (for current target) and global history cache (for all objects).
        Saves both the annotated image (with detection boxes) and raw image (without boxes).
        """
        # Skip low confidence detections
        if msg.detection_confidence < self.min_confidence:
            return

        # Skip if no image available
        if self.latest_rgb_image is None:
            return

        # Create candidate frame with both annotated and raw images
        frame = CandidateFrame(
            timestamp=msg.header.stamp.to_sec(),
            object_id=msg.object_id,
            confidence=msg.detection_confidence,
            distance=msg.distance_to_object,
            image=self.latest_rgb_image.copy(),
            raw_image=self.latest_rgb_raw_image.copy() if self.latest_rgb_raw_image is not None else None
        )

        with self.verification_lock:
            # Layer 1: Add to navigation cache (current target only)
            if msg.is_current_target:
                self.navigation_frames.append(frame)
                self.current_target_id = msg.object_id
                self.current_nav_target_id = msg.object_id

            # Layer 2: Add to global history cache (for ALL objects)
            self._update_global_history(frame)

        rospy.logdebug(
            f"[VLMApproach] Collected frame: obj={msg.object_id}, is_target={msg.is_current_target}, "
            f"conf={msg.detection_confidence:.3f}, dist={msg.distance_to_object:.2f}, "
            f"nav_cache={len(self.navigation_frames)}, "
            f"global_history[{msg.object_id}]={len(self.global_history.get(msg.object_id, []))}"
        )

    def _update_global_history(self, frame: CandidateFrame):
        """
        Update global history cache with a new frame.
        Maintains top-k highest confidence frames per object.
        Must be called with verification_lock held.
        """
        object_id = frame.object_id
        history = self.global_history[object_id]

        # Add new frame
        history.append(frame)

        # Keep only top-k highest confidence frames
        if len(history) > self.global_history_size:
            # Sort by confidence (descending) and keep top-k
            history.sort(key=lambda f: f.confidence, reverse=True)
            self.global_history[object_id] = history[:self.global_history_size]

    def target_switch_callback(self, msg: Int32):
        """
        Handle target switch notification from C++ FSM.
        Clears navigation cache and approach frames cache when target changes or enters frontier exploration.

        msg.data meanings:
        - -1: Entering frontier exploration (no specific target)
        - >= 0: New target object ID
        """
        new_target_id = msg.data

        with self.verification_lock:
            old_target_id = self.current_nav_target_id

            if new_target_id == -1:
                # Entering frontier exploration mode
                rospy.loginfo(
                    f"[VLMApproach] Target switch: entering frontier exploration, "
                    f"clearing navigation cache (had {len(self.navigation_frames)} frames for obj {old_target_id}), "
                    f"approach cache (had {len(self.approach_frames)} frames)"
                )
            elif new_target_id != old_target_id:
                rospy.loginfo(
                    f"[VLMApproach] Target switch: {old_target_id} -> {new_target_id}, "
                    f"clearing navigation cache (had {len(self.navigation_frames)} frames), "
                    f"approach cache (had {len(self.approach_frames)} frames)"
                )
            else:
                # Same target, no need to clear
                return

            # Clear navigation cache and approach frames cache
            self.navigation_frames.clear()
            self.approach_frames.clear()
            self.current_nav_target_id = new_target_id
            self.approach_target_id = new_target_id  # Start collecting approach frames for new target

    def verification_request_callback(self, msg: VLMVerificationRequest):
        """
        Handle VLM verification request from C++ FSM.
        This is called when the robot approaches close enough to the target.
        """
        if self.vlm_disabled_this_episode:
            rospy.logwarn("[VLMApproach] VLM disabled for this episode, sending timeout result")
            self._publish_timeout_result(msg.object_id)
            return

        # Skip if another verification is already in progress
        if self.verification_in_progress:
            rospy.logwarn("[VLMApproach] Verification already in progress, skipping request")
            return

        rospy.loginfo(
            f"[VLMApproach] Verification request: object_id={msg.object_id}, "
            f"trigger_type={msg.trigger_type}, distance={msg.distance_to_target:.2f}, "
            f"fused_conf={msg.target_fused_confidence:.3f}, obs_count={msg.target_observation_count}, "
            f"threshold={msg.current_threshold:.3f}, episode_id={self.episode_id}"
        )

        # Extract object map confidence info for VLM prompt
        object_map_info = {
            'target_fused_confidence': msg.target_fused_confidence,
            'target_observation_count': msg.target_observation_count,
            'similar_objects_info': msg.similar_objects_info,
            'current_threshold': msg.current_threshold,
        }

        # Run verification in separate thread to not block ROS
        # Pass episode_id to detect stale results
        current_episode_id = self.episode_id
        self.verification_in_progress = True

        thread = threading.Thread(
            target=self._perform_verification,
            args=(msg.object_id, msg.target_category, msg.trigger_type,
                  msg.distance_to_target, current_episode_id, object_map_info)
        )
        thread.start()

    def _perform_verification(self, object_id: int, target_category: str,
                               trigger_type: int, distance: float,
                               request_episode_id: int = -1,
                               object_map_info: Optional[Dict] = None):
        """
        Perform the actual VLM verification.

        Uses dual-layer cache selection:
        1. First, try navigation cache (frames from current approach to this target)
        2. If not enough, supplement from global history cache

        Args:
            object_id: Target object ID
            target_category: Target object category name
            trigger_type: 0=distance, 1=early, 2=rescue
            distance: Current distance to target
            request_episode_id: Episode ID when request was made (for stale detection)
            object_map_info: Dict containing object map confidence info for VLM prompt
        """
        start_time = time.time()

        try:
            # Check if episode has changed since request was made
            if request_episode_id >= 0 and request_episode_id != self.episode_id:
                rospy.logwarn(
                    f"[VLMApproach] Stale verification request (episode {request_episode_id} vs current {self.episode_id}), skipping"
                )
                return

            # Select best frames using dual-layer cache
            with self.verification_lock:
                selected_frames = self._select_best_frames_for_object(object_id)

            # If we have fewer frames than desired, try to supplement with current frame
            if len(selected_frames) < self.top_k and self.latest_rgb_image is not None:
                # Add current frame as supplementary evidence (include both raw and annotated)
                current_frame = CandidateFrame(
                    timestamp=self.latest_image_timestamp or time.time(),
                    object_id=object_id,
                    confidence=0.5,  # Default confidence for current frame
                    distance=distance,
                    image=self.latest_rgb_image.copy(),
                    raw_image=self.latest_rgb_raw_image.copy() if self.latest_rgb_raw_image is not None else None
                )
                # Avoid duplicates (check if we already have a very recent frame)
                if not selected_frames or (time.time() - selected_frames[-1].timestamp > 0.5):
                    selected_frames.append(current_frame)
                    rospy.loginfo(f"[VLMApproach] Added current frame as supplement, now have {len(selected_frames)} frames")

            if not selected_frames:
                rospy.logwarn("[VLMApproach] No candidate frames, using current frame only")
                if self.latest_rgb_image is not None:
                    selected_frames = [CandidateFrame(
                        timestamp=self.latest_image_timestamp or time.time(),
                        object_id=object_id,
                        confidence=0.5,
                        distance=distance,
                        image=self.latest_rgb_image.copy(),
                        raw_image=self.latest_rgb_raw_image.copy() if self.latest_rgb_raw_image is not None else None
                    )]
                else:
                    rospy.logerr("[VLMApproach] No image available for verification")
                    self._publish_result(object_id, 0, 0.0, "No image available", False)  # UNCERTAIN
                    return

            # Check again if episode changed during frame selection
            if request_episode_id >= 0 and request_episode_id != self.episode_id:
                rospy.logwarn(
                    f"[VLMApproach] Episode changed during verification, discarding result"
                )
                return

            # Call VLM API - returns 3-level decision
            decision_level, vlm_confidence, reason = self._call_vlm_api(
                selected_frames, target_category, object_map_info
            )

            # Final check if episode changed during VLM API call
            if request_episode_id >= 0 and request_episode_id != self.episode_id:
                rospy.logwarn(
                    f"[VLMApproach] Episode changed during VLM API call, discarding result"
                )
                return

            # Check timeout
            duration = time.time() - start_time
            timeout = duration > self.VLM_TIMEOUT

            if timeout:
                rospy.logwarn(f"[VLMApproach] VLM timeout after {duration:.2f}s, disabling for episode")
                self.vlm_disabled_this_episode = True

            # Publish result with decision_level
            self._publish_result(object_id, decision_level, vlm_confidence, reason, timeout, duration)

            # Update tracking
            self.vlm_used_this_episode = True
            self.vlm_verify_count += 1
            self.verification_count += 1

            # Save debug info
            if self.debug_save_images:
                self._save_debug_info(
                    selected_frames, target_category, decision_level,
                    vlm_confidence, reason, duration
                )

            decision_name = {1: "CONFIRM", 0: "UNCERTAIN", -1: "REJECT"}
            rospy.loginfo(
                f"[VLMApproach] Verification complete: decision={decision_name[decision_level]}, "
                f"conf={vlm_confidence:.3f}, duration={duration:.2f}s"
            )

        except Exception as e:
            rospy.logerr(f"[VLMApproach] Verification error: {e}")
            import traceback
            traceback.print_exc()
            self._publish_result(object_id, 0, 0.0, f"Error: {str(e)}", False)  # UNCERTAIN on error

        finally:
            # Always reset verification_in_progress flag
            self.verification_in_progress = False

    def _select_best_frames_for_object(self, object_id: int) -> List[CandidateFrame]:
        """
        Select the best frames for a specific object using triple-layer cache.

        Priority:
        1. Navigation cache frames (high-confidence detection frames from current approach)
        2. Global history cache frames (if navigation cache is insufficient)
        3. Approach frames cache (fallback: frames collected during approach without detection)
           - Used when no high-confidence frames available (hallucination scenario)

        Only selects frames that match the requested object_id.
        Must be called with verification_lock held.

        Args:
            object_id: The target object ID to select frames for

        Returns:
            List of selected CandidateFrame objects
        """
        # Layer 1: Get frames from navigation cache matching this object_id
        nav_frames = [f for f in self.navigation_frames if f.object_id == object_id]

        # Layer 2: Get frames from global history for this object
        history_frames = self.global_history.get(object_id, [])

        # Layer 3: Get approach frames (fallback for hallucination scenario)
        approach_frames_for_obj = [f for f in self.approach_frames if f.object_id == object_id]

        rospy.loginfo(
            f"[VLMApproach] Frame sources for object {object_id}: "
            f"navigation_cache={len(nav_frames)}, global_history={len(history_frames)}, "
            f"approach_frames={len(approach_frames_for_obj)}"
        )

        # Combine frames, prioritizing navigation cache (more recent)
        # Use a set to avoid duplicates (based on timestamp)
        seen_timestamps = set()
        combined_frames = []

        # Add navigation frames first (priority)
        for frame in nav_frames:
            if frame.timestamp not in seen_timestamps:
                combined_frames.append(frame)
                seen_timestamps.add(frame.timestamp)

        # Supplement from global history if needed
        if len(combined_frames) < self.top_k:
            for frame in history_frames:
                if frame.timestamp not in seen_timestamps:
                    combined_frames.append(frame)
                    seen_timestamps.add(frame.timestamp)

        # Check if we have enough high-confidence frames
        high_conf_frame_count = len(combined_frames)

        # If no high-confidence frames available (hallucination scenario),
        # use approach frames as fallback
        if high_conf_frame_count == 0 and len(approach_frames_for_obj) >= self.MIN_APPROACH_FRAMES_FOR_VLM:
            rospy.logwarn(
                f"[VLMApproach] No high-confidence frames for object {object_id}, "
                f"using {len(approach_frames_for_obj)} approach frames as fallback (hallucination scenario)"
            )
            # Select approach frames, sorted by confidence (descending), then by recency
            # Note: approach frames typically have confidence=0.0, so recency (timestamp) is secondary
            sorted_approach = sorted(approach_frames_for_obj,
                                     key=lambda f: (f.confidence, f.timestamp), reverse=True)
            recent_approach_frames = sorted_approach[:self.MIN_APPROACH_FRAMES_FOR_VLM]

            rospy.loginfo(
                f"[VLMApproach] Selected {len(recent_approach_frames)} approach frames for verification, "
                f"confidences: {[f'{f.confidence:.3f}' for f in recent_approach_frames]}"
            )
            return recent_approach_frames

        # If still not enough, supplement with approach frames
        if len(combined_frames) < self.MIN_APPROACH_FRAMES_FOR_VLM and approach_frames_for_obj:
            needed = self.MIN_APPROACH_FRAMES_FOR_VLM - len(combined_frames)
            # Take the most recent approach frames
            recent_approach = list(approach_frames_for_obj)[-needed:]
            for frame in recent_approach:
                if frame.timestamp not in seen_timestamps:
                    combined_frames.append(frame)
                    seen_timestamps.add(frame.timestamp)
            rospy.loginfo(
                f"[VLMApproach] Supplemented with {len(recent_approach)} approach frames, "
                f"total now: {len(combined_frames)}"
            )

        if not combined_frames:
            rospy.logwarn(f"[VLMApproach] No frames available for object {object_id}")
            return []

        # Sort by confidence (descending) and select top-k
        # Note: approach frames have confidence=0.0, so they will be at the end
        sorted_frames = sorted(combined_frames, key=lambda f: f.confidence, reverse=True)

        # Dynamic top-k selection
        k = min(self.top_k, len(sorted_frames))
        k = max(k, self.MIN_FRAMES_FOR_VLM)

        selected = sorted_frames[:k]

        rospy.loginfo(
            f"[VLMApproach] Selected {len(selected)} frames for object {object_id}, "
            f"confidences: {[f'{f.confidence:.3f}' for f in selected]}, "
            f"high_conf_count: {high_conf_frame_count}"
        )

        return selected

    def _select_best_frames(self) -> List[CandidateFrame]:
        """
        Legacy method for backward compatibility.
        Select the top-k highest confidence frames from the navigation cache.
        """
        if not self.navigation_frames:
            return []

        sorted_frames = sorted(
            self.navigation_frames,
            key=lambda f: f.confidence,
            reverse=True
        )

        k = min(self.top_k, len(sorted_frames))
        k = max(k, self.MIN_FRAMES_FOR_VLM)

        return sorted_frames[:k]

    def _call_vlm_api(self, frames: List[CandidateFrame],
                       target_category: str,
                       object_map_info: Optional[Dict] = None) -> Tuple[int, float, str]:
        """
        Call VLM API with selected frames.

        Sends top-3 high-confidence raw images (without detection boxes) for cleaner
        visual verification, PLUS one annotated image with highest detection confidence
        to show the detector's bounding box results.

        Args:
            frames: List of candidate frames to verify
            target_category: Target object category name
            object_map_info: Dict containing object map confidence info for VLM prompt

        Returns:
            Tuple of (decision_level, confidence, reason)
            - decision_level: 1 (CONFIRM), 0 (UNCERTAIN), -1 (REJECT)
            - confidence: VLM's confidence in its decision (0.0-1.0)
            - reason: Explanation string
        """
        if not self.client:
            return 0, 0.0, "VLM client not initialized"  # Return UNCERTAIN on error

        try:
            # Select top-3 frames with raw images, sorted by confidence
            frames_with_raw = [f for f in frames if f.raw_image is not None]
            frames_with_raw.sort(key=lambda f: f.confidence, reverse=True)
            selected_raw_frames = frames_with_raw[:3]

            # Find the frame with highest detection confidence that has annotated image
            frames_with_annotated = [f for f in frames if f.image is not None and f.confidence > 0]
            frames_with_annotated.sort(key=lambda f: f.confidence, reverse=True)
            best_annotated_frame = frames_with_annotated[0] if frames_with_annotated else None

            # Fallback: if no raw images, use annotated images
            if not selected_raw_frames:
                rospy.logwarn("[VLMApproach] No raw images available, falling back to annotated images")
                selected_raw_frames = frames_with_annotated[:4]
                use_raw = False
                best_annotated_frame = None  # Already using annotated images, no need for extra
            else:
                use_raw = True

            num_images = len(selected_raw_frames)
            if num_images == 0:
                return 0, 0.0, "No images available for verification"

            # Count total images (raw + 1 annotated if available)
            total_images = num_images + (1 if best_annotated_frame is not None else 0)

            # Build prompt with object map confidence info
            prompt = self._build_verification_prompt(
                target_category, total_images, object_map_info
            )

            # Build message content with raw images
            content = []
            for frame in selected_raw_frames:
                if use_raw and frame.raw_image is not None:
                    image_url = self._image_to_base64_url(frame.raw_image)
                elif frame.image is not None:
                    image_url = self._image_to_base64_url(frame.image)
                else:
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })

            # Add ONE annotated image with highest detection confidence
            if best_annotated_frame is not None:
                annotated_url = self._image_to_base64_url(best_annotated_frame.image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": annotated_url}
                })
                rospy.loginfo(f"[VLMApproach] Added best annotated frame with conf={best_annotated_frame.confidence:.2f}")

            content.append({
                "type": "text",
                "text": prompt
            })

            conf_list = [f"{f.confidence:.2f}" for f in selected_raw_frames]
            rospy.loginfo(f"[VLMApproach] Calling VLM API with {num_images} raw + {1 if best_annotated_frame else 0} annotated image(s), raw confidences: {conf_list}")

            # Call API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                extra_body={'enable_thinking': False}
            )

            # Parse response
            response_text = completion.choices[0].message.content
            rospy.loginfo(f"[VLMApproach] VLM response: {response_text}")

            decision_level, confidence, reason = self._parse_vlm_response(response_text, target_category)
            return decision_level, confidence, reason

        except Exception as e:
            rospy.logerr(f"[VLMApproach] VLM API error: {e}")
            return 0, 0.0, f"API error: {str(e)}"  # Return UNCERTAIN on error

    # Objects that need stricter VLM verification due to high false positive rates
    # HM3D categories
    HIGH_FP_CATEGORIES = {'chair', 'couch', 'toilet'}
    # MP3D additional high FP categories (based on similarity scores in llm_answer_mp3d.txt)
    MP3D_HIGH_FP_CATEGORIES = {'chair', 'couch', 'toilet', 'stool', 'seating', 'counter', 'table'}

    def _build_verification_prompt(self, target_category: str, num_images: int,
                                     _object_map_info: Optional[Dict] = None) -> str:
        """
        Build VLM verification prompt with 3-level decision support.

        Decision levels:
        - CONFIRM (1): High confidence this IS the target object
        - UNCERTAIN (0): Cannot determine, keep original confidence
        - REJECT (-1): High confidence this is NOT the target (false positive)

        Args:
            target_category: Target object category name
            num_images: Total number of images being sent
            object_map_info: Dict containing object map confidence info (not used in prompt to avoid bias)
        """
        # Normalize category name (e.g., "potted plant" -> "plant")
        display_category = target_category.lower().replace("potted ", "")

        # Get object-specific guidance
        object_characteristics = self._get_object_characteristics(target_category)
        common_confusions = self._get_common_confusions(target_category)

        # Build image description
        if num_images == 1:
            image_desc = "this image"
        else:
            image_desc = f"these {num_images} images from different angles"

        # NOTE: Intentionally NOT including detector confidence to avoid biasing VLM judgment
        # VLM should make independent decisions based solely on visual evidence

        # Check if this category needs stricter verification (supports both HM3D and MP3D)
        cat_lower = target_category.lower()
        is_high_fp_category = cat_lower in self.HIGH_FP_CATEGORIES or cat_lower in self.MP3D_HIGH_FP_CATEGORIES

        # Build strictness note for high false-positive categories
        strictness_note = ""
        if is_high_fp_category:
            strictness_note = f"""
**Note**: "{display_category}" is often confused with similar objects. Please verify carefully.
"""

        # Build the complete prompt - emphasize independent judgment
        prompt = f"""A robot is navigating indoors to find a "{display_category}". Examine {image_desc} and determine if the target object is present.

**IMPORTANT**: Make your judgment based ONLY on what you see in the images. The detector that triggered this verification may have made a false positive detection - do NOT assume the target is present just because verification was triggered.
{strictness_note}
**Target**: {display_category} - {object_characteristics}
**Common confusions**: {common_confusions}

**Rules**:
1. **Independent Judgment**: Judge purely from visual evidence. If you see a bounding box in any image, focus on the object INSIDE the box, but verify independently whether it matches the target category - the detection may be a false positive.
2. **Reachability**: REJECT if the object is outside a window, a reflection in mirror, or behind glass barrier.
3. **Multiple Objects**: There may be multiple "{display_category}" objects in the scene - CONFIRM if ANY ONE of them is clearly the target. The bounding box may only highlight one, but check the entire image.
4. **Partial/Occluded View**: The object may be partially visible due to occlusion or camera angle. CONFIRM if visible features are sufficient to identify it. If too little is visible to judge, choose UNCERTAIN.
5. **When Uncertain**: If the visual evidence is ambiguous, unclear, or you cannot confidently identify the object, choose UNCERTAIN. Do NOT guess.

Reply:
DECISION: CONFIRM / UNCERTAIN / REJECT
CONFIDENCE: 0.0-1.0
REASON: Brief explanation based on visual features you observed"""

        return prompt

    def _build_confidence_context(self, object_map_info: Optional[Dict]) -> str:
        """
        Build confidence context string from object map info.

        Args:
            object_map_info: Dict containing:
                - target_fused_confidence: float
                - target_observation_count: int
                - similar_objects_info: str (format: "label:conf:count,...")
                - current_threshold: float
        """
        if not object_map_info:
            return ""

        lines = ["\n**Detection Context** (from robot's object map):"]

        # Target object confidence
        target_conf = object_map_info.get('target_fused_confidence', 0.0)
        obs_count = object_map_info.get('target_observation_count', 0)
        threshold = object_map_info.get('current_threshold', 0.0)

        if target_conf > 0:
            conf_level = "HIGH" if target_conf >= 0.7 else ("MEDIUM" if target_conf >= 0.4 else "LOW")
            lines.append(f"- Target confidence: {target_conf:.2f} ({conf_level}), observed {obs_count} times, threshold={threshold:.2f}")

        # Similar objects info
        similar_info = object_map_info.get('similar_objects_info', '')
        if similar_info:
            lines.append(f"- Similar objects also detected: {similar_info}")
            lines.append("  (If similar object has higher confidence, be more careful to distinguish)")

        if len(lines) > 1:
            return "\n".join(lines)
        return ""

    def _get_object_characteristics(self, category: str) -> str:
        """
        Get visual characteristics to identify the target object.

        Supports both HM3D (6 categories) and MP3D (21 categories) datasets.
        """
        characteristics = {
            # === Common to HM3D and MP3D ===
            "toilet": "Porcelain bowl with seat/lid, water tank behind. Usually in bathroom.",
            "sink": "Basin with faucet for washing hands/dishes.",
            "bathtub": "Large basin for bathing/soaking, usually in bathroom.",
            "shower": "Shower head with enclosure or curtain, standing bathing area.",
            "bed": "Mattress for sleeping, usually with bedding/sheets/pillows.",
            "couch": "Upholstered multi-seat furniture (2-3+ persons) with cushions and backrest.",
            "sofa": "Same as couch. Upholstered multi-seat furniture with cushions.",
            "chair": "Single-seat furniture for ONE person. Includes dining chair, armchair, office chair.",
            "tv": "Screen device mounted or on stand. IGNORE screen content, judge by hardware shape only.",
            "table": "Flat horizontal surface supported by legs. Includes dining table, coffee table, desk.",

            # === MP3D specific objects ===
            # Large objects (tau_high=0.45, tau_low=0.30)
            "fireplace": "Heating structure with chimney, usually built into wall. May have mantel.",
            "gym equipment": "Exercise machines or fitness equipment. Includes treadmill, weights, exercise bike.",

            # Medium-large objects (tau_high=0.42, tau_low=0.27)
            "counter": "Flat work surface, usually in kitchen or bathroom. Higher than table, often with cabinets below.",
            "cabinet": "Storage furniture with doors/drawers. May be wall-mounted or freestanding.",
            "seating": "Any furniture for sitting. Includes chairs, benches, stools, couches.",

            # Medium objects (tau_high=0.43, tau_low=0.28)
            "nightstand": "Small bedside table, usually next to bed. May have drawer or shelf.",
            "stool": "Backless seat, often tall (bar stool) or short (step stool).",
            "dining table": "Table designed for eating meals, usually with chairs around it.",
            "coffee table": "Low table in living room, usually in front of couch.",
            "desk": "Work surface, may have drawers. Used for writing/computer work.",

            # Small objects (tau_high=0.40, tau_low=0.25)
            "potted plant": "ANY plant or plant-like decoration. Includes live plants, dried flowers, artificial plants, decorative branches in vase. Be lenient!",
            "plant": "Same as potted plant. ANY plant or vegetation. Be lenient!",
            "pillow": "Soft cushion for comfort. Usually on bed, couch, or chair.",
            "towel": "Fabric for drying. Usually in bathroom or kitchen.",
            "clothes": "Garments/clothing items. May be hanging, folded, or scattered.",

            # Fine objects (tau_high=0.42, tau_low=0.27)
            "picture": "Framed artwork or photograph on wall. May be painting, print, or photo.",
            "framed photograph": "Same as picture. Framed image on wall.",

            # === Other HM3D objects ===
            "refrigerator": "Tall cooling cabinet with door handles.",
            "oven": "Heating compartment with door for cooking.",
            "microwave": "Small box with front window for heating food.",
            "stove": "Cooking surface with burners.",
            "lamp": "ANY light fixture. Be lenient - CONFIRM if you see a light source or lamp shade.",
            "mirror": "Reflective surface. Be lenient - CONFIRM if you see any mirror or reflective glass.",
            "vase": "Container for flowers. Be lenient - CONFIRM if you see any vase-like object.",
            "clock": "Time display device. Be lenient - CONFIRM if you see any clock.",
            "book": "Reading material. Be lenient - CONFIRM if you see any books.",
        }
        # Normalize category name
        normalized_category = category.lower().replace("potted ", "").replace("framed ", "")
        return characteristics.get(normalized_category, characteristics.get(category.lower(), f"A {category}."))

    def _get_common_confusions(self, category: str) -> str:
        """
        Get similar objects that may be confused with target - help VLM distinguish.

        Based on llm_answer_hm3d.txt and llm_answer_mp3d.txt confusion matrices.

        HM3D categories (6):
        - couch: ['chair', 'bed', 'bench', 'dining table']
        - tv: ['laptop', 'picture frame', 'window', 'couch']
        - chair: ['couch', 'toilet', 'potted plant', 'bench', 'dining table']
        - toilet: ['chair', 'bench', 'sink', 'potted plant']
        - bed: ['couch', 'dining table', 'bench']
        - potted plant: ['lamp', 'teddy bear', 'toilet', 'dining table']

        MP3D categories (21) - from llm_answer_mp3d.txt:
        - couch: ['bed', 'bench', 'chair', 'dining table', 0.34]
        - tv: ['laptop', 'monitor', 'picture frame', 'window', 0.27]
        - chair: ['couch', 'toilet', 'stool', 'potted plant', 0.30]
        - toilet: ['chair', 'bench', 'potted plant', 'sink', 0.27]
        - bed: ['couch', 'dining table', 'bench', 0.35]
        - potted plant: ['lamp', 'teddy bear', 'toilet', 0.25]
        - cabinet: ['refrigerator', 'door', 0.32]
        - table/dining table/coffee table/desk: ['bed', 'couch', 'bench', 'counter', 0.30]
        - pillow: ['suitcase', 'backpack', 'handbag', 'teddy bear', 0.25]
        - counter: ['dining table', 'bed', 'couch', 'bench', 0.30]
        - sink: ['toilet', 'bathtub', 'washing machine', 'refrigerator', 0.27]
        - picture/framed photograph: ['tv', 'laptop', 'book', 'clock', 0.27]
        - fireplace: ['oven', 'refrigerator', 'tv', 0.30]
        - towel: ['blanket', 'curtain', 'rug', 0.25]
        - seating: ['couch', 'bed', 'potted plant', 'dining table', 0.30]
        - nightstand: ['bench', 'chair', 'stool', 0.32]
        - shower: ['toilet', 'sink', 'refrigerator', 0.25]
        - bathtub: ['bed', 'couch', 'dining table', 'potted plant', 0.30]
        - clothes: ['towel', 'blanket', 'curtain', 0.27]
        - stool: ['chair', 'potted plant', 'toilet', 0.28]
        - gym equipment: ['bench', 'chair', 'sports ball', 'dumbbell', 0.28]

        NOTE: chair, couch, toilet, stool, seating have HIGH false positive rates.
        """
        confusions = {
            # === HM3D and MP3D common categories ===
            "chair": "Often confused with: couch (multi-seat), toilet (has tank), stool (no backrest), potted plant, bench. "
                     "CHAIR is SINGLE-seat furniture for ONE person with backrest. "
                     "Key: vs couch (seats multiple), vs toilet (porcelain bowl + tank), vs stool (no backrest).",

            "couch": "Often confused with: bed (for sleeping), bench (no cushions), chair (single-seat), dining table. "
                     "COUCH is LARGE multi-seat furniture (2-3+ persons) with soft cushions and backrest. "
                     "Key: seats multiple people side by side, has armrests, vs chair (single person).",

            "toilet": "Often confused with: chair (no tank), bench, potted plant (white pot), sink (has faucet). "
                      "TOILET has porcelain bowl with seat/lid and water tank behind. In bathroom. "
                      "Key: has water tank, vs sink (has faucet), vs chair (no porcelain bowl).",

            "tv": "Often confused with: laptop (has keyboard), monitor (similar but with computer), picture frame, window. "
                  "TV is standalone screen device mounted or on stand. IGNORE screen content.",

            "bed": "Often confused with: couch (has armrests), dining table, bench (no mattress). "
                   "BED has mattress with bedding/sheets, designed for sleeping/lying down.",

            "potted plant": "Often confused with: lamp (tall standing), teddy bear, toilet (white). "
                           "Includes live plants, dried flowers, artificial plants. Be lenient!",
            "plant": "Same as potted plant. Often confused with: lamp, teddy bear, toilet. Be lenient!",

            # === MP3D specific categories ===
            "cabinet": "Often confused with: refrigerator (has cooling), door (wall-mounted). "
                       "CABINET is storage furniture with doors/drawers.",

            "table": "Often confused with: bed (horizontal), couch, bench, counter (higher). "
                     "TABLE is flat surface on legs for placing items.",
            "dining table": "Same as table. Often confused with: bed, couch, bench, counter. "
                           "DINING TABLE has chairs around it for eating meals.",
            "coffee table": "Low table in living room. Often confused with: bench, ottoman. "
                           "Usually in front of couch, lower than dining table.",
            "desk": "Work surface. Often confused with: table, counter. May have drawers.",

            "pillow": "Often confused with: suitcase (rectangular), backpack, handbag, teddy bear. "
                      "PILLOW is soft cushion, usually on bed or couch. Be lenient!",

            "counter": "Often confused with: dining table, bed, couch, bench. "
                       "COUNTER is work surface, usually higher than table, often in kitchen/bathroom.",

            "sink": "Often confused with: toilet (no faucet), bathtub (larger), washing machine, refrigerator. "
                    "SINK has basin with faucet for washing.",

            "picture": "Often confused with: tv (electronic screen), laptop, book, clock. "
                       "PICTURE is framed artwork/photo on wall. Static image, not electronic.",
            "framed photograph": "Same as picture. Framed image on wall.",

            "fireplace": "Often confused with: oven (cooking), refrigerator (appliance), tv (screen). "
                         "FIREPLACE is heating structure built into wall, may have mantel.",

            "towel": "Often confused with: blanket (larger), curtain (hanging vertically), rug (on floor). "
                     "TOWEL is fabric for drying, usually in bathroom/kitchen. Be lenient!",

            "seating": "Often confused with: couch, bed, potted plant, dining table. "
                       "SEATING is ANY furniture for sitting - includes chair, bench, stool, couch.",

            "nightstand": "Often confused with: bench, chair, stool. "
                          "NIGHTSTAND is small table next to bed, usually with lamp on top.",

            "shower": "Often confused with: toilet (bathroom fixture), sink (water), refrigerator. "
                      "SHOWER has shower head with enclosure/curtain, standing bathing area.",

            "bathtub": "Often confused with: bed (large horizontal), couch, dining table, potted plant. "
                       "BATHTUB is large basin for bathing/soaking, usually in bathroom.",

            "clothes": "Often confused with: towel, blanket, curtain. "
                       "CLOTHES are garments/clothing items. May be hanging, folded, or scattered.",

            "stool": "Often confused with: chair (has backrest), potted plant (round), toilet. "
                     "STOOL is backless seat, often tall (bar stool) or short.",

            "gym equipment": "Often confused with: bench (sitting), chair, sports ball, dumbbell. "
                            "GYM EQUIPMENT includes exercise machines, weights, treadmill, etc.",

            # === Other HM3D objects ===
            "sofa": "Same as couch. Multi-seat upholstered furniture.",
            "refrigerator": "Often confused with: cabinet, wardrobe. Has door handles and cooling.",
            "lamp": "Be lenient! May be confused with plant or vase. CONFIRM if ANY light fixture visible.",
            "mirror": "Be lenient! Check if real mirror or just glass/window.",
            "vase": "Be lenient! CONFIRM if ANY vase-like container visible.",
            "clock": "Be lenient! CONFIRM if ANY time display device visible.",
            "book": "Be lenient! CONFIRM if ANY books/reading material visible.",
        }
        # Normalize category name
        normalized_category = category.lower().replace("potted ", "").replace("framed ", "")
        return confusions.get(normalized_category, confusions.get(category.lower(), "Be lenient - CONFIRM if the object reasonably matches the target category."))

    def _parse_vlm_response(self, response_text: str,
                            target_category: str) -> Tuple[int, float, str]:
        """
        Parse the VLM response to extract 3-level decision, confidence, and reason.

        Returns:
            Tuple of (decision_level, confidence, reason)
            - decision_level: 1 (CONFIRM), 0 (UNCERTAIN), -1 (REJECT)
        """
        response_lower = response_text.lower().strip()

        # Try to extract structured response with 3-level decision
        decision_match = re.search(r'decision:\s*(confirm|uncertain|reject|yes|no)', response_lower)
        confidence_match = re.search(r'confidence:\s*([0-9.]+)', response_lower)
        reason_match = re.search(r'reason:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)

        # Parse decision level
        decision_level = 0  # Default to UNCERTAIN
        if decision_match:
            decision_str = decision_match.group(1)
            if decision_str == 'confirm' or decision_str == 'yes':
                decision_level = 1
            elif decision_str == 'reject' or decision_str == 'no':
                decision_level = -1
            else:  # 'uncertain'
                decision_level = 0
        else:
            # Fallback: look for keywords
            has_confirm = bool(re.search(r'\bconfirm', response_lower))
            has_reject = bool(re.search(r'\breject', response_lower))
            has_uncertain = bool(re.search(r'\buncertain', response_lower))

            if has_confirm and not has_reject:
                decision_level = 1
            elif has_reject and not has_confirm:
                decision_level = -1
            else:
                # Ambiguous or uncertain
                decision_level = 0

        # Parse confidence
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        else:
            # Default confidence based on decision
            if decision_level == 1:
                confidence = 0.8
            elif decision_level == -1:
                confidence = 0.8
            else:
                confidence = 0.5

        # Parse reason
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = response_text[:200]  # Use first 200 chars as reason

        decision_name = {1: "CONFIRM", 0: "UNCERTAIN", -1: "REJECT"}
        rospy.loginfo(f"[VLMApproach] Parsed: decision={decision_name[decision_level]}, conf={confidence:.2f}")

        return decision_level, confidence, reason

    def _image_to_base64_url(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 data URL."""
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"

    def _publish_result(self, object_id: int, decision_level: int,
                        vlm_confidence: float, reason: str,
                        timeout: bool, duration: float = 0.0):
        """
        Publish VLM verification result to C++.

        Args:
            object_id: Target object ID
            decision_level: 1 (CONFIRM), 0 (UNCERTAIN), -1 (REJECT)
            vlm_confidence: VLM's confidence in its decision
            reason: Explanation string
            timeout: Whether the verification timed out
            duration: Time taken for verification
        """
        msg = VLMVerificationResult()
        msg.header.stamp = rospy.Time.now()
        msg.object_id = object_id
        msg.decision_level = decision_level
        msg.vlm_confidence = vlm_confidence
        msg.reason = reason
        msg.timeout = timeout
        msg.duration = duration

        self.result_pub.publish(msg)
        decision_name = {1: "CONFIRM", 0: "UNCERTAIN", -1: "REJECT"}
        rospy.loginfo(
            f"[VLMApproach] Published result: object_id={object_id}, "
            f"decision={decision_name.get(decision_level, 'UNKNOWN')}, "
            f"conf={vlm_confidence:.3f}, timeout={timeout}"
        )

    def _publish_timeout_result(self, object_id: int):
        """Publish a timeout result with UNCERTAIN decision."""
        self._publish_result(
            object_id, 0,  # UNCERTAIN on timeout
            0.0,
            "VLM disabled due to previous timeout",
            True, 0.0
        )

    def _save_debug_info(self, frames: List[CandidateFrame], target_category: str,
                          decision_level: int, confidence: float, reason: str, duration: float):
        """Save debug images and info with 3-level decision support.

        Saves the EXACT images sent to VLM, using the same selection logic as _call_vlm_api:
        1. Top-3 raw images (without detection boxes)
        2. ONE annotated image with highest detection confidence (with bounding boxes)
        3. Fallback to annotated images if no raw images available
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # Map decision level to string for filename
            decision_str = {1: "CONFIRM", 0: "UNCERTAIN", -1: "REJECT"}.get(decision_level, "UNKNOWN")

            # === Use EXACT same logic as _call_vlm_api to select frames ===
            # First try raw images (top-3)
            frames_with_raw = [f for f in frames if f.raw_image is not None]
            frames_with_raw.sort(key=lambda f: f.confidence, reverse=True)
            selected_frames = frames_with_raw[:3]
            use_raw = True

            # Find the frame with highest detection confidence that has annotated image
            frames_with_annotated = [f for f in frames if f.image is not None and f.confidence > 0]
            frames_with_annotated.sort(key=lambda f: f.confidence, reverse=True)
            best_annotated_frame = frames_with_annotated[0] if frames_with_annotated else None

            # Fallback to annotated images if no raw images
            if not selected_frames:
                selected_frames = frames_with_annotated[:4]
                use_raw = False
                best_annotated_frame = None  # Already using annotated images

            # Count high-confidence vs approach frames
            high_conf_count = sum(1 for f in selected_frames if f.confidence > 0)
            approach_count = sum(1 for f in selected_frames if f.confidence == 0)

            # Save raw images (the EXACT ones sent to VLM)
            for i, frame in enumerate(selected_frames):
                # Use raw_image if available, otherwise use annotated image
                if use_raw and frame.raw_image is not None:
                    image_to_save = frame.raw_image
                    img_type = "raw"
                elif frame.image is not None:
                    image_to_save = frame.image
                    img_type = "annotated"
                else:
                    continue

                bgr_image = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
                # Mark approach frames differently in filename
                frame_type = "approach" if frame.confidence == 0 else "detect"
                filename = f"{timestamp}_{target_category}_{decision_str}_{img_type}{i}_{frame_type}_conf{frame.confidence:.2f}.jpg"
                filepath = os.path.join(self.debug_dir, filename)
                cv2.imwrite(filepath, bgr_image)

            # Save the best annotated frame (with detection boxes) - same as sent to VLM
            if best_annotated_frame is not None:
                bgr_annotated = cv2.cvtColor(best_annotated_frame.image, cv2.COLOR_RGB2BGR)
                annotated_filename = f"{timestamp}_{target_category}_{decision_str}_best_detection_conf{best_annotated_frame.confidence:.2f}.jpg"
                annotated_filepath = os.path.join(self.debug_dir, annotated_filename)
                cv2.imwrite(annotated_filepath, bgr_annotated)

            # Save reasoning
            reason_filename = f"{timestamp}_{target_category}_{decision_str}_result.txt"
            reason_filepath = os.path.join(self.debug_dir, reason_filename)
            with open(reason_filepath, 'w', encoding='utf-8') as f:
                f.write(f"Target: {target_category}\n")
                f.write(f"Decision: {decision_str} ({decision_level})\n")
                f.write(f"  1=CONFIRM (boost confidence)\n")
                f.write(f"  0=UNCERTAIN (keep confidence)\n")
                f.write(f" -1=REJECT (reduce confidence)\n")
                f.write(f"VLM Confidence: {confidence:.3f}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"\n")
                f.write(f"=== Images Sent to VLM ===\n")
                total_images = len(selected_frames) + (1 if best_annotated_frame else 0)
                f.write(f"Total images sent: {total_images}\n")
                f.write(f"  - Raw images (no detection boxes): {len(selected_frames)}\n")
                f.write(f"  - Best annotated image (with detection boxes): {'1' if best_annotated_frame else '0'}\n")
                if best_annotated_frame:
                    f.write(f"    Best detection confidence: {best_annotated_frame.confidence:.3f}\n")
                f.write(f"  - High-confidence detection frames: {high_conf_count}\n")
                f.write(f"  - Approach frames (fallback): {approach_count}\n")
                if approach_count > 0 and high_conf_count == 0:
                    f.write(f"  [HALLUCINATION MODE] Using only approach frames - no target detection!\n")
                if not use_raw:
                    f.write(f"  [ANNOTATED FALLBACK] No raw images available, using annotated images with detection boxes.\n")
                    f.write(f"  Note: VLM may see detection boxes and labels in the images.\n")
                f.write(f"\nFrame details (raw images):\n")
                for i, frame in enumerate(selected_frames):
                    frame_type = "APPROACH" if frame.confidence == 0 else "DETECTION"
                    has_raw = "yes" if frame.raw_image is not None else "no"
                    has_annot = "yes" if frame.image is not None else "no"
                    f.write(f"  Frame {i} [{frame_type}]: conf={frame.confidence:.3f}, dist={frame.distance:.2f}m, raw={has_raw}, annotated={has_annot}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"VLM Reason:\n{reason}\n")

        except Exception as e:
            rospy.logwarn(f"[VLMApproach] Failed to save debug info: {e}")

    def episode_reset_callback(self, msg: Empty):
        """Reset state for new episode."""
        # Increment episode ID first to invalidate any in-progress verification
        old_episode_id = self.episode_id
        self.episode_id += 1

        rospy.loginfo(f"[VLMApproach] Episode reset: {old_episode_id} -> {self.episode_id}")

        # Wait for any in-progress verification to complete (with timeout)
        wait_start = time.time()
        max_wait = 2.0  # seconds
        while self.verification_in_progress and (time.time() - wait_start) < max_wait:
            rospy.loginfo("[VLMApproach] Waiting for in-progress verification to complete...")
            time.sleep(0.1)

        if self.verification_in_progress:
            rospy.logwarn("[VLMApproach] Verification still in progress after timeout, forcing reset")
            self.verification_in_progress = False

        with self.verification_lock:
            # Clear all caches
            self.navigation_frames.clear()
            self.global_history.clear()
            self.approach_frames.clear()
            self.current_target_id = -1
            self.current_nav_target_id = -1
            self.approach_target_id = -1

        self.vlm_disabled_this_episode = False
        self.vlm_used_this_episode = False
        self.vlm_verify_count = 0

        rospy.loginfo(f"[VLMApproach] Episode {self.episode_id} reset complete - all caches cleared")

    def run(self):
        """Main run loop."""
        rospy.loginfo("[VLMApproach] Node running, waiting for verification requests...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = VLMApproachVerifier()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[VLMApproach] Node crashed: {e}")
        import traceback
        traceback.print_exc()
