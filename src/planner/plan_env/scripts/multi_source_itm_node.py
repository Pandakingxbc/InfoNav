#!/usr/bin/env python3
"""
Multi-Source ITM Query Node

This node bridges the semantic prompts with BLIP2 ITM service to enable
multi-source semantic value mapping.

Functionality:
1. Subscribes to /semantic_prompts to receive multiple semantic hypotheses
2. Subscribes to /map_ros/depth (or RGB topic) to get current images
3. Queries BLIP2 ITM service for each prompt
4. Publishes ITM scores to /semantic_scores

Author: Claude (Anthropic)
Date: 2025-12-17
"""

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from plan_env.msg import SemanticPrompts, SemanticScores
import sys
import os

# Add ApexNav root to path to find vlm module
# Script is in src/planner/plan_env/scripts/, need to go up 4 levels to ApexNav root
apexnav_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, apexnav_root)

from vlm.itm.blip2itm_client import BLIP2ITMClient


class MultiSourceITMNode:
    """
    Multi-Source ITM Query Node

    Queries BLIP2 ITM service for multiple semantic prompts and publishes
    the aggregated scores for multi-source semantic value mapping.
    """

    def __init__(self):
        rospy.init_node('multi_source_itm_node', anonymous=False)

        # Parameters
        self.blip2_port = rospy.get_param('~blip2_port', 12182)
        self.rgb_topic = rospy.get_param('~rgb_topic', '/map_ros/rgb')
        self.query_rate = rospy.get_param('~query_rate', 5.0)  # Hz
        self.use_cosine = rospy.get_param('~use_cosine', False)  # Use cosine or ITM score

        # Initialize BLIP2 ITM client
        rospy.loginfo(f"[MultiSourceITM] Connecting to BLIP2 ITM at port {self.blip2_port}...")
        try:
            self.itm_client = BLIP2ITMClient(port=self.blip2_port)
            rospy.loginfo("[MultiSourceITM] BLIP2 ITM client initialized")
        except Exception as e:
            rospy.logerr(f"[MultiSourceITM] Failed to initialize BLIP2 ITM client: {e}")
            rospy.logerr("[MultiSourceITM] Make sure BLIP2 ITM server is running on port 12182")
            rospy.signal_shutdown("BLIP2 ITM server not available")
            return

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # State
        self.current_prompts = []
        self.current_target = ""
        self.latest_rgb_image = None
        self.has_prompts = False

        # Subscribers
        self.prompts_sub = rospy.Subscriber(
            '/semantic_prompts',
            SemanticPrompts,
            self.prompts_callback,
            queue_size=1
        )

        self.rgb_sub = rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.rgb_callback,
            queue_size=10
        )

        # Publisher
        self.scores_pub = rospy.Publisher(
            '/semantic_scores',
            SemanticScores,
            queue_size=10
        )

        # Timer for periodic ITM queries
        self.query_timer = rospy.Timer(
            rospy.Duration(1.0 / self.query_rate),
            self.query_timer_callback
        )

        rospy.loginfo("[MultiSourceITM] Node initialized successfully")
        rospy.loginfo(f"[MultiSourceITM] Query rate: {self.query_rate} Hz")
        rospy.loginfo(f"[MultiSourceITM] Using {'cosine similarity' if self.use_cosine else 'ITM score'}")

    def prompts_callback(self, msg):
        """
        Callback for semantic prompts.

        Args:
            msg (SemanticPrompts): Message containing target object and semantic prompts
        """
        self.current_prompts = msg.prompts
        self.current_target = msg.target_object
        self.has_prompts = len(self.current_prompts) > 0

        rospy.loginfo(f"[MultiSourceITM] Received {len(self.current_prompts)} prompts for target: {self.current_target}")
        for i, prompt in enumerate(self.current_prompts):
            rospy.logdebug(f"  [{i}] {prompt}")

    def rgb_callback(self, msg):
        """
        Callback for RGB images.

        Args:
            msg (Image): RGB image message
        """
        try:
            # Convert ROS Image to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB for BLIP2
            self.latest_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"[MultiSourceITM] Failed to convert image: {e}")

    def query_timer_callback(self, event):
        """
        Timer callback to periodically query ITM for all prompts.

        Args:
            event: ROS timer event
        """
        # Check if we have prompts and an image
        if not self.has_prompts:
            rospy.logdebug_throttle(5.0, "[MultiSourceITM] No prompts received yet")
            return

        if self.latest_rgb_image is None:
            rospy.logwarn_throttle(5.0, "[MultiSourceITM] No RGB image received yet")
            return

        # Query ITM for each prompt
        scores = []
        query_success = True

        for i, prompt in enumerate(self.current_prompts):
            try:
                if self.use_cosine:
                    score = self.itm_client.cosine(self.latest_rgb_image, prompt)
                else:
                    score = self.itm_client.itm_score(self.latest_rgb_image, prompt)

                scores.append(score)
                rospy.logdebug(f"[MultiSourceITM] Prompt {i}: {prompt} -> Score: {score:.4f}")

            except Exception as e:
                rospy.logerr(f"[MultiSourceITM] Failed to query ITM for prompt '{prompt}': {e}")
                query_success = False
                break

        # Publish scores if all queries succeeded
        if query_success and len(scores) == len(self.current_prompts):
            msg = SemanticScores()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.target_object = self.current_target
            msg.prompts = self.current_prompts
            msg.scores = scores

            self.scores_pub.publish(msg)

            rospy.loginfo_throttle(2.0,
                f"[MultiSourceITM] Published scores for {len(scores)} prompts "
                f"(avg: {np.mean(scores):.4f}, max: {np.max(scores):.4f})"
            )

    def run(self):
        """
        Main run loop.
        """
        rospy.loginfo("[MultiSourceITM] Node running. Waiting for prompts and images...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = MultiSourceITMNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[MultiSourceITM] Node crashed: {e}")
        import traceback
        traceback.print_exc()
