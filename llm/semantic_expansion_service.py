#!/usr/bin/env python3
"""
ROS Service for Semantic Prompt Expansion

This node provides a ROS service that expands navigation targets into
multiple semantic hypotheses using LLM reasoning.

Service: /semantic_expansion
Request: target_object (string)
Response: prompts (string[]), weights (float64[])

Author: Zager-Zhang
"""

import rospy
from std_srvs.srv import Trigger, TriggerResponse
import sys
import os

# Add llm module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm.utils.semantic_prompt_expansion import (
    get_semantic_prompts,
    get_default_semantic_prompts
)
from llm.utils.http_llm_client import HTTPLLMClient


class SemanticExpansionService:
    """ROS service wrapper for semantic prompt expansion."""

    def __init__(self):
        rospy.init_node('semantic_expansion_service', anonymous=False)

        # Load parameters
        self.use_llm = rospy.get_param('~use_llm', True)
        self.llm_url = rospy.get_param('~llm_url', 'http://localhost:20006')
        self.llm_client = None

        # Initialize LLM client if enabled
        if self.use_llm:
            try:
                # Use HTTP LLM client
                self.llm_client = HTTPLLMClient(base_url=self.llm_url)

                # Test connection with a simple query
                test_response = self.llm_client.get_answer("Hello")
                if test_response:
                    rospy.loginfo(f"[SemanticExpansion] Initialized with LLM at {self.llm_url}")
                else:
                    rospy.logwarn(f"[SemanticExpansion] LLM at {self.llm_url} not responding, using default prompts")
                    self.use_llm = False
            except Exception as e:
                rospy.logwarn(f"[SemanticExpansion] Failed to connect to LLM at {self.llm_url}: {e}")
                rospy.logwarn("[SemanticExpansion] Falling back to default prompts")
                self.use_llm = False
        else:
            rospy.loginfo("[SemanticExpansion] LLM disabled, using default prompts")

        # Cache for semantic expansions
        self.cache = {}

        rospy.loginfo("[SemanticExpansion] Service ready")

    def handle_request(self, target_object: str):
        """
        Handle semantic expansion request.

        Args:
            target_object: Navigation target object name

        Returns:
            Tuple of (prompts, weights)
        """
        # Check cache
        if target_object in self.cache:
            rospy.loginfo(f"[SemanticExpansion] Using cached expansion for '{target_object}'")
            return self.cache[target_object]

        # Generate semantic prompts
        if self.use_llm and self.llm_client is not None:
            prompts_and_weights = get_semantic_prompts(target_object, self.llm_client)
        else:
            prompts_and_weights = get_default_semantic_prompts(target_object)

        # Separate into lists
        if prompts_and_weights:
            prompts = [p for p, w in prompts_and_weights]
            weights = [w for p, w in prompts_and_weights]
        else:
            # Fallback: just use the target object itself
            rospy.logwarn(f"[SemanticExpansion] No prompts generated, using fallback")
            prompts = [f"Is there a {target_object} ahead?"]
            weights = [1.0]

        # Cache result
        self.cache[target_object] = (prompts, weights)

        return prompts, weights

    def publish_to_topic(self, target_object: str):
        """
        Publish semantic prompts to ROS topics for C++ nodes to consume.

        This is an alternative to using a service - we publish the results
        to topics that map_ros can subscribe to.
        """
        prompts, weights = self.handle_request(target_object)

        # TODO: Publish to appropriate ROS topics
        # This would require defining custom message types
        rospy.loginfo(f"[SemanticExpansion] Would publish {len(prompts)} prompts for '{target_object}'")
        for i, (prompt, weight) in enumerate(zip(prompts, weights)):
            rospy.loginfo(f"  [{i}] weight={weight:.2f}: {prompt}")


def main():
    """Main entry point."""
    service = SemanticExpansionService()

    # For now, just test with some example objects
    # In production, this would be triggered by navigation requests
    test_objects = rospy.get_param('~test_objects', [])

    if test_objects:
        rospy.loginfo(f"[SemanticExpansion] Testing with objects: {test_objects}")
        for obj in test_objects:
            service.publish_to_topic(obj)

    # Keep node running
    rospy.loginfo("[SemanticExpansion] Service running. Press Ctrl+C to exit.")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
