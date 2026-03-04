#!/usr/bin/env python3
"""
Test script for publishing semantic prompts

This script demonstrates how to publish semantic prompts for multi-source
semantic value mapping.

Usage:
    rosrun plan_env test_semantic_prompts.py

Example prompts for common objects:
    - dining table: ["Is there a dining table ahead?", "Is there a dining room ahead?", "Is there furniture ahead?"]
    - bed: ["Is there a bed ahead?", "Is there a bedroom ahead?", "Is there sleeping furniture ahead?"]
    - sofa: ["Is there a sofa ahead?", "Is there a living room ahead?", "Is there seating furniture ahead?"]
"""

import rospy
from plan_env.msg import SemanticPrompts


def publish_semantic_prompts(target_object, prompts, weights=None):
    """
    Publish semantic prompts for a target object.

    Args:
        target_object (str): Target object name
        prompts (list): List of semantic prompts
        weights (list, optional): List of weights for each prompt
    """
    pub = rospy.Publisher('/semantic_prompts', SemanticPrompts, queue_size=1)

    # Wait for subscribers
    rospy.sleep(0.5)

    msg = SemanticPrompts()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.target_object = target_object
    msg.prompts = prompts

    if weights is None:
        # Default equal weights
        msg.weights = [1.0] * len(prompts)
    else:
        msg.weights = weights

    rospy.loginfo(f"Publishing semantic prompts for target: {target_object}")
    for i, prompt in enumerate(prompts):
        weight = msg.weights[i] if i < len(msg.weights) else 1.0
        rospy.loginfo(f"  [{i}] {prompt} (weight: {weight:.2f})")

    pub.publish(msg)
    rospy.loginfo("Semantic prompts published successfully!")


def main():
    rospy.init_node('test_semantic_prompts', anonymous=True)

    # Example 1: Dining table with multiple semantic hypotheses
    rospy.loginfo("=" * 60)
    rospy.loginfo("Example 1: Searching for dining table")
    rospy.loginfo("=" * 60)

    publish_semantic_prompts(
        target_object="dining table",
        prompts=[
            "Is there a dining table ahead?",
            "Is there a dining room ahead?",
            "Is there furniture ahead?"
        ],
        weights=[1.0, 0.9, 0.7]  # Decreasing weights for broader hypotheses
    )

    rospy.sleep(2.0)

    # Example 2: Bed
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("Example 2: Searching for bed")
    rospy.loginfo("=" * 60)

    publish_semantic_prompts(
        target_object="bed",
        prompts=[
            "Is there a bed ahead?",
            "Is there a bedroom ahead?",
            "Is there sleeping furniture ahead?"
        ],
        weights=[1.0, 0.85, 0.6]
    )

    rospy.sleep(2.0)

    # Example 3: Kitchen
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("Example 3: Searching for kitchen")
    rospy.loginfo("=" * 60)

    publish_semantic_prompts(
        target_object="kitchen",
        prompts=[
            "Is there a kitchen ahead?",
            "Is there a stove ahead?",
            "Is there a refrigerator ahead?",
            "Is there cooking equipment ahead?"
        ],
        weights=[1.0, 0.9, 0.9, 0.7]
    )

    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo("All examples published. Check /semantic_scores for results.")
    rospy.loginfo("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
