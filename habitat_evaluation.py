"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zhi Yang -modified from ApexNav 
"""

# Standard library imports
import argparse
import gc
import gzip
import json
import os
import signal
import subprocess
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64, Empty
import tqdm

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
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence, SemanticScores, IGScore, VLMVerificationResult

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from basic_utils.logging import get_log_manager
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_object_utils import get_object
from pathlib import Path

# VLM (BLIP2) imports - will be loaded conditionally based on --disable-vlm flag
# These are set in main() after parsing arguments
get_multi_source_cosine_with_ig = None
VLM_DISABLED = False

# Initialize logging manager
log_manager = get_log_manager()

# Initialize global variables at module level
msg_observations = None
global_action = None
ros_state = None
# NOTE: fusion_threshold 已移除，现在在 src/exploration_manager 中配置
# 使用固定的默认检测置信度阈值
DEFAULT_DETECTION_CONFIDENCE = 0.5
ros_pub = None
trigger_pub = None
obj_point_cloud_pub = None
confidence_threshold_pub = None
final_state = 0
expl_result = 0

# VLM verification tracking (for statistics)
vlm_verify_count_this_episode = 0


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def load_llm_hypotheses(env_base_dir, dataset, episode_id):
    """
    加载LLM假说分析结果

    Args:
        env_base_dir: 环境数据根目录 (如 "env")
        dataset: 数据集名称 (如 "hm3dv2")
        episode_id: Episode ID

    Returns:
        List[dict]: 假说列表，如果文件不存在则返回空列表
    """
    hypothesis_file = Path(env_base_dir) / f"env_{dataset}" / f"task{episode_id}" / "llm_hypothesis_analysis.json"

    if hypothesis_file.exists():
        try:
            with open(hypothesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            hypotheses = data.get('semantic_hypotheses', [])
            print(f"[LLM Hypothesis] Loaded {len(hypotheses)} hypotheses from {hypothesis_file}")
            return hypotheses

        except Exception as e:
            print(f"[LLM Hypothesis] Failed to load {hypothesis_file}: {e}")
            return []
    else:
        print(f"[LLM Hypothesis] File not found: {hypothesis_file}, using fallback")
        return []


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, DEFAULT_DETECTION_CONFIDENCE)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def vlm_verification_result_callback(msg):
    """Callback to count VLM verification calls per episode."""
    global vlm_verify_count_this_episode
    vlm_verify_count_this_episode += 1


def _write_evaluation_stats(episode_stats_list, video_output_path):
    """
    Write evaluation statistics to a summary file in the debug directory.

    Args:
        episode_stats_list: List of episode statistics dictionaries
        video_output_path: Path to video output directory (used to determine debug path)
    """
    # Create debug directory if it doesn't exist
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Generate timestamp for filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(debug_dir, f"evaluation_stats_{timestamp}.txt")

    # Calculate summary statistics
    total_episodes = len(episode_stats_list)
    total_success = sum(1 for e in episode_stats_list if e["success"] == 1)
    total_steps = sum(e["steps"] for e in episode_stats_list)
    total_time = sum(e["time_seconds"] for e in episode_stats_list)
    total_vlm_calls = sum(e["vlm_calls"] for e in episode_stats_list)
    total_spl = sum(e["spl"] for e in episode_stats_list)

    avg_steps = total_steps / total_episodes if total_episodes > 0 else 0
    avg_time = total_time / total_episodes if total_episodes > 0 else 0
    avg_vlm_calls = total_vlm_calls / total_episodes if total_episodes > 0 else 0
    success_rate = total_success / total_episodes * 100 if total_episodes > 0 else 0
    avg_spl = total_spl / total_episodes * 100 if total_episodes > 0 else 0

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("                    EVALUATION STATISTICS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write("\n")

        f.write("-" * 40 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Success Rate:        {success_rate:.2f}% ({total_success}/{total_episodes})\n")
        f.write(f"Average SPL:         {avg_spl:.2f}%\n")
        f.write(f"Average Steps:       {avg_steps:.1f}\n")
        f.write(f"Average Time:        {avg_time:.1f}s\n")
        f.write(f"Average VLM Calls:   {avg_vlm_calls:.2f}\n")
        f.write(f"Total VLM Calls:     {total_vlm_calls}\n")
        f.write("\n")

        f.write("-" * 40 + "\n")
        f.write("PER-EPISODE DETAILS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'No.':<6}{'EpiID':<10}{'Target':<15}{'Steps':<8}{'Time(s)':<10}{'VLM':<6}{'Success':<8}{'Result':<20}\n")
        f.write("-" * 80 + "\n")

        for stats in episode_stats_list:
            success_str = "Yes" if stats["success"] == 1 else "No"
            f.write(f"{stats['episode_num']:<6}{stats['episode_id']:<10}{stats['target']:<15}"
                    f"{stats['steps']:<8}{stats['time_seconds']:<10.1f}{stats['vlm_calls']:<6}"
                    f"{success_str:<8}{stats['result']:<20}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[Stats] Evaluation statistics written to: {stats_file}")
    return stats_file


def _parse_dataset_arg():
    """Parse CLI to choose dataset, detection mode, and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    parser.add_argument(
        "--detector-mode",
        type=int,
        choices=[1, 2],
        default=None,
        help="Detection mode: 1=YOLOv7+GroundingDINO (ApexNav), 2=D-FINE+GroundingDINO (InfoNav). "
             "If not specified, uses config file setting.",
    )
    parser.add_argument(
        "--disable-vlm",
        action="store_true",
        default=False,
        help="Disable VLM (BLIP2) inference for ablation experiments. "
             "When enabled, skips semantic scoring and IG estimation. "
             "Use this with exploration/policy=0 (distance greedy) for pure frontier exploration baseline.",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, args.detector_mode, args.disable_vlm, unknown


def main(cfg: DictConfig, dataset: str = "hm3dv2", disable_vlm: bool = False) -> None:
    global msg_observations, global_action, ros_state
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result
    global vlm_verify_count_this_episode
    global get_multi_source_cosine_with_ig, VLM_DISABLED

    # Set VLM disabled flag
    VLM_DISABLED = disable_vlm

    # Conditionally import VLM module
    if not disable_vlm:
        from vlm.utils.get_itm_message import get_multi_source_cosine_with_ig as _get_multi_source_cosine_with_ig
        get_multi_source_cosine_with_ig = _get_multi_source_cosine_with_ig
        print("[VLM] BLIP2 ITM client initialized")
    else:
        print("[VLM] DISABLED - Skipping BLIP2 initialization (ablation mode)")
        get_multi_source_cosine_with_ig = None

    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    # ==================== Publish Patience-Aware Navigation Thresholds to ROS ====================
    # This ensures the YAML config (patience_nav/threshold_by_size) is passed to C++ exploration_manager
    # C++ expects parameters at: object/threshold/large_high, object/threshold/large_low, etc.
    if hasattr(cfg, 'patience_nav'):
        patience_cfg = cfg.patience_nav

        # Publish global defaults
        if hasattr(patience_cfg, 'tau_high'):
            rospy.set_param('/object/tau_high', float(patience_cfg.tau_high))
        if hasattr(patience_cfg, 'tau_low'):
            rospy.set_param('/object/tau_low', float(patience_cfg.tau_low))
        if hasattr(patience_cfg, 'T_max'):
            rospy.set_param('/object/T_max', int(patience_cfg.T_max))
        if hasattr(patience_cfg, 'min_detection_confidence'):
            rospy.set_param('/object/min_detection_confidence', float(patience_cfg.min_detection_confidence))

        # Publish target-specific thresholds by object size
        if hasattr(patience_cfg, 'threshold_by_size'):
            size_cfg = patience_cfg.threshold_by_size

            # Large objects: bed, couch, bathtub, fireplace
            if hasattr(size_cfg, 'large'):
                rospy.set_param('/object/threshold/large_high', float(size_cfg.large.tau_high))
                rospy.set_param('/object/threshold/large_low', float(size_cfg.large.tau_low))

            # Medium-large objects: tv, counter, cabinet, shower
            if hasattr(size_cfg, 'medium_large'):
                rospy.set_param('/object/threshold/medium_large_high', float(size_cfg.medium_large.tau_high))
                rospy.set_param('/object/threshold/medium_large_low', float(size_cfg.medium_large.tau_low))

            # Medium objects: chair, toilet, sink, table
            if hasattr(size_cfg, 'medium'):
                rospy.set_param('/object/threshold/medium_high', float(size_cfg.medium.tau_high))
                rospy.set_param('/object/threshold/medium_low', float(size_cfg.medium.tau_low))

            # Small objects: plant, pillow, towel
            if hasattr(size_cfg, 'small'):
                rospy.set_param('/object/threshold/small_high', float(size_cfg.small.tau_high))
                rospy.set_param('/object/threshold/small_low', float(size_cfg.small.tau_low))

            # Fine objects: picture
            if hasattr(size_cfg, 'fine'):
                rospy.set_param('/object/threshold/fine_high', float(size_cfg.fine.tau_high))
                rospy.set_param('/object/threshold/fine_low', float(size_cfg.fine.tau_low))

        print("[Patience-Aware Nav] Published threshold configs to ROS parameter server:")
        print(f"  Global: tau_high={patience_cfg.get('tau_high', 'N/A')}, tau_low={patience_cfg.get('tau_low', 'N/A')}, T_max={patience_cfg.get('T_max', 'N/A')}")
        if hasattr(patience_cfg, 'threshold_by_size'):
            size_cfg = patience_cfg.threshold_by_size
            if hasattr(size_cfg, 'large'):
                print(f"  Large:        tau_high={size_cfg.large.tau_high}, tau_low={size_cfg.large.tau_low}")
            if hasattr(size_cfg, 'medium_large'):
                print(f"  Medium-Large: tau_high={size_cfg.medium_large.tau_high}, tau_low={size_cfg.medium_large.tau_low}")
            if hasattr(size_cfg, 'medium'):
                print(f"  Medium:       tau_high={size_cfg.medium.tau_high}, tau_low={size_cfg.medium.tau_low}")
            if hasattr(size_cfg, 'small'):
                print(f"  Small:        tau_high={size_cfg.small.tau_high}, tau_low={size_cfg.small.tau_low}")
            if hasattr(size_cfg, 'fine'):
                print(f"  Fine:         tau_high={size_cfg.fine.tau_high}, tau_low={size_cfg.fine.tau_low}")

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run (deprecated)
    test_task_num = cfg.get('test_task_num', -1)  # Which task index to test
    flag_once = (env_num_once != -1) or (test_task_num != -1)  # Whether to run single test

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

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

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    # If testing specific task, override num_total with test_task_num
    if test_task_num != -1:
        num_total = test_task_num
        num_success = 0
        spl_all = 0.0
        soft_spl_all = 0.0
        distance_to_goal_all = 0.0
        distance_to_goal_reward_all = 0.0
        last_time = 0.0
        print(f"[Test Mode] Testing specific task index: {test_task_num}")
    else:
        (
            num_total,
            num_success,
            spl_all,
            soft_spl_all,
            distance_to_goal_all,
            distance_to_goal_reward_all,
            last_time,
        ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    # Determine starting position
    if test_task_num != -1:
        env_count = test_task_num  # Start from specific task index
    elif env_num_once != -1:
        env_count = env_num_once  # Old behavior for backward compatibility
    else:
        env_count = num_total  # Continue from last completed task

    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    # Episode reset signal - clears all maps and Value Maps on ROS side
    episode_reset_pub = rospy.Publisher("/habitat/episode_reset", Empty, queue_size=10)
    # Multi-source semantic scores publisher
    semantic_scores_pub = rospy.Publisher("/habitat/semantic_scores", SemanticScores, queue_size=10)
    # Information Gain score publisher (for exploration guidance)
    ig_score_pub = rospy.Publisher("/habitat/ig_score", IGScore, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    # Subscribe to VLM verification results for statistics tracking
    rospy.Subscriber("/vlm/verification_result", VLMVerificationResult,
                     vlm_verification_result_callback, queue_size=10)

    # Statistics tracking for evaluation report
    # Each entry: {episode_id, target, steps, time, vlm_calls, success}
    episode_stats_list = []
    stats_written = False  # Flag to track if stats have been written

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        # Reset VLM verification counter for this episode
        vlm_verify_count_this_episode = 0
        # Record episode start time
        episode_start_time = time.time()

        camera_pitch = 0.0
        observations = env.reset()

        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]

        # Notify ROS to clear all maps and Value Maps for new episode
        episode_reset_pub.publish(Empty())
        print("[Episode Reset] Published reset signal to ROS, waiting for ROS to reset...")

        # Immediately publish the new episode's initial odometry and confidence
        # This ensures ROS nodes have odom data after reset clears have_odom_ flag
        for _ in range(5):  # Publish multiple times to ensure delivery
            ros_pub.habitat_publish_ros_topic(msg_observations)
            publish_float64(confidence_threshold_pub, 0.5)
            rospy.sleep(0.1)

        rospy.sleep(0.5)  # Give ROS nodes time to process reset and reinitialize

        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer for object detection (移除了 room 和 fusion_threshold)
        # NOTE: fusion_threshold 现在在 src/exploration_manager 中配置
        llm_answer = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Load LLM semantic hypotheses for HSVM (Hierarchical Semantic Value Map)
        # HSVM 已经包含了多层级假说: room_type, target_object, co_occurrence, part_attribute
        episode_id = env.current_episode.episode_id
        task_index = num_total  # Use task index instead of episode_id for hypothesis loading
        current_hypotheses = load_llm_hypotheses("env", dataset, task_index)
        print(f"[Task {task_index}, Episode {episode_id}] Target: {label}, Hypotheses: {len(current_hypotheses)}")

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop the initialization timer
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        # Create a new timer for odometry and confidence publishing (not trigger)
        # This keeps ROS node alive by providing continuous odometry updates
        # Reduced frequency to 5Hz to decrease CPU load
        def publish_odom_and_confidence(event):
            ros_pub.habitat_publish_ros_topic(deepcopy(msg_observations))
            publish_float64(confidence_threshold_pub, DEFAULT_DETECTION_CONFIDENCE)

        odom_timer = rospy.Timer(rospy.Duration(0.2), publish_odom_and_confidence)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # ========== PERFORMANCE PROFILING START ==========
            import time as time_module
            step_start_time = time_module.time()
            timing_breakdown = {}

            # Notify ROS system that action execution is starting
            t0 = time_module.time()
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)
            timing_breakdown['ros_publish_start'] = time_module.time() - t0

            # Habitat simulation step
            t0 = time_module.time()
            observations = env.step(action)
            timing_breakdown['habitat_step'] = time_module.time() - t0

            # Calculate multi-source ITM cosine similarity scores + IG score
            # Using combined function for efficiency (single batch inference)
            # NOTE: 移除了 room 参数，HSVM 已包含多层级假说
            t0 = time_module.time()

            if not VLM_DISABLED:
                # Normal mode: run VLM inference
                hypotheses_data, ig_data = get_multi_source_cosine_with_ig(
                    observations["rgb"], label, current_hypotheses
                )
                timing_breakdown['vlm_inference'] = time_module.time() - t0
                timing_breakdown['vlm_num_prompts'] = len(hypotheses_data) + 3  # +3 for IG prompts

                # Build and publish SemanticScores message
                t0 = time_module.time()
                semantic_scores_msg = SemanticScores()
                semantic_scores_msg.header.stamp = rospy.Time.now()
                semantic_scores_msg.target_object = label

                # Extract data from hypotheses_data
                semantic_scores_msg.hypothesis_ids = [h["id"] for h in hypotheses_data]
                semantic_scores_msg.hypothesis_types = [h["type"] for h in hypotheses_data]
                semantic_scores_msg.prompts = [h["prompt"] for h in hypotheses_data]
                semantic_scores_msg.confidences = [h["confidence"] for h in hypotheses_data]
                semantic_scores_msg.navigation_values = [h["navigation_value"] for h in hypotheses_data]
                semantic_scores_msg.weights = [h["weight"] for h in hypotheses_data]
                semantic_scores_msg.scores = [h["score"] for h in hypotheses_data]

                # Publish multi-source semantic scores
                semantic_scores_pub.publish(semantic_scores_msg)

                # Build and publish IG score message
                ig_score_msg = IGScore()
                ig_score_msg.header.stamp = rospy.Time.now()
                ig_score_msg.ig_score = ig_data["ig_score"]
                ig_score_msg.corridor_score = ig_data["corridor_score"]
                ig_score_msg.doorway_score = ig_data["doorway_score"]
                ig_score_msg.passage_score = ig_data["passage_score"]
                ig_score_pub.publish(ig_score_msg)
                timing_breakdown['semantic_msg_build'] = time_module.time() - t0

                # Print summary (controlled by logging config)
                if log_manager.should_log_semantic_scores():
                    print(f"Multi-source semantic scores:")
                    for i, h in enumerate(hypotheses_data):
                        print(f"  [{i}] {h['type']:15s} | Score: {h['score']:.3f} | Weight: {h['weight']:.3f} | {h['prompt'][:50]}")
                    print(f"Information Gain (IG) score: {ig_data['ig_score']:.3f} "
                          f"[corridor: {ig_data['corridor_score']:.3f}, "
                          f"doorway: {ig_data['doorway_score']:.3f}, "
                          f"passage: {ig_data['passage_score']:.3f}]")
            else:
                # Ablation mode: skip VLM inference entirely
                timing_breakdown['vlm_inference'] = 0.0
                timing_breakdown['vlm_num_prompts'] = 0

                # Still need to publish SemanticScores with target_object so C++ knows the target category
                # This is required for ObjectMap2D::setTargetCategory() and VLM approach verifier
                t0 = time_module.time()
                semantic_scores_msg = SemanticScores()
                semantic_scores_msg.header.stamp = rospy.Time.now()
                semantic_scores_msg.target_object = label
                # Empty lists for other fields (no VLM inference)
                semantic_scores_msg.hypothesis_ids = []
                semantic_scores_msg.hypothesis_types = []
                semantic_scores_msg.prompts = []
                semantic_scores_msg.confidences = []
                semantic_scores_msg.navigation_values = []
                semantic_scores_msg.weights = []
                semantic_scores_msg.scores = []
                semantic_scores_pub.publish(semantic_scores_msg)
                timing_breakdown['semantic_msg_build'] = time_module.time() - t0
                # No IG scores published in ablation mode

            # Save raw RGB image before detection (for VLM verification)
            raw_rgb_image = observations["rgb"].copy()

            # Detect objects in the current observation
            t0 = time_module.time()
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, observations["rgb"], detector_cfg, llm_answer
            )
            timing_breakdown['object_detection'] = time_module.time() - t0
            timing_breakdown['num_detections'] = len(score_list)

            # Publish habitat observations to ROS
            t0 = time_module.time()
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)
            # Publish raw RGB image (without detection boxes) for VLM verification
            ros_pub.publish_rgb_raw(rospy.Time.now(), raw_rgb_image)
            timing_breakdown['ros_habitat_publish'] = time_module.time() - t0

            # Generate and publish object point clouds
            t0 = time_module.time()
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )
            timing_breakdown['point_cloud_gen'] = time_module.time() - t0

            # Publish detection-related information
            t0 = time_module.time()
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)
            timing_breakdown['ros_detection_publish'] = time_module.time() - t0

            # Generate video frame
            t0 = time_module.time()
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)
            timing_breakdown['video_generation'] = time_module.time() - t0

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            t0 = time_module.time()
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            timing_breakdown['ros_publish_finish'] = time_module.time() - t0

            # ROS rate.sleep()
            t0 = time_module.time()
            rate.sleep()
            timing_breakdown['ros_rate_sleep'] = time_module.time() - t0

            # Total step time
            total_step_time = time_module.time() - step_start_time
            timing_breakdown['total_step_time'] = total_step_time

            # Print timing breakdown (controlled by logging config)
            if log_manager.should_print_timing_this_step(count_steps):
                if log_manager.should_log_timing_breakdown():
                    # Detailed breakdown
                    log_manager.log_timing_breakdown(f"\n[TIMING] Step {count_steps} Performance Breakdown:")
                    log_manager.log_timing_breakdown(f"  Total Step Time:      {total_step_time:.3f}s")
                    log_manager.log_timing_breakdown(f"  - Habitat Step:       {timing_breakdown['habitat_step']:.3f}s ({timing_breakdown['habitat_step']/total_step_time*100:.1f}%)")
                    log_manager.log_timing_breakdown(f"  - VLM Inference:      {timing_breakdown['vlm_inference']:.3f}s ({timing_breakdown['vlm_inference']/total_step_time*100:.1f}%) [{timing_breakdown['vlm_num_prompts']} prompts]")
                    log_manager.log_timing_breakdown(f"  - Object Detection:   {timing_breakdown['object_detection']:.3f}s ({timing_breakdown['object_detection']/total_step_time*100:.1f}%) [{timing_breakdown['num_detections']} objs]")
                    log_manager.log_timing_breakdown(f"  - Point Cloud Gen:    {timing_breakdown['point_cloud_gen']:.3f}s ({timing_breakdown['point_cloud_gen']/total_step_time*100:.1f}%)")
                    log_manager.log_timing_breakdown(f"  - Video Generation:   {timing_breakdown['video_generation']:.3f}s ({timing_breakdown['video_generation']/total_step_time*100:.1f}%)")
                    log_manager.log_timing_breakdown(f"  - ROS Publish Total:  {timing_breakdown['ros_publish_start'] + timing_breakdown['semantic_msg_build'] + timing_breakdown['ros_habitat_publish'] + timing_breakdown['ros_detection_publish'] + timing_breakdown['ros_publish_finish']:.3f}s")
                    log_manager.log_timing_breakdown(f"  - ROS Rate Sleep:     {timing_breakdown['ros_rate_sleep']:.3f}s ({timing_breakdown['ros_rate_sleep']/total_step_time*100:.1f}%)")
                elif log_manager.should_log_timing_summary():
                    # Summary only
                    log_manager.log_timing_summary(f"[TIMING] Step {count_steps}: Total {total_step_time:.3f}s | VLM {timing_breakdown['vlm_inference']:.3f}s | Det {timing_breakdown['object_detection']:.3f}s")
            # ========== PERFORMANCE PROFILING END ==========

        # Stop odometry timer at end of episode
        odom_timer.shutdown()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        # Record episode statistics
        episode_time = time.time() - episode_start_time
        episode_stats = {
            "episode_num": num_total,
            "episode_id": episode_id,
            "target": label,
            "steps": count_steps,
            "time_seconds": episode_time,
            "vlm_calls": vlm_verify_count_this_episode,
            "success": success,
            "spl": spl,
            "result": result_text,
        }
        episode_stats_list.append(episode_stats)
        print(f"[Stats] Episode {num_total}: steps={count_steps}, time={episode_time:.1f}s, vlm_calls={vlm_verify_count_this_episode}, success={success}")

        # Write stats at 100 episodes milestone (but don't stop the test)
        if len(episode_stats_list) == 100 and not stats_written:
            print(f"\n[Stats] Reached 100 episodes milestone, generating summary...")
            _write_evaluation_stats(episode_stats_list, video_output_path)
            stats_written = True

        # Force garbage collection every episode to prevent memory buildup
        gc.collect()

        # Print memory usage every 10 episodes
        if (num_total + 1) % 10 == 0:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"\n[Memory Monitor] Episode {num_total + 1}:")
            print(f"  RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
            print(f"  VMS: {mem_info.vms / 1024 / 1024:.1f} MB")
            cpu_percent = process.cpu_percent(interval=0.1)
            print(f"  CPU: {cpu_percent:.1f}%\n")

        # NOTE: Node restart disabled - exploration_node will run continuously
        # If memory issues occur, uncomment the restart logic below
        # if (num_total + 1) % 5 == 0 and not flag_once:
        #     ... (restart logic)

        # Exit loop after saving data if testing single episode
        if flag_once:
            break

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    # Write final stats if not already written (or update with all episodes)
    if episode_stats_list:
        print(f"\n[Stats] Test finished, generating final summary with {len(episode_stats_list)} episodes...")
        _write_evaluation_stats(episode_stats_list, video_output_path)

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, detector_mode, disable_vlm, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)

        # Override detection mode if specified via CLI
        if detector_mode is not None:
            from omegaconf import OmegaConf
            OmegaConf.set_struct(cfg, False)
            cfg.detector.mode = detector_mode
            OmegaConf.set_struct(cfg, True)
            mode_desc = "YOLOv7+GroundingDINO" if detector_mode == 1 else "D-FINE+GroundingDINO"
            print(f"[Detection Mode] Using Mode {detector_mode}: {mode_desc}")

        # Print VLM status
        if disable_vlm:
            print(f"[Ablation Mode] VLM (BLIP2) DISABLED - Pure frontier exploration baseline")

        main(cfg, dataset, disable_vlm)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
