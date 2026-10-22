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
from pathlib import Path

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
import yaml
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

# Real-time analysis imports (for autonomous hypothesis generation)
try:
    import cv2
    from scripts.vlm_environment_analyzer import VLMEnvironmentAnalyzer
    from scripts.llm_hypothesis_analyzer import LLMHypothesisAnalyzer, AnalysisReportWriter
    REALTIME_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Real-time analysis modules not available: {e}")
    REALTIME_ANALYSIS_AVAILABLE = False

# ============================================================
# Configuration Loading
# ============================================================
# Load unified configuration from config/default.yaml
# If file does not exist, use hardcoded defaults (backward compatible)
CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    print(f"[Config] Loaded configuration from {CONFIG_PATH}")
else:
    CONFIG = {}
    print(f"[Config] Configuration file not found at {CONFIG_PATH}, using hardcoded defaults")

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
# 使用固定的默认检测置信度阈值 (从配置文件读取，回退到 0.5)
DEFAULT_DETECTION_CONFIDENCE = CONFIG.get('object', {}).get('default_detection_confidence', 0.5)
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


def load_llm_hypotheses(env_base_dir, dataset, episode_id, max_hierarchy_level=4):
    """
    加载LLM假说分析结果，支持层级过滤

    Args:
        env_base_dir: 环境数据根目录 (如 "env")，可从配置文件读取
        dataset: 数据集名称 (如 "hm3dv2")
        episode_id: Episode ID
        max_hierarchy_level: 最大层级 (1-4)，从 Hydra 配置传入

    Returns:
        List[dict]: 假说列表（已按层级过滤），如果文件不存在则返回空列表
    """
    # Get hypothesis file name from config (with fallback)
    hypothesis_filename = CONFIG.get('paths', {}).get('llm', {}).get('hypothesis_file', 'llm_hypothesis_analysis.json')
    hypothesis_file = Path(env_base_dir) / f"env_{dataset}" / f"task{episode_id}" / hypothesis_filename

    if hypothesis_file.exists():
        try:
            with open(hypothesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            hypotheses = data.get('semantic_hypotheses', [])

            # 使用传入的层级过滤参数
            max_level = max_hierarchy_level

            # 定义层级映射（根据消融实验需求修正）
            # L1: 仅目标物体
            # L2: 目标物体 + 房间类型
            # L3: 目标物体 + 房间类型 + 共现物体
            # L4: 目标物体 + 房间类型 + 共现物体 + 物体部件 (Full)
            level_map = {
                "target_object": 1,      # Level 1: 核心目标
                "room_type": 2,          # Level 2: 空间上下文
                "co_occurrence": 3,      # Level 3: 共现物体线索
                "object_part": 4         # Level 4: 细粒度部件
            }

            # 过滤假设：只保留 level <= max_level 的假设
            filtered_hypotheses = [
                h for h in hypotheses
                if level_map.get(h.get('hypothesis_type', 'unknown'), 999) <= max_level
            ]

            # 打印过滤信息
            if len(filtered_hypotheses) < len(hypotheses):
                print(f"[LLM Hypothesis] Hierarchy filtering: {len(hypotheses)} → {len(filtered_hypotheses)} "
                      f"hypotheses (max_level={max_level})")
                # 显示被过滤掉的类型
                filtered_types = set(h.get('hypothesis_type') for h in hypotheses) - \
                                set(h.get('hypothesis_type') for h in filtered_hypotheses)
                if filtered_types:
                    print(f"[LLM Hypothesis] Filtered out types: {', '.join(filtered_types)}")
            else:
                print(f"[LLM Hypothesis] Loaded {len(hypotheses)} hypotheses from {hypothesis_file} (max_level={max_level}, no filtering)")

            return filtered_hypotheses

        except Exception as e:
            print(f"[LLM Hypothesis] Failed to load {hypothesis_file}: {e}")
            return []
    else:
        print(f"[LLM Hypothesis] File not found: {hypothesis_file}, using fallback")
        return []


# ============================================================
# Real-time Panorama Collection and Hypothesis Generation
# ============================================================
# These functions enable autonomous hypothesis generation at the start
# of each episode, replacing pre-computed hypotheses with real-time analysis.

# Configuration constants for panorama collection
TURN_ANGLE = 30  # Turn angle per action (degrees)
TURNS_PER_90_DEG = 3  # Number of turns needed for 90 degrees (90/30=3)
VIEW_DIRECTIONS = {0: "front", 1: "left", 2: "back", 3: "right"}


def collect_panorama_views_realtime(env, observations) -> list:
    """
    Collect 4 panorama views by rotating the agent in place.

    This function rotates the agent 90° three times to capture views at:
    - View 0: Front (0°)
    - View 1: Left (90°)
    - View 2: Back (180°)
    - View 3: Right (270°)

    Args:
        env: Habitat environment
        observations: Current observations dict

    Returns:
        List of 4 RGB images (numpy arrays)
    """
    views = []

    # Capture front view (current position)
    rgb_image = observations["rgb"].copy()
    views.append(rgb_image)
    print(f"  [Panorama] Captured view 0 (front)")

    # Rotate and capture remaining 3 views
    for view_idx in range(1, 4):
        # Rotate left 90 degrees (3 turns × 30°)
        for _ in range(TURNS_PER_90_DEG):
            observations = env.step(HabitatSimActions.turn_left)

        rgb_image = observations["rgb"].copy()
        views.append(rgb_image)
        direction = VIEW_DIRECTIONS[view_idx]
        print(f"  [Panorama] Captured view {view_idx} ({direction})")

    # Rotate back to original orientation (one more 90° turn)
    for _ in range(TURNS_PER_90_DEG):
        observations = env.step(HabitatSimActions.turn_left)

    return views


def save_panorama_views(views: list, output_dir: Path, task_info: dict) -> bool:
    """
    Save panorama views and task info to disk.

    Args:
        views: List of 4 RGB images
        output_dir: Directory to save views
        task_info: Dictionary containing task_index, episode_id, target_object, scene_id

    Returns:
        True if successful
    """
    if not REALTIME_ANALYSIS_AVAILABLE:
        print("[Error] cv2 not available for saving images")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each view as PNG
    for i, rgb_image in enumerate(views):
        view_path = output_dir / f"view{i}.png"
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(view_path), bgr_image)

    # Save task_info.txt
    info_path = output_dir / "task_info.txt"
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Task Index: {task_info.get('task_index', 'N/A')}\n")
        f.write(f"Episode ID: {task_info.get('episode_id', 'N/A')}\n")
        f.write(f"Target Object: {task_info.get('target_object', 'unknown')}\n")
        f.write(f"Scene ID: {task_info.get('scene_id', 'N/A')}\n")

    print(f"  [Panorama] Saved 4 views and task_info to {output_dir}")
    return True


def run_realtime_vlm_analysis(task_dir: Path, target_object: str,
                               vlm_config: dict) -> tuple:
    """
    Run VLM environment analysis on collected panorama views.

    Args:
        task_dir: Directory containing view*.png files
        target_object: Target object name
        vlm_config: VLM configuration dict with api_url, model, api_type, timeout

    Returns:
        (success, vlm_content, error_msg)
    """
    if not REALTIME_ANALYSIS_AVAILABLE:
        return False, None, "VLMEnvironmentAnalyzer not available"

    try:
        analyzer = VLMEnvironmentAnalyzer(
            api_url=vlm_config.get('api_url', 'http://localhost:20004/api/chat'),
            model=vlm_config.get('model', 'qwen3-vl:32b'),
            env_base_dir=str(task_dir.parent.parent),  # env directory
            timeout=vlm_config.get('timeout', 300),
            api_type=vlm_config.get('api_type', 'ollama')
        )

        # Extract env_name and task_num from task_dir path
        # task_dir format: env/env_{dataset}/task{num}
        env_name = task_dir.parent.name.replace('env_', '')
        task_num = int(task_dir.name.replace('task', ''))

        success, content, duration, error_msg = analyzer.analyze_single(
            env_name, task_num,
            temperature=vlm_config.get('temperature', 0.2)
        )

        if success:
            print(f"  [VLM] Analysis completed in {duration:.2f}s")
            return True, content, None
        else:
            return False, None, error_msg

    except Exception as e:
        return False, None, str(e)


def run_realtime_llm_hypothesis(task_dir: Path, target_object: str,
                                 llm_config: dict, generate_ig: bool = True) -> tuple:
    """
    Run LLM hypothesis generation (HSVM + IG) based on VLM analysis.

    Args:
        task_dir: Directory containing vlm_analysis.txt
        target_object: Target object name
        llm_config: LLM configuration dict with api_url, model, temperature
        generate_ig: Whether to generate IG hypotheses

    Returns:
        (success, hypotheses_list, ig_prompts_list, error_msg)
    """
    if not REALTIME_ANALYSIS_AVAILABLE:
        return False, [], [], "LLMHypothesisAnalyzer not available"

    try:
        analyzer = LLMHypothesisAnalyzer(
            api_url=llm_config.get('api_url', 'https://api.deepseek.com/chat/completions'),
            model=llm_config.get('model', 'deepseek-chat'),
            env_base_dir=str(task_dir.parent.parent)
        )

        # Extract env_name and task_num from task_dir path
        env_name = task_dir.parent.name.replace('env_', '')
        task_num = int(task_dir.name.replace('task', ''))

        # Run analysis with IG generation
        if generate_ig:
            success, analysis, error_msg = analyzer.analyze_single_with_ig(
                env_name, task_num,
                temperature=llm_config.get('temperature', 0.3),
                generate_ig=True
            )
        else:
            success, analysis, error_msg = analyzer.analyze_single(
                env_name, task_num,
                temperature=llm_config.get('temperature', 0.3)
            )

        if success and analysis:
            # Save JSON result
            writer = AnalysisReportWriter()
            json_file = task_dir / "llm_hypothesis_analysis.json"
            writer.write_json(json_file, analysis)
            print(f"  [LLM] Hypotheses saved to {json_file}")

            # Extract hypotheses for return
            hypotheses = [h.to_habitat_dict() for h in analysis.semantic_hypotheses]

            # Extract IG prompts
            ig_prompts = []
            if analysis.ig_hypotheses:
                ig_prompts = [h.prompt for h in analysis.ig_hypotheses if h.prompt]

            print(f"  [LLM] Generated {len(hypotheses)} HSVM hypotheses, {len(ig_prompts)} IG prompts")
            return True, hypotheses, ig_prompts, None
        else:
            return False, [], [], error_msg

    except Exception as e:
        return False, [], [], str(e)


def generate_hypotheses_realtime(env, observations, dataset: str, task_index: int,
                                  target_object: str, episode_id: str, scene_id: str,
                                  env_base_dir: str, vlm_config: dict, llm_config: dict,
                                  skip_existing: bool = True) -> tuple:
    """
    Complete real-time hypothesis generation pipeline:
    1. Collect 4 panorama views by rotating the agent
    2. Run VLM environment analysis
    3. Generate HSVM + IG hypotheses

    Args:
        env: Habitat environment
        observations: Current observations
        dataset: Dataset name (hm3dv2, mp3d, etc.)
        task_index: Task index number
        target_object: Target object name
        episode_id: Episode ID
        scene_id: Scene ID
        env_base_dir: Base directory for env data
        vlm_config: VLM configuration dict
        llm_config: LLM configuration dict
        skip_existing: Skip if hypotheses already exist

    Returns:
        (observations, hypotheses_list, ig_prompts_list, success)
        - observations: Updated observations after rotation
        - hypotheses_list: List of HSVM hypothesis dicts
        - ig_prompts_list: List of IG prompt strings
        - success: Whether generation was successful
    """
    task_dir = Path(env_base_dir) / f"env_{dataset}" / f"task{task_index}"
    json_file = task_dir / "llm_hypothesis_analysis.json"

    # Check if hypotheses already exist
    if skip_existing and json_file.exists():
        print(f"[Real-time Hypothesis] Existing hypotheses found at {json_file}, loading...")
        # Load existing hypotheses
        hypotheses = load_llm_hypotheses(env_base_dir, dataset, task_index, max_hierarchy_level=4)
        ig_prompts = load_ig_hypotheses(env_base_dir, dataset, task_index)
        return observations, hypotheses, ig_prompts, True

    print(f"\n{'='*60}")
    print(f"[Real-time Hypothesis Generation] Task {task_index}")
    print(f"{'='*60}")
    print(f"Target: {target_object}")
    print(f"Scene: {scene_id}")

    # Default fallback values
    DEFAULT_IG_PROMPTS = [
        "This view shows a corridor or hallway leading to other areas",
        "There is a doorway or opening that leads to another room",
        "This passage connects to multiple rooms or spaces"
    ]

    if not REALTIME_ANALYSIS_AVAILABLE:
        print("[Warning] Real-time analysis not available, using defaults")
        return observations, [], DEFAULT_IG_PROMPTS, False

    # Step 1: Collect panorama views
    print(f"\n[Step 1/3] Collecting panorama views...")
    views = collect_panorama_views_realtime(env, observations)

    # Save views
    task_info = {
        'task_index': task_index,
        'episode_id': episode_id,
        'target_object': target_object,
        'scene_id': scene_id
    }
    save_panorama_views(views, task_dir, task_info)

    # Step 2: Run VLM analysis
    print(f"\n[Step 2/3] Running VLM environment analysis...")
    vlm_success, vlm_content, vlm_error = run_realtime_vlm_analysis(
        task_dir, target_object, vlm_config
    )

    if not vlm_success:
        print(f"[Warning] VLM analysis failed: {vlm_error}")
        print("[Warning] Using default IG prompts")
        return observations, [], DEFAULT_IG_PROMPTS, False

    # Step 3: Generate hypotheses
    print(f"\n[Step 3/3] Generating HSVM + IG hypotheses...")
    llm_success, hypotheses, ig_prompts, llm_error = run_realtime_llm_hypothesis(
        task_dir, target_object, llm_config, generate_ig=True
    )

    if not llm_success:
        print(f"[Warning] LLM hypothesis generation failed: {llm_error}")
        print("[Warning] Using default IG prompts")
        return observations, [], DEFAULT_IG_PROMPTS, False

    # Use default IG prompts if none were generated
    if not ig_prompts:
        print("[Warning] No IG prompts generated, using defaults")
        ig_prompts = DEFAULT_IG_PROMPTS

    print(f"\n{'='*60}")
    print(f"[Real-time Hypothesis] Complete!")
    print(f"  HSVM Hypotheses: {len(hypotheses)}")
    print(f"  IG Prompts: {len(ig_prompts)}")
    print(f"{'='*60}\n")

    return observations, hypotheses, ig_prompts, True


def load_ig_hypotheses(env_base_dir, dataset, episode_id):
    """
    加载LLM生成的IG假设（Information Gain假设）

    IG假设是目标无关的，用于估计视图的探索潜力。
    与HSVM假设不同，IG假设描述空间连通性结构（走廊、门洞、通道等）。

    Args:
        env_base_dir: 环境数据根目录 (如 "env")
        dataset: 数据集名称 (如 "hm3dv2")
        episode_id: Episode ID

    Returns:
        List[str]: IG prompts列表，如果文件不存在则返回默认prompts
    """
    # 默认的IG prompts（固定版本，作为fallback）
    DEFAULT_IG_PROMPTS = [
        "This view shows a corridor or hallway leading to other areas",
        "There is a doorway or opening that leads to another room",
        "This passage connects to multiple rooms or spaces"
    ]

    # Get hypothesis file name from config (with fallback)
    hypothesis_filename = CONFIG.get('paths', {}).get('llm', {}).get('hypothesis_file', 'llm_hypothesis_analysis.json')
    hypothesis_file = Path(env_base_dir) / f"env_{dataset}" / f"task{episode_id}" / hypothesis_filename

    if hypothesis_file.exists():
        try:
            with open(hypothesis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            ig_hypotheses = data.get('ig_hypotheses', [])

            if ig_hypotheses:
                # 从IG假设中提取prompts
                ig_prompts = [h.get('prompt', '') for h in ig_hypotheses if h.get('prompt')]
                if ig_prompts:
                    print(f"[IG Hypothesis] Loaded {len(ig_prompts)} dynamic IG prompts from {hypothesis_file}")
                    return ig_prompts
                else:
                    print(f"[IG Hypothesis] No valid prompts in ig_hypotheses, using default")
                    return DEFAULT_IG_PROMPTS
            else:
                print(f"[IG Hypothesis] No ig_hypotheses found in {hypothesis_file}, using default")
                return DEFAULT_IG_PROMPTS

        except Exception as e:
            print(f"[IG Hypothesis] Failed to load {hypothesis_file}: {e}, using default")
            return DEFAULT_IG_PROMPTS
    else:
        print(f"[IG Hypothesis] File not found: {hypothesis_file}, using default")
        return DEFAULT_IG_PROMPTS


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
    # Keep each arm's summaries inside its own output directory.  A global
    # debug directory mixes concurrently or sequentially evaluated protocols.
    debug_dir = os.path.join(video_output_path, "debug")
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

    # Also save JSON format for per-category analysis
    json_file = os.path.join(debug_dir, f"evaluation_stats_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(episode_stats_list, f, indent=2, ensure_ascii=False)
    print(f"[Stats] JSON data written to: {json_file}")

    return stats_file


def _append_episode_metrics(video_output_path, episode_stats):
    """Persist one comparison-ready row, rejecting conflicting duplicates."""
    metrics_path = os.path.join(video_output_path, "episode_metrics.jsonl")
    episode_key = (
        str(episode_stats["scene_id"]),
        str(episode_stats["episode_id"]),
    )
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                existing = json.loads(line)
                existing_key = (
                    str(existing["scene_id"]),
                    str(existing["episode_id"]),
                )
                if existing_key == episode_key:
                    if int(existing["episode_num"]) != int(episode_stats["episode_num"]):
                        raise RuntimeError(
                            f"episode key {episode_key} has conflicting sequence numbers"
                        )
                    return
    with open(metrics_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(episode_stats, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    # Real-time hypothesis generation options
    parser.add_argument(
        "--realtime-hypothesis",
        action="store_true",
        default=False,
        help="Enable real-time hypothesis generation. At the start of each episode, "
             "the agent will: (1) collect 4 panorama views by rotating, "
             "(2) run VLM environment analysis, (3) generate HSVM+IG hypotheses. "
             "This replaces loading pre-computed hypotheses from disk.",
    )
    parser.add_argument(
        "--vlm-api-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="VLM API URL for real-time analysis (default: DashScope API)",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="qwen-vl-max",
        help="VLM model name (default: qwen-vl-max for DashScope)",
    )
    parser.add_argument(
        "--vlm-api-type",
        type=str,
        choices=["ollama", "dashscope"],
        default="dashscope",
        help="VLM API type: ollama (local) or dashscope (Alibaba Cloud, default)",
    )
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default="https://api.deepseek.com/chat/completions",
        help="LLM API URL for hypothesis generation (default: DeepSeek API)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-chat",
        help="LLM model name (default: deepseek-chat)",
    )
    parser.add_argument(
        "--skip-existing-hypothesis",
        action="store_true",
        default=True,
        help="Skip hypothesis generation if JSON file already exists (default: True)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        default=False,
        help="Force regenerate hypotheses even if they exist",
    )
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=10.0,
        help="Habitat action/control loop frequency (default: 10 Hz)",
    )
    parser.add_argument(
        "--max-episodes-per-process",
        type=int,
        default=-1,
        help="Exit cleanly after this many episodes; -1 runs all remaining episodes",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()

    # Build config dicts for real-time analysis
    vlm_config = {
        'api_url': args.vlm_api_url,
        'model': args.vlm_model,
        'api_type': args.vlm_api_type,
        'timeout': 300,
        'temperature': 0.2
    }
    llm_config = {
        'api_url': args.llm_api_url,
        'model': args.llm_model,
        'temperature': 0.3
    }

    return (args.dataset, args.detector_mode, args.disable_vlm,
            args.realtime_hypothesis, vlm_config, llm_config,
            not args.force_regenerate, args.control_rate_hz,
            args.max_episodes_per_process, unknown)


def main(cfg: DictConfig, dataset: str = "hm3dv2", disable_vlm: bool = False,
         realtime_hypothesis: bool = False, vlm_config: dict = None,
         llm_config: dict = None, skip_existing: bool = True,
         control_rate_hz: float = 10.0,
         max_episodes_per_process: int = -1) -> None:
    global msg_observations, global_action, ros_state
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result
    global vlm_verify_count_this_episode
    global get_multi_source_cosine_with_ig, get_multi_source_cosine_with_dynamic_ig, VLM_DISABLED

    # Set VLM disabled flag
    VLM_DISABLED = disable_vlm

    # Conditionally import VLM module
    if not disable_vlm:
        from vlm.utils.get_itm_message import (
            get_multi_source_cosine_with_ig as _get_multi_source_cosine_with_ig,
            get_multi_source_cosine_with_dynamic_ig as _get_multi_source_cosine_with_dynamic_ig
        )
        get_multi_source_cosine_with_ig = _get_multi_source_cosine_with_ig
        get_multi_source_cosine_with_dynamic_ig = _get_multi_source_cosine_with_dynamic_ig
        print("[VLM] BLIP2 ITM client initialized (with dynamic IG support)")
    else:
        print("[VLM] DISABLED - Skipping BLIP2 initialization (ablation mode)")
        get_multi_source_cosine_with_ig = None
        get_multi_source_cosine_with_dynamic_ig = None

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
    # Hierarchical hypothesis level for ablation study (1-4)
    max_hierarchy_level = llm_cfg.get('max_hierarchy_level', 4)
    print(f"[LLM Hypothesis] Using max_hierarchy_level={max_hierarchy_level}")

    # ==================== Publish Exploration Parameters to ROS ====================
    # Override ROS launch defaults with Hydra config values (for ablation study)
    # NOTE: Parameters must be set in the exploration_node's private namespace
    # because launch file defines them inside <node> tag
    if hasattr(cfg, 'exploration'):
        expl_cfg = cfg.exploration
        if hasattr(expl_cfg, 'fusion_alpha'):
            # Set in node's private namespace (where C++ reads from)
            rospy.set_param('/exploration_node/exploration/fusion_alpha', float(expl_cfg.fusion_alpha))
            print(f"[Exploration] fusion_alpha={expl_cfg.fusion_alpha} (V_total = α*V_sem + (1-α)*V_ig)")
        if hasattr(expl_cfg, 'use_ig_fusion'):
            rospy.set_param('/exploration_node/exploration/use_ig_fusion', bool(expl_cfg.use_ig_fusion))
            print(f"[Exploration] use_ig_fusion={expl_cfg.use_ig_fusion}")

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

    process_episode_count = number_of_episodes - num_total
    if max_episodes_per_process > 0:
        process_episode_count = min(process_episode_count, max_episodes_per_process)
    print(
        f"[Runtime] control_rate={control_rate_hz:g}Hz, "
        f"episodes_this_process={process_episode_count}"
    )
    for epi in range(process_episode_count):
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

        # Load or generate LLM semantic hypotheses for HSVM (Hierarchical Semantic Value Map)
        # HSVM 已经包含了多层级假说: room_type, target_object, co_occurrence, part_attribute
        episode_id = env.current_episode.episode_id
        scene_id = env.current_episode.scene_id
        task_index = num_total  # Use task index instead of episode_id for hypothesis loading
        # Get env_base_dir from config (with fallback to "env")
        env_base_dir = CONFIG.get('paths', {}).get('llm', {}).get('env_base_dir', 'env')

        # Real-time hypothesis generation mode
        if realtime_hypothesis and REALTIME_ANALYSIS_AVAILABLE:
            print(f"\n[Real-time Mode] Generating hypotheses for Task {task_index}...")
            # Use generate_hypotheses_realtime which handles:
            # 1. Panorama view collection (rotates agent)
            # 2. VLM environment analysis
            # 3. HSVM + IG hypothesis generation
            observations, current_hypotheses, current_ig_prompts, gen_success = generate_hypotheses_realtime(
                env=env,
                observations=observations,
                dataset=dataset,
                task_index=task_index,
                target_object=label,
                episode_id=episode_id,
                scene_id=scene_id,
                env_base_dir=env_base_dir,
                vlm_config=vlm_config or {},
                llm_config=llm_config or {},
                skip_existing=skip_existing
            )

            # Update msg_observations after rotation
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]

            if not gen_success:
                print("[Warning] Real-time hypothesis generation failed, falling back to file-based loading")
                current_hypotheses = load_llm_hypotheses(env_base_dir, dataset, task_index, max_hierarchy_level)
                current_ig_prompts = load_ig_hypotheses(env_base_dir, dataset, task_index)
        else:
            # Load from pre-computed files (original behavior)
            current_hypotheses = load_llm_hypotheses(env_base_dir, dataset, task_index, max_hierarchy_level)
            # Load IG hypotheses (target-agnostic connectivity hypotheses)
            # IG假设用于估计视图的探索潜力，描述走廊、门洞、通道等空间连通性结构
            current_ig_prompts = load_ig_hypotheses(env_base_dir, dataset, task_index)

        print(f"[Task {task_index}, Episode {episode_id}] Target: {label}, "
              f"HSVM Hypotheses: {len(current_hypotheses)}, IG Prompts: {len(current_ig_prompts)}")

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

        rate = rospy.Rate(control_rate_hz)
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
                # Normal mode: run VLM inference with dynamic IG hypotheses
                hypotheses_data, ig_data = get_multi_source_cosine_with_dynamic_ig(
                    observations["rgb"], label, current_hypotheses, current_ig_prompts
                )
                timing_breakdown['vlm_inference'] = time_module.time() - t0
                timing_breakdown['vlm_num_prompts'] = len(hypotheses_data) + len(current_ig_prompts)

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
                # NOTE: 支持动态IG prompts，向后兼容固定3个字段
                ig_score_msg = IGScore()
                ig_score_msg.header.stamp = rospy.Time.now()
                ig_score_msg.ig_score = ig_data["ig_score"]
                # 向后兼容：如果使用动态prompts，用平均分填充固定字段
                if ig_data.get("is_dynamic", False):
                    ig_avg = ig_data["ig_score"]
                    ig_score_msg.corridor_score = ig_avg
                    ig_score_msg.doorway_score = ig_avg
                    ig_score_msg.passage_score = ig_avg
                else:
                    ig_score_msg.corridor_score = ig_data.get("corridor_score", ig_data["ig_score"])
                    ig_score_msg.doorway_score = ig_data.get("doorway_score", ig_data["ig_score"])
                    ig_score_msg.passage_score = ig_data.get("passage_score", ig_data["ig_score"])
                ig_score_pub.publish(ig_score_msg)
                timing_breakdown['semantic_msg_build'] = time_module.time() - t0

                # Print summary (controlled by logging config)
                if log_manager.should_log_semantic_scores():
                    print(f"Multi-source semantic scores:")
                    for i, h in enumerate(hypotheses_data):
                        print(f"  [{i}] {h['type']:15s} | Score: {h['score']:.3f} | Weight: {h['weight']:.3f} | {h['prompt'][:50]}")
                    # 动态IG信息
                    is_dynamic = ig_data.get("is_dynamic", False)
                    individual_scores = ig_data.get("individual_scores", [])
                    if is_dynamic and individual_scores:
                        scores_str = ", ".join(f"{s:.3f}" for s in individual_scores)
                        print(f"Information Gain (IG) score: {ig_data['ig_score']:.3f} "
                              f"[dynamic, {len(individual_scores)} prompts: {scores_str}]")
                    else:
                        print(f"Information Gain (IG) score: {ig_data['ig_score']:.3f} "
                              f"[corridor: {ig_data.get('corridor_score', 0):.3f}, "
                              f"doorway: {ig_data.get('doorway_score', 0):.3f}, "
                              f"passage: {ig_data.get('passage_score', 0):.3f}]")
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
            "scene_id": scene_id,
            "target": label,
            "target_category": env.current_episode.object_category,
            "steps": count_steps,
            "time_seconds": episode_time,
            "vlm_calls": vlm_verify_count_this_episode,
            "success": success,
            "spl": spl,
            "soft_spl": soft_spl,
            "distance_to_goal": distance_to_goal,
            "result": result_text,
        }
        episode_stats_list.append(episode_stats)
        _append_episode_metrics(video_output_path, episode_stats)
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
        (dataset, detector_mode, disable_vlm,
         realtime_hypothesis, vlm_config, llm_config,
         skip_existing, control_rate_hz,
         max_episodes_per_process, overrides) = _parse_dataset_arg()

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

        # Print real-time hypothesis mode status
        if realtime_hypothesis:
            print(f"\n{'='*60}")
            print(f"[Real-time Hypothesis Mode] ENABLED")
            print(f"  VLM: {vlm_config['model']} @ {vlm_config['api_url']}")
            print(f"  LLM: {llm_config['model']} @ {llm_config['api_url']}")
            print(f"  Skip existing: {skip_existing}")
            print(f"{'='*60}\n")

        main(cfg, dataset, disable_vlm,
             realtime_hypothesis, vlm_config, llm_config, skip_existing,
             control_rate_hz, max_episodes_per_process)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
