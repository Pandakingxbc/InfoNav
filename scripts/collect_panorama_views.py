#!/usr/bin/env python3
"""
全景视图采集脚本
================

在 Habitat 环境中让智能体原地旋转采集四张全景图（每张间隔90度）。

采集的图片将用于 VLM 环境分析，与 vlm_environment_analyzer.py 配合使用。

参数说明:
---------
- turn_angle: 30度 (配置文件中定义)
- hfov: 79度 (相机水平视野)
- 每90度采集一张，共4张图 (0°, 90°, 180°, 270°)
- 需要旋转3次才能转90度 (30×3=90)

输出:
-----
- view0.png: 前方 (0°, front)
- view1.png: 左方 (90°, left) - 左转90度
- view2.png: 后方 (180°, back) - 左转180度
- view3.png: 右方 (270°, right) - 左转270度
- task_info.txt: 任务信息

使用示例:
---------
# 采集单个任务的全景图
python scripts/collect_panorama_views.py --dataset hm3dv2 --task 0

# 采集多个任务
python scripts/collect_panorama_views.py --dataset hm3dv2 --start 0 --end 10

# 指定输出目录
python scripts/collect_panorama_views.py --dataset hm3dv2 --task 0 --output-dir test_output

# 可视化模式（显示采集的图片）
python scripts/collect_panorama_views.py --dataset hm3dv2 --task 0 --visualize

"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import habitat
    from habitat.config.default import get_config
    from habitat.core.env import Env
    from omegaconf import DictConfig, OmegaConf
    import cv2
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please ensure habitat-sim and habitat-lab are installed.")
    sys.exit(1)


# ==================== 配置 ====================

# 相机参数
HFOV = 79  # 水平视野角度
TURN_ANGLE = 30  # 每次旋转角度
TURNS_PER_90_DEG = 3  # 转90度需要的次数 (90/30=3)

# 视图方向映射
VIEW_DIRECTIONS = {
    0: "front",   # 0°
    1: "left",    # 90° (左转)
    2: "back",    # 180°
    3: "right"    # 270°
}

# 数据集配置
DATASET_CONFIGS = {
    "hm3dv1": "config/habitat_eval_hm3dv1.yaml",
    "hm3dv2": "config/habitat_eval_hm3dv2.yaml",
    "mp3d": "config/habitat_eval_mp3d.yaml"
}


# ==================== 工具函数 ====================

def load_config(dataset: str) -> DictConfig:
    """加载 Habitat 配置 (与 habitat_evaluation.py 一致)"""
    config_path = DATASET_CONFIGS.get(dataset)
    if not config_path:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 使用 Hydra 加载配置
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from habitat.config.default import patch_config

    # 清理之前的 Hydra 实例
    GlobalHydra.instance().clear()

    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.basename(config_path).replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    # 使用 patch_config 补全必要的配置字段 (如 agents_order)
    cfg = patch_config(cfg)

    return cfg


def save_image(image: np.ndarray, output_path: Path) -> None:
    """保存 RGB 图像为 PNG 文件"""
    # Habitat 返回 RGB，OpenCV 需要 BGR
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)


def save_task_info(output_dir: Path, episode, target_object: str, task_idx: int) -> None:
    """保存任务信息到 task_info.txt (与现有格式一致)"""
    info_path = output_dir / "task_info.txt"

    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Task Index: {task_idx}\n")
        f.write(f"Episode ID: {episode.episode_id}\n")
        f.write(f"Target Object: {target_object}\n")
        f.write(f"Scene ID: {episode.scene_id}\n")


def visualize_views(views: list, output_dir: Path) -> None:
    """可视化采集的四张图片"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Panorama Views - {output_dir.name}", fontsize=14)

    titles = ["View 0: Front (0°)", "View 1: Left (90°)",
              "View 2: Back (180°)", "View 3: Right (270°)"]

    for idx, (ax, view, title) in enumerate(zip(axes.flat, views, titles)):
        ax.imshow(view)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "panorama_overview.png", dpi=150)
    plt.show()


# ==================== 目标物体映射 ====================

# HM3D/MP3D 目标物体 ID 到名称的映射
HM3D_CATEGORY_MAPPING = {
    0: "chair",
    1: "bed",
    2: "plant",
    3: "toilet",
    4: "tv_monitor",
    5: "sofa"
}

MP3D_CATEGORY_MAPPING = {
    # MP3D 有更多类别，这里列出常见的
    0: "chair",
    1: "table",
    2: "picture",
    3: "cabinet",
    4: "cushion",
    5: "sofa",
    6: "bed",
    7: "chest_of_drawers",
    8: "plant",
    9: "sink",
    10: "toilet",
    11: "stool",
    12: "towel",
    13: "tv_monitor",
    14: "shower",
    15: "bathtub",
    16: "counter",
    17: "fireplace",
    18: "gym_equipment",
    19: "seating",
    20: "clothes"
}


def get_target_object_name(episode, dataset: str) -> str:
    """从 episode 获取目标物体名称"""
    # 获取目标物体的 category ID
    if hasattr(episode, 'object_category'):
        category = episode.object_category
    elif hasattr(episode, 'goals') and len(episode.goals) > 0:
        goal = episode.goals[0]
        if hasattr(goal, 'object_category'):
            category = goal.object_category
        elif hasattr(goal, 'object_id'):
            # 尝试从 object_id 提取
            category = goal.object_id
        else:
            category = "unknown"
    else:
        category = "unknown"

    # 根据数据集选择映射
    if dataset.startswith("hm3d"):
        mapping = HM3D_CATEGORY_MAPPING
    else:
        mapping = MP3D_CATEGORY_MAPPING

    # 如果是字符串，直接返回
    if isinstance(category, str):
        return category

    # 如果是数字，查找映射
    return mapping.get(category, f"object_{category}")


# ==================== 主采集函数 ====================

def collect_panorama_views(
    dataset: str,
    task_idx: int,
    output_dir: Path,
    visualize: bool = False
) -> bool:
    """
    采集单个任务的四张全景图

    Args:
        dataset: 数据集名称 (hm3dv1, hm3dv2, mp3d)
        task_idx: 任务索引
        output_dir: 输出目录
        visualize: 是否可视化

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*60}")
    print(f"Collecting Panorama Views")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Task Index: {task_idx}")
    print(f"Output Directory: {output_dir}")
    print(f"Turn Angle: {TURN_ANGLE}° (3 turns = 90°)")
    print(f"HFOV: {HFOV}°")

    try:
        # 1. 加载配置
        print(f"\n[1/5] Loading configuration...")
        cfg = load_config(dataset)

        # 2. 创建环境
        print(f"[2/5] Creating Habitat environment...")
        env = Env(config=cfg.habitat)

        # 3. 跳转到指定任务
        print(f"[3/5] Navigating to task {task_idx}...")
        observations = env.reset()

        # 跳过前面的 episodes 到达目标任务
        for _ in range(task_idx):
            observations = env.reset()

        # 获取当前 episode 信息
        episode = env.current_episode
        target_object = get_target_object_name(episode, dataset)

        print(f"  Episode ID: {episode.episode_id}")
        print(f"  Scene: {episode.scene_id}")
        print(f"  Target Object: {target_object}")

        # 4. 采集四张全景图
        print(f"\n[4/5] Collecting 4 panorama views...")

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        views = []

        for view_idx in range(4):
            direction = VIEW_DIRECTIONS[view_idx]

            # 获取当前 RGB 图像
            rgb_image = observations["rgb"]
            views.append(rgb_image.copy())

            # 保存图像
            view_path = output_dir / f"view{view_idx}.png"
            save_image(rgb_image, view_path)
            print(f"  View {view_idx} ({direction}): saved to {view_path}")

            # 如果不是最后一张，则左转90度（3次左转）
            if view_idx < 3:
                for _ in range(TURNS_PER_90_DEG):
                    observations = env.step({"action": "turn_left"})

        # 5. 保存任务信息
        print(f"\n[5/5] Saving task info...")
        save_task_info(output_dir, episode, target_object, task_idx)
        print(f"  Task info saved to {output_dir / 'task_info.txt'}")

        # 可视化
        if visualize:
            print(f"\nVisualizing panorama views...")
            visualize_views(views, output_dir)

        # 关闭环境
        env.close()

        print(f"\n{'='*60}")
        print(f"SUCCESS: Panorama views collected!")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def collect_batch(
    dataset: str,
    start_task: int,
    end_task: int,
    output_base_dir: Path,
    visualize: bool = False
) -> dict:
    """批量采集多个任务的全景图"""

    print(f"\n{'#'*60}")
    print(f"# Batch Panorama Collection")
    print(f"# Dataset: {dataset}")
    print(f"# Tasks: {start_task} to {end_task}")
    print(f"# Output: {output_base_dir}")
    print(f"{'#'*60}\n")

    stats = {
        "total": end_task - start_task + 1,
        "success": 0,
        "failed": 0,
        "failed_tasks": []
    }

    for task_idx in range(start_task, end_task + 1):
        output_dir = output_base_dir / f"task{task_idx}"

        print(f"\n[Progress: {task_idx - start_task + 1}/{stats['total']}]")

        success = collect_panorama_views(
            dataset=dataset,
            task_idx=task_idx,
            output_dir=output_dir,
            visualize=visualize
        )

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            stats["failed_tasks"].append(task_idx)

    # 打印统计
    print(f"\n{'#'*60}")
    print(f"# Batch Collection Complete")
    print(f"# Success: {stats['success']}/{stats['total']}")
    print(f"# Failed: {stats['failed']}")
    if stats["failed_tasks"]:
        print(f"# Failed Tasks: {stats['failed_tasks']}")
    print(f"{'#'*60}\n")

    return stats


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="Collect panorama views from Habitat environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single task
  python scripts/collect_panorama_views.py --dataset hm3dv2 --task 0

  # Multiple tasks
  python scripts/collect_panorama_views.py --dataset hm3dv2 --start 0 --end 10

  # With visualization
  python scripts/collect_panorama_views.py --dataset hm3dv2 --task 0 --visualize
        """
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        help="Dataset to use"
    )

    parser.add_argument(
        "--task", "-t",
        type=int,
        default=None,
        help="Single task index to collect"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start task index for batch collection"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End task index for batch collection"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: env/env_{dataset}/task{idx})"
    )

    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Visualize collected views"
    )

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path("env") / f"env_{args.dataset}"

    # 单任务模式
    if args.task is not None:
        output_dir = output_base / f"task{args.task}" if args.output_dir is None else Path(args.output_dir)
        success = collect_panorama_views(
            dataset=args.dataset,
            task_idx=args.task,
            output_dir=output_dir,
            visualize=args.visualize
        )
        return 0 if success else 1

    # 批量模式
    elif args.start is not None and args.end is not None:
        stats = collect_batch(
            dataset=args.dataset,
            start_task=args.start,
            end_task=args.end,
            output_base_dir=output_base,
            visualize=args.visualize
        )
        return 0 if stats["failed"] == 0 else 1

    else:
        parser.print_help()
        print("\nError: Please specify --task or both --start and --end")
        return 1


if __name__ == "__main__":
    sys.exit(main())
