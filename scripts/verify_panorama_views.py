#!/usr/bin/env python3
"""
全景视图验证脚本
================

验证采集的四张全景图是否正确（格式、方向、覆盖范围）。

使用示例:
---------
python scripts/verify_panorama_views.py --task-dir env/env_hm3dv2/task0
python scripts/verify_panorama_views.py --task-dir env/env_hm3dv2/task0 --show

"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)


def verify_views(task_dir: Path, show: bool = False) -> bool:
    """
    验证任务目录中的四张全景图

    检查项目:
    1. 文件存在性 (view0.png ~ view3.png)
    2. 图像尺寸一致性
    3. 图像格式 (RGB/BGR, 8-bit)
    4. task_info.txt 存在
    """
    print(f"\n{'='*60}")
    print(f"Verifying Panorama Views")
    print(f"{'='*60}")
    print(f"Task Directory: {task_dir}")

    errors = []
    warnings = []

    # 1. 检查 task_info.txt
    task_info_path = task_dir / "task_info.txt"
    if task_info_path.exists():
        print(f"\n[1/4] task_info.txt: OK")
        with open(task_info_path, 'r') as f:
            content = f.read()
            print(f"  Content:\n{content}")
    else:
        errors.append("task_info.txt not found")
        print(f"\n[1/4] task_info.txt: MISSING")

    # 2. 检查四张图片
    print(f"\n[2/4] Checking view images...")
    views = []
    view_info = []

    directions = ["front (0°)", "left (90°)", "back (180°)", "right (270°)"]

    for i in range(4):
        view_path = task_dir / f"view{i}.png"

        if not view_path.exists():
            errors.append(f"view{i}.png not found")
            print(f"  view{i}.png ({directions[i]}): MISSING")
            continue

        # 读取图像
        img = cv2.imread(str(view_path))
        if img is None:
            errors.append(f"view{i}.png cannot be read")
            print(f"  view{i}.png ({directions[i]}): CANNOT READ")
            continue

        views.append(img)
        height, width, channels = img.shape
        view_info.append({
            "index": i,
            "path": view_path,
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(img.dtype)
        })

        print(f"  view{i}.png ({directions[i]}): {width}x{height}, {channels}ch, {img.dtype}")

    # 3. 检查尺寸一致性
    print(f"\n[3/4] Checking size consistency...")
    if len(view_info) >= 2:
        sizes = set((v["width"], v["height"]) for v in view_info)
        if len(sizes) == 1:
            print(f"  All images have consistent size: {view_info[0]['width']}x{view_info[0]['height']}")
        else:
            warnings.append("Images have inconsistent sizes")
            print(f"  WARNING: Inconsistent sizes detected: {sizes}")
    else:
        print(f"  Cannot check (need at least 2 images)")

    # 4. 计算视图间的差异（检查是否真的是不同方向）
    print(f"\n[4/4] Checking view diversity...")
    if len(views) == 4:
        # 计算相邻视图的差异
        differences = []
        for i in range(4):
            j = (i + 1) % 4
            diff = np.mean(np.abs(views[i].astype(float) - views[j].astype(float)))
            differences.append(diff)
            print(f"  Diff(view{i}, view{j}): {diff:.2f}")

        avg_diff = np.mean(differences)
        if avg_diff < 10:
            warnings.append(f"Views may be too similar (avg diff: {avg_diff:.2f})")
            print(f"  WARNING: Views might not cover different directions (avg diff: {avg_diff:.2f})")
        else:
            print(f"  Views appear to cover different directions (avg diff: {avg_diff:.2f})")
    else:
        print(f"  Cannot check (need all 4 images)")

    # 显示图像
    if show and len(views) == 4:
        print(f"\nDisplaying panorama overview...")
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle(f"Panorama Views - {task_dir.name}", fontsize=14)

            titles = ["View 0: Front (0°)", "View 1: Left (90°)",
                      "View 2: Back (180°)", "View 3: Right (270°)"]

            for idx, (ax, view, title) in enumerate(zip(axes.flat, views, titles)):
                # OpenCV 读取的是 BGR，转换为 RGB 显示
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()

            # 保存概览图
            overview_path = task_dir / "panorama_overview.png"
            plt.savefig(overview_path, dpi=150)
            print(f"  Overview saved to: {overview_path}")

            plt.show()

        except ImportError:
            print("  matplotlib not available, skipping visualization")

    # 总结
    print(f"\n{'='*60}")
    print(f"Verification Summary")
    print(f"{'='*60}")

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"No errors found.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    success = len(errors) == 0
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}\n")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Verify panorama views collected from Habitat"
    )

    parser.add_argument(
        "--task-dir", "-t",
        type=str,
        required=True,
        help="Task directory containing view*.png files"
    )

    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Show the panorama overview"
    )

    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    if not task_dir.exists():
        print(f"Error: Directory not found: {task_dir}")
        return 1

    success = verify_views(task_dir, show=args.show)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
