"""
Stair Detection Utilities - Following ASCENT's Approach
完全参考ASCENT的实现,不使用独立的StairDetector服务

ASCENT楼梯检测流程:
1. GroundingDINO检测 "stair ." → 获得候选bbox
2. MobileSAM精确分割bbox → 获得stair_mask (SAM mask)
3. RedNet语义分割 → 获得seg_mask (semantic mask, class_id=17为楼梯)
4. 融合: fusion_stair_mask = stair_mask & (seg_mask == 17)
5. 条件: np.any(stair_mask) > 0 AND np.sum(seg_mask == 17) > 20
"""

import sys
import numpy as np
import torch
from typing import Tuple, Optional

# Add RedNet to path
sys.path.insert(0, "vlm/")
try:
    from RedNet.RedNet_model import load_rednet
except ImportError:
    load_rednet = None
    print("Warning: Could not import RedNet")
sys.path.pop(0)

from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.segmentor.sam import MobileSAMClient

# MPCAT40 semantic class IDs (参考ASCENT)
STAIR_CLASS_ID = 17
FLOOR_CLASS_ID = 2
WALL_CLASS_ID = 1


class StairDetectionResult:
    """楼梯检测结果"""
    def __init__(self):
        self.has_stair = False
        self.is_upstair = False
        self.is_downstair = False
        self.stair_mask = None  # SAM mask (GroundingDINO + MobileSAM)
        self.seg_mask = None    # RedNet semantic mask (完整语义分割)
        self.fusion_mask = None  # Fusion mask (stair_mask & seg_mask==17)
        self.num_gdino_detections = 0
        self.num_sam_pixels = 0
        self.num_rednet_pixels = 0
        self.num_fusion_pixels = 0


def load_rednet_model(checkpoint_path: str, device=None):
    """
    加载RedNet语义分割模型
    参考: ascent/utils.py::load_rednet_model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_rednet is None:
        raise RuntimeError("RedNet not available. Cannot load model.")

    print(f"Loading RedNet from {checkpoint_path}...")
    red_sem_pred = load_rednet(
        device,
        ckpt=checkpoint_path,
        resize=True,
        stabilize=False
    )
    red_sem_pred.eval()
    print("RedNet loaded successfully!")
    return red_sem_pred


def detect_stairs_ascent(
    rgb: np.ndarray,
    depth: np.ndarray,
    rednet_model: torch.nn.Module,
    gdino_client: GroundingDINOClient,
    sam_client: MobileSAMClient,
    pitch_angle: float = 0.0,
    gdino_conf_threshold: float = 0.60,  # ASCENT使用0.60
    rednet_pixel_threshold: int = 20,     # ASCENT使用20
    device=None
) -> StairDetectionResult:
    """
    完整的ASCENT风格楼梯检测

    参考:
    - ascent/map_controller.py::_get_object_detections_with_stair_and_person (行783-789)
    - ascent/mapping/obstacle_map.py::update_map (行520-527)

    Args:
        rgb: RGB图像 (H, W, 3)
        depth: 深度图像 (H, W) 或 (H, W, 1)
        rednet_model: 预加载的RedNet模型
        gdino_client: GroundingDINO客户端
        sam_client: MobileSAM客户端
        pitch_angle: 相机俯仰角 (正=向上看, 负=向下看)
        gdino_conf_threshold: GroundingDINO置信度阈值
        rednet_pixel_threshold: RedNet最小像素数阈值
        device: 计算设备

    Returns:
        StairDetectionResult对象
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = StairDetectionResult()
    height, width = rgb.shape[:2]

    # ============================================================
    # Stage 1: GroundingDINO检测 "stair ."
    # 参考: map_controller.py 行701-704, 783-784
    # ============================================================
    print("[Stage 1] GroundingDINO detection for 'stair .'")

    non_coco_detections = gdino_client.predict(
        rgb,
        caption="stair .",  # ASCENT只用 "stair ."
        box_threshold=0.35,  # 初始检测阈值较低
        text_threshold=0.25
    )

    # 过滤: 只保留"stair"类别 + 置信度>0.60
    # 参考: map_controller.py 行783-784
    non_coco_detections.filter_by_class(["stair"])
    non_coco_detections.filter_by_conf(gdino_conf_threshold)

    num_detections = len(non_coco_detections.logits)
    result.num_gdino_detections = num_detections
    print(f"  Found {num_detections} stair detections (conf > {gdino_conf_threshold})")

    # ============================================================
    # Stage 2: MobileSAM分割
    # 参考: map_controller.py 行786-789
    # ============================================================
    print("[Stage 2] MobileSAM segmentation")

    stair_mask = np.zeros((height, width), dtype=bool)

    for idx in range(num_detections):
        stair_bbox_denorm = non_coco_detections.boxes[idx] * np.array([width, height, width, height])
        print(f"  Processing bbox {idx+1}/{num_detections}: {stair_bbox_denorm.astype(int)}")

        try:
            # 参考: map_controller.py 行788
            mask = sam_client.segment_bbox(rgb, stair_bbox_denorm.tolist())
            stair_mask[mask > 0] = True  # 累积所有检测的mask
        except Exception as e:
            print(f"  Warning: SAM failed for bbox {idx+1}: {e}")

    result.stair_mask = stair_mask
    result.num_sam_pixels = int(np.sum(stair_mask))
    print(f"  SAM mask pixels: {result.num_sam_pixels}")

    # ============================================================
    # Stage 3: RedNet语义分割
    # 参考: obstacle_map.py 行520-521
    # ============================================================
    print("[Stage 3] RedNet semantic segmentation")

    # 准备输入
    rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).to(device)
    if depth.ndim == 2:
        depth = depth[:, :, np.newaxis]
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)

    # RedNet推理
    with torch.no_grad():
        semantic_pred = rednet_model(rgb_tensor, depth_tensor)

    seg_mask = semantic_pred.squeeze(0).cpu().numpy().astype(np.uint8)
    result.seg_mask = seg_mask

    # 提取楼梯类别 (class_id = 17)
    stair_map = (seg_mask == STAIR_CLASS_ID)
    result.num_rednet_pixels = int(np.sum(stair_map))
    print(f"  RedNet stair pixels (class_id={STAIR_CLASS_ID}): {result.num_rednet_pixels}")

    # ============================================================
    # Stage 4: 融合判断 (ASCENT的关键步骤)
    # 参考: obstacle_map.py 行520-523
    # 条件: np.any(stair_mask) > 0 AND np.sum(seg_mask == STAIR_CLASS_ID) > 20
    # ============================================================
    print("[Stage 4] Mask fusion and decision")

    # ASCENT的判断条件
    condition_sam = np.any(stair_mask) > 0
    condition_rednet = np.sum(seg_mask == STAIR_CLASS_ID) > rednet_pixel_threshold

    print(f"  Fusion conditions:")
    print(f"    SAM has detections: {condition_sam}")
    print(f"    RedNet pixels > {rednet_pixel_threshold}: {condition_rednet} ({result.num_rednet_pixels})")

    if condition_sam and condition_rednet:
        # 融合: 取交集 (参考: obstacle_map.py 行522)
        fusion_stair_mask = stair_mask & stair_map
        result.fusion_mask = fusion_stair_mask
        result.num_fusion_pixels = int(np.sum(fusion_stair_mask))

        print(f"  Fusion pixels: {result.num_fusion_pixels}")

        # ASCENT的判断: 只要融合后有像素就认为检测到楼梯
        if np.any(fusion_stair_mask) > 0:
            result.has_stair = True
            print("  ✓ Stair detected (fusion consensus)")

            # 判断上楼/下楼 (参考: obstacle_map.py 行535, 542)
            # pitch_angle >= 0 → 向上看 → upstair
            # pitch_angle < 0  → 向下看 → downstair
            if pitch_angle >= 0:
                result.is_upstair = True
                print(f"  Direction: UPSTAIR (pitch={pitch_angle:.2f}°)")
            else:
                result.is_downstair = True
                print(f"  Direction: DOWNSTAIR (pitch={pitch_angle:.2f}°)")
        else:
            print("  ✗ No fusion pixels - false positive")
    else:
        print("  ✗ Fusion conditions not met")

    return result


def detect_stairs_simple(
    rgb: np.ndarray,
    depth: np.ndarray,
    rednet_model: torch.nn.Module,
    gdino_client: GroundingDINOClient,
    sam_client: MobileSAMClient,
    pitch_angle: float = 0.0
) -> bool:
    """
    简化接口: 只返回是否检测到楼梯

    Args:
        rgb: RGB图像
        depth: 深度图像
        rednet_model: RedNet模型
        gdino_client: GroundingDINO客户端
        sam_client: MobileSAM客户端
        pitch_angle: 相机俯仰角

    Returns:
        True if stairs detected, False otherwise
    """
    result = detect_stairs_ascent(
        rgb, depth, rednet_model,
        gdino_client, sam_client,
        pitch_angle
    )
    return result.has_stair


# 使用示例
if __name__ == "__main__":
    print("""
使用示例:

# 初始化 (启动时一次)
from vlm.utils.stair_utils import load_rednet_model, detect_stairs_ascent
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.segmentor.sam import MobileSAMClient

rednet_model = load_rednet_model("pretrained_weights/rednet_semmap_mp3d_40.pth")
gdino_client = GroundingDINOClient(port=13184)
sam_client = MobileSAMClient(port=13183)

# 检测楼梯 (每一帧)
result = detect_stairs_ascent(
    rgb, depth, rednet_model,
    gdino_client, sam_client,
    pitch_angle=camera_pitch
)

if result.has_stair:
    print(f"检测到楼梯!")
    print(f"  方向: {'上楼' if result.is_upstair else '下楼'}")
    print(f"  融合像素: {result.num_fusion_pixels}")
    # 使用 result.fusion_mask 进行后续处理
    """)
