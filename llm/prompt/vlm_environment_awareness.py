"""
VLM Environment Awareness Prompt Templates
为多源语义融合提供专门的环境感知 Prompt

核心思路：
1. 通过 qwen3-vl 分析 4 张周围环境图片（360° 视角）
2. 提取结构化环境信息（场景、物体、可通行性、危险等）
3. 结合目标物体信息，输出"目标搜索上下文"
4. 供后续 LLM 和检测策略使用

设计遵循：
- 结构化 JSON 输出便于自动化处理
- 与 COCO 类别对齐，便于与检测器协同
- 明确的空间关系表示（相对位置、方向）
"""

import json
from typing import List, Dict, Any

# yangzhi_12_23: 拟采用的promot模板
prompt = f"""
    You are searching for **{target_object}** in an indoor environment.
    Four images show the same location:Image 0=front, Image 1=left, Image 2=back, Image 3=right.
    Analyze them as a single scene and provide:
    1. **Environment Type**: Room type and key characteristics.
    2. **Visible Clues**: Objects or features in these views that are relevant to finding {target_object}.
    3. **Most Likely Area**: Which part of the scene is the target most likely located?
    4. **Priority Views**: Which image(s) (0-3) are most relevant for searching the target, based on visible clues?
    Be concise, task-oriented
    """

# ============================================================================
# PART 3: 后续 LLM 分析 Prompt（基于 VLM 环境感知的结果）
# ============================================================================

LLM_TARGET_SEARCH_STRATEGY_PROMPT = """
你是一个搜索策略专家。基于以下信息，为代理提供搜索目标物体的策略建议。

目标物体: {target_object}
目标特征: {target_features}

VLM 环境分析结果:
{vlm_fusion_result}  # 上面 fusion_summary 部分的 JSON

你需要给出以下信息（JSON 格式）：
{
  "target_object": "{target_object}",
  "search_strategy": {{
    "priority_search_areas": ["area_1", "area_2"],  # 优先搜索的区域（基于环境与目标相关性）
    "expected_companion_objects": ["obj1", "obj2"],  # 预期会看到的伴随物体
    "misdetection_risks": ["risk1", "risk2"],  # 可能被误检的相似物体
    "confidence_threshold": 0.0,  # 建议的检测置信度阈值（0.25-0.65）
    "search_direction_priority": ["front", "left", "right", "back"],  # 搜索方向优先级排序
    "movement_strategy": "forward_and_explore|turn_in_place|follow_wall|none",  # 建议的移动策略
    "notes": "其他建议"
  }}
}

分析思路：
1. 结合 VLM 识别的场景类型，判断目标物体在该场景中的常见位置。
2. 利用 common_companion_objects，缩小搜索范围（例如如果看到 dining_table，sofa 更可能在附近）。
3. 考虑 environmental_constraints 对搜索的影响（如狭窄走廊限制转向）。
4. 提出具体的移动和转向策略，以覆盖最有可能找到目标的区域。
5. 基于环境特征调整置信度阈值（在目标可能被遮挡或背景复杂时降低）。
"""

# ============================================================================
# PART 4: 实用函数 - 管理和格式化 Prompt
# ============================================================================

def get_single_image_prompt() -> str:
    """获取单张图片分析的 prompt"""
    return VLM_SINGLE_IMAGE_PROMPT


def get_multi_image_fusion_prompt(
    target_object: str,
    target_object_cn: str = "",
    target_features: str = ""
) -> str:
    """
    生成多张图片融合的 prompt，包含目标物体信息
    
    Args:
        target_object: 英文目标物体名（如 "cup"）
        target_object_cn: 中文名（如 "杯子"）
        target_features: 目标的特征描述（如 "red color, ceramic, size: small"）
    
    Returns:
        格式化后的 prompt 字符串
    """
    if not target_object_cn:
        target_object_cn = target_object
    if not target_features:
        target_features = f"common {target_object} in indoor environment"
    
    return VLM_MULTI_IMAGE_FUSION_PROMPT_TEMPLATE.format(
        target_object=target_object,
        target_object_cn=target_object_cn,
        target_features=target_features
    )


def get_llm_search_strategy_prompt(
    target_object: str,
    target_features: str,
    vlm_fusion_result: Dict[str, Any]
) -> str:
    """
    生成基于 VLM 环境感知结果的 LLM 搜索策略 prompt
    
    Args:
        target_object: 目标物体名
        target_features: 目标特征描述
        vlm_fusion_result: VLM 返回的 fusion_summary 字典
    
    Returns:
        格式化后的 prompt 字符串
    """
    vlm_json_str = json.dumps(vlm_fusion_result, ensure_ascii=False, indent=2)
    
    return LLM_TARGET_SEARCH_STRATEGY_PROMPT.format(
        target_object=target_object,
        target_features=target_features,
        vlm_fusion_result=vlm_json_str
    )


# ============================================================================
# PART 5: 示例和测试
# ============================================================================

if __name__ == "__main__":
    # 示例 1: 获取单张图片 prompt
    print("=" * 60)
    print("Example 1: Single Image Prompt")
    print("=" * 60)
    print(get_single_image_prompt())
    
    # 示例 2: 获取多张图片融合 prompt
    print("\n" + "=" * 60)
    print("Example 2: Multi-Image Fusion Prompt")
    print("=" * 60)
    fusion_prompt = get_multi_image_fusion_prompt(
        target_object="cup",
        target_object_cn="杯子",
        target_features="ceramic, red color, size: small to medium"
    )
    print(fusion_prompt)
    
    # 示例 3: 获取 LLM 搜索策略 prompt
    print("\n" + "=" * 60)
    print("Example 3: LLM Search Strategy Prompt")
    print("=" * 60)
    mock_vlm_result = {
        "consensus_scene_type": "kitchen",
        "scene_description": "A bright kitchen with white cabinets and a wooden dining table.",
        "merged_objects": [
            {"name": "cup", "locations": ["front", "center"], "aggregate_confidence": 0.92},
            {"name": "table", "locations": ["center"], "aggregate_confidence": 0.95},
        ],
        "target_context": {
            "target_object": "cup",
            "likely_room_type": "kitchen",
            "common_companion_objects": ["table", "sink", "bottle"],
            "possible_occlusion": "cup might be on table surface",
            "search_direction": ["front", "center"],
            "environmental_constraints": "cup likely on kitchen table or counter",
            "confidence_on_finding": 0.88
        },
        "common_passable": ["front", "left", "right"],
        "critical_hazards": []
    }
    
    strategy_prompt = get_llm_search_strategy_prompt(
        target_object="cup",
        target_features="ceramic, red color",
        vlm_fusion_result=mock_vlm_result
    )
    print(strategy_prompt)
