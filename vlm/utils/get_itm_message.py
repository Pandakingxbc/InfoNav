import cv2
import numpy as np
import os
import yaml
from pathlib import Path

# ============================================================
# Load VLM Configuration
# ============================================================
_config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
if _config_path.exists():
    with open(_config_path, encoding='utf-8') as f:
        _VLM_CONFIG = yaml.safe_load(f)
else:
    _VLM_CONFIG = {}

# Check if parallel mode is enabled (priority: env var > config file > default)
USE_PARALLEL_BLIP2 = os.environ.get('USE_PARALLEL_BLIP2',
                                     str(_VLM_CONFIG.get('vlm', {}).get('blip2_itm', {}).get('parallel', {}).get('enabled', False))).lower() == 'true'

# Get port configuration from config file (with fallback to hardcoded defaults)
_BLIP2_PORT = _VLM_CONFIG.get('vlm', {}).get('blip2_itm', {}).get('port', 12182)
_BLIP2_PARALLEL_PORTS = _VLM_CONFIG.get('vlm', {}).get('blip2_itm', {}).get('parallel', {}).get('ports', [12182, 12192])

if USE_PARALLEL_BLIP2:
    from vlm.itm.blip2itm_parallel import ParallelBLIP2ITMClient
    # Use 2 BLIP2 servers for parallel processing
    itmclient = ParallelBLIP2ITMClient(ports=_BLIP2_PARALLEL_PORTS)
    print(f"[ITM] Using ParallelBLIP2ITMClient with {len(_BLIP2_PARALLEL_PORTS)} servers (ports {_BLIP2_PARALLEL_PORTS})")
else:
    from vlm.itm.blip2itm import BLIP2ITMClient
    # Use single BLIP2 server
    itmclient = BLIP2ITMClient(port=_BLIP2_PORT)
    print(f"[ITM] Using single BLIP2ITMClient (port {_BLIP2_PORT})")

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score


# 下一步修改这里，添加多源融合的text+image itm评分函数
def get_itm_message_cosine(rgb_image, label, room):
    """
    原始单源cosine计算，保持向后兼容
    """
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    cosine = itmclient.cosine(rgb_image, txt)
    return cosine


def get_multi_source_cosine(rgb_image, label, room, semantic_hypotheses):
    """
    计算多源语义假说的cosine分数（批量推理优化版）

    支持两种 JSON 格式:
    1. 新格式 (HSVM): 使用 'type', 'prompt', 'weight' 字段
    2. 旧格式: 使用 'hypothesis_type', 'description' 字段

    Args:
        rgb_image: 当前观测的RGB图像
        label: 目标物体名称 (如 "bed", "chair")
        room: 目标物体所在房间类型 (如 "bedroom", "everywhere")
        semantic_hypotheses: LLM生成的假说列表
            新格式: [{"id": 1, "type": "room_type", "prompt": "bedroom", ...}, ...]
            旧格式: [{"id": 1, "hypothesis_type": "room_context", "description": "bedroom", ...}, ...]

    Returns:
        hypotheses_data: List[dict] - 包含所有假说的元数据
            [
                {
                    "id": 0,
                    "type": "target_object",
                    "prompt": "Seems like there is a bedroom or a bed ahead?",
                    "confidence": 1.0,
                    "navigation_value": 1.0,
                    "weight": 1.0,
                    "score": 0.65
                },
                ...
            ]
    """
    import time
    t_start_total = time.time()

    hypotheses_data = []

    # 1. 构建所有prompt（目标物体 + 假说）
    t0 = time.time()
    prompts_list = []
    metadata_list = []

    # 添加目标物体prompt
    if room != "everywhere":
        target_prompt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        target_prompt = f"Seems like there is a {label} ahead?"

    prompts_list.append(target_prompt)
    metadata_list.append({
        "id": 0,
        "type": "target_object",
        "prompt": target_prompt,
        "confidence": 1.0,
        "navigation_value": 1.0,
        "weight": 1.0
    })

    # 添加所有假说prompts (支持新旧两种格式)
    for hyp in semantic_hypotheses:
        # 兼容新旧两种字段名
        hyp_description = hyp.get("prompt") or hyp.get("description", "")
        hyp_type = hyp.get("type") or hyp.get("hypothesis_type", "unknown")
        hyp_confidence = hyp.get("confidence", 0.5)
        hyp_nav_value = hyp.get("navigation_value", 0.5)
        # 优先使用预计算的 weight，否则计算
        hyp_weight = hyp.get("weight") or (hyp_confidence * hyp_nav_value)

        hyp_prompt = f"Seems like there is a {hyp_description} ahead?"
        prompts_list.append(hyp_prompt)
        metadata_list.append({
            "id": hyp.get("id", -1),
            "type": hyp_type,
            "prompt": hyp_prompt,
            "confidence": hyp_confidence,
            "navigation_value": hyp_nav_value,
            "weight": hyp_weight
        })
    t_prompt_build = time.time() - t0

    # 2. 批量调用BLIP2（一次HTTP请求获取所有scores）
    t0 = time.time()
    cosine_scores = itmclient.cosine_batch(rgb_image, prompts_list)
    t_blip2_inference = time.time() - t0

    # 3. 组装结果
    t0 = time.time()
    for metadata, score in zip(metadata_list, cosine_scores):
        metadata["score"] = score
        hypotheses_data.append(metadata)
    t_result_assembly = time.time() - t0

    t_total = time.time() - t_start_total

    # Print detailed VLM timing only if enabled
    try:
        from basic_utils.logging import get_log_manager
        logger = get_log_manager()
        logger.log_vlm_detail(
            f"  [VLM Detail] Total: {t_total:.3f}s | Prompt Build: {t_prompt_build:.4f}s | "
            f"BLIP2 Inference: {t_blip2_inference:.3f}s ({len(prompts_list)} prompts) | "
            f"Assembly: {t_result_assembly:.4f}s"
        )
    except ImportError:
        # Fallback if LogManager not available
        pass

    return hypotheses_data


# ============================================================
# Information Gain (IG) Estimation
# ============================================================
# Reference: main.tex Section III-D "VLM-Based Future IG Estimation"
#
# The IG score estimates the exploration potential of the current view
# by measuring visual-semantic connectivity using BLIP-2 ITM scores.
# Higher scores indicate the view shows passages/doorways leading to
# unexplored areas, suggesting higher expected information gain.
# ============================================================

# Connectivity prompts for IG estimation (from paper Section III-D)
IG_CONNECTIVITY_PROMPTS = [
    "This view shows a corridor or hallway leading to other areas",
    "There is a doorway or opening that leads to another room",
    "This passage connects to multiple rooms or spaces"
]


def get_ig_score(rgb_image):
    """
    Calculate Information Gain (IG) score for the current observation.

    This implements the VLM-Based Future IG Estimation from paper Section III-D:

        IG(f_i) ∝ (1/3) * Σ_{j=1}^{3} BLIP2-ITM(I_i, T_j)

    where T_j are connectivity prompts describing high-connectivity scenes
    (corridors, doorways, passages).

    Args:
        rgb_image: Current RGB observation (numpy array, HxWx3)

    Returns:
        float: IG score in [0, 1], higher = more exploration potential
    """
    import time
    t_start = time.time()

    # Batch compute ITM scores for all connectivity prompts
    cosine_scores = itmclient.cosine_batch(rgb_image, IG_CONNECTIVITY_PROMPTS)

    # Average the scores as per paper equation
    ig_score = sum(cosine_scores) / len(cosine_scores)

    t_elapsed = time.time() - t_start

    # Log timing if available
    try:
        from basic_utils.logging import get_log_manager
        logger = get_log_manager()
        logger.log_vlm_detail(
            f"  [IG Score] {ig_score:.3f} | Time: {t_elapsed:.3f}s | "
            f"Scores: [{', '.join(f'{s:.3f}' for s in cosine_scores)}]"
        )
    except ImportError:
        pass

    return ig_score


def get_ig_score_detailed(rgb_image):
    """
    Calculate Information Gain (IG) score with detailed breakdown.

    Args:
        rgb_image: Current RGB observation (numpy array, HxWx3)

    Returns:
        dict: {
            "ig_score": float,           # Average IG score
            "corridor_score": float,     # Score for corridor prompt
            "doorway_score": float,      # Score for doorway prompt
            "passage_score": float,      # Score for passage prompt
            "prompts": list[str],        # The prompts used
        }
    """
    import time
    t_start = time.time()

    # Batch compute ITM scores for all connectivity prompts
    cosine_scores = itmclient.cosine_batch(rgb_image, IG_CONNECTIVITY_PROMPTS)

    # Average the scores
    ig_score = sum(cosine_scores) / len(cosine_scores)

    t_elapsed = time.time() - t_start

    result = {
        "ig_score": ig_score,
        "corridor_score": cosine_scores[0],
        "doorway_score": cosine_scores[1],
        "passage_score": cosine_scores[2],
        "prompts": IG_CONNECTIVITY_PROMPTS,
        "inference_time": t_elapsed
    }

    return result


def get_ig_score_dynamic(rgb_image, ig_prompts=None):
    """
    Calculate Information Gain (IG) score with dynamic prompts.

    This function supports LLM-generated IG hypotheses for exploration potential
    estimation. Falls back to default prompts if none provided.

    Args:
        rgb_image: Current RGB observation (numpy array, HxWx3)
        ig_prompts: List of IG prompts (optional). If None, uses default prompts.

    Returns:
        dict: {
            "ig_score": float,           # Average IG score
            "individual_scores": list,   # Score for each prompt
            "prompts": list[str],        # The prompts used
            "inference_time": float,     # Time taken for inference
            "is_dynamic": bool           # Whether dynamic prompts were used
        }
    """
    import time
    t_start = time.time()

    # Use provided prompts or fall back to default
    if ig_prompts and len(ig_prompts) > 0:
        prompts = ig_prompts
        is_dynamic = True
    else:
        prompts = IG_CONNECTIVITY_PROMPTS
        is_dynamic = False

    # Batch compute ITM scores for all prompts
    cosine_scores = itmclient.cosine_batch(rgb_image, prompts)

    # Average the scores
    ig_score = sum(cosine_scores) / len(cosine_scores)

    t_elapsed = time.time() - t_start

    # Log timing if available
    try:
        from basic_utils.logging import get_log_manager
        logger = get_log_manager()
        logger.log_vlm_detail(
            f"  [IG Score Dynamic] {ig_score:.3f} | Time: {t_elapsed:.3f}s | "
            f"Prompts: {len(prompts)} ({'dynamic' if is_dynamic else 'default'}) | "
            f"Scores: [{', '.join(f'{s:.3f}' for s in cosine_scores)}]"
        )
    except ImportError:
        pass

    result = {
        "ig_score": ig_score,
        "individual_scores": cosine_scores,
        "prompts": prompts,
        "inference_time": t_elapsed,
        "is_dynamic": is_dynamic
    }

    return result


def get_multi_source_cosine_with_ig(rgb_image, label, semantic_hypotheses, ig_weights=None):
    """
    计算多源语义假说的cosine分数 + IG分数

    HSVM (Hierarchical Semantic Value Map) 包含多层级假说:
    - room_type: 房间类型
    - target_object: 目标物体
    - co_occurrence: 共现物体
    - part_attribute: 物体部件

    IG Score 在 server 端直接加权计算。

    Args:
        rgb_image: 当前观测的RGB图像
        label: 目标物体名称 (用于日志记录)
        semantic_hypotheses: LLM生成的假说列表
            格式: [{"id": 1, "type": "room_type", "prompt": "bedroom",
                   "confidence": 0.8, "navigation_value": 0.9, "weight": 0.72}, ...]
        ig_weights: IG 三个 prompt 的权重，默认等权 [1/3, 1/3, 1/3]

    Returns:
        tuple: (hypotheses_data, ig_data)
            - hypotheses_data: List[dict] - 语义假说结果
            - ig_data: dict - IG分数结果 (server端加权计算)
    """
    import time
    t_start_total = time.time()

    # ========== 1. Build HSVM semantic prompts ==========
    prompts_list = []
    metadata_list = []

    for hyp in semantic_hypotheses:
        hyp_description = hyp.get("prompt") or hyp.get("description", "")
        hyp_type = hyp.get("type") or hyp.get("hypothesis_type", "unknown")
        hyp_confidence = hyp.get("confidence", 0.5)
        hyp_nav_value = hyp.get("navigation_value", 0.5)
        hyp_weight = hyp.get("weight") or (hyp_confidence * hyp_nav_value)

        hyp_prompt = f"Seems like there is a {hyp_description} ahead?"
        prompts_list.append(hyp_prompt)
        metadata_list.append({
            "id": hyp.get("id", -1),
            "type": hyp_type,
            "prompt": hyp_prompt,
            "confidence": hyp_confidence,
            "navigation_value": hyp_nav_value,
            "weight": hyp_weight
        })

    # ========== 2. HSVM semantic inference (batch) ==========
    t0 = time.time()
    if prompts_list:
        semantic_scores = itmclient.cosine_batch(rgb_image, prompts_list)
    else:
        semantic_scores = []
    t_semantic_inference = time.time() - t0

    # Build semantic hypotheses data
    hypotheses_data = []
    for metadata, score in zip(metadata_list, semantic_scores):
        metadata["score"] = score
        hypotheses_data.append(metadata)

    # ========== 3. IG Score (server端加权计算) ==========
    t0 = time.time()
    ig_data = itmclient.ig_score_weighted(rgb_image, weights=ig_weights)
    t_ig_inference = time.time() - t0

    t_total = time.time() - t_start_total

    # Log timing
    try:
        from basic_utils.logging import get_log_manager
        logger = get_log_manager()
        logger.log_vlm_detail(
            f"  [VLM] Total: {t_total:.3f}s | "
            f"HSVM: {t_semantic_inference:.3f}s ({len(prompts_list)} prompts) | "
            f"IG: {t_ig_inference:.3f}s (score: {ig_data['ig_score']:.3f})"
        )
    except ImportError:
        pass

    return hypotheses_data, ig_data


def get_multi_source_cosine_with_dynamic_ig(rgb_image, label, semantic_hypotheses, ig_prompts=None):
    """
    计算多源语义假说的cosine分数 + 动态IG分数

    与 get_multi_source_cosine_with_ig 类似，但支持LLM动态生成的IG假设，
    而不是固定的3个connectivity prompts。

    HSVM (Hierarchical Semantic Value Map) 包含多层级假说:
    - room_type: 房间类型
    - target_object: 目标物体
    - co_occurrence: 共现物体
    - part_attribute: 物体部件

    IG假设是目标无关的连通性描述（corridors, doorways, passages等）。

    Args:
        rgb_image: 当前观测的RGB图像
        label: 目标物体名称 (用于日志记录)
        semantic_hypotheses: LLM生成的HSVM假说列表
            格式: [{"id": 1, "type": "room_type", "prompt": "bedroom",
                   "confidence": 0.8, "navigation_value": 0.9, "weight": 0.72}, ...]
        ig_prompts: LLM生成的IG假说prompts列表 (可选)
            格式: ["corridor leading to bedrooms", "open doorway", ...]
            如果为None或空，使用默认的IG_CONNECTIVITY_PROMPTS

    Returns:
        tuple: (hypotheses_data, ig_data)
            - hypotheses_data: List[dict] - 语义假说结果
            - ig_data: dict - IG分数结果 (动态或默认prompts)
    """
    import time
    t_start_total = time.time()

    # ========== 1. Build HSVM semantic prompts ==========
    prompts_list = []
    metadata_list = []

    for hyp in semantic_hypotheses:
        hyp_description = hyp.get("prompt") or hyp.get("description", "")
        hyp_type = hyp.get("type") or hyp.get("hypothesis_type", "unknown")
        hyp_confidence = hyp.get("confidence", 0.5)
        hyp_nav_value = hyp.get("navigation_value", 0.5)
        hyp_weight = hyp.get("weight") or (hyp_confidence * hyp_nav_value)

        hyp_prompt = f"Seems like there is a {hyp_description} ahead?"
        prompts_list.append(hyp_prompt)
        metadata_list.append({
            "id": hyp.get("id", -1),
            "type": hyp_type,
            "prompt": hyp_prompt,
            "confidence": hyp_confidence,
            "navigation_value": hyp_nav_value,
            "weight": hyp_weight
        })

    # A missing/corrupt hypothesis file must not silently turn the semantic
    # map into IG-only exploration. Keep one explicit target-object channel.
    if not any(item["type"] == "target_object" for item in metadata_list):
        target_prompt = f"Seems like there is a {label} ahead?"
        prompts_list.insert(0, target_prompt)
        metadata_list.insert(0, {
            "id": 0,
            "type": "target_object",
            "prompt": target_prompt,
            "confidence": 1.0,
            "navigation_value": 1.0,
            "weight": 1.0
        })

    # ========== 2. Select dynamic/default IG prompts ==========
    if ig_prompts and len(ig_prompts) > 0:
        active_ig_prompts = ig_prompts
        is_dynamic = True
    else:
        active_ig_prompts = IG_CONNECTIVITY_PROMPTS
        is_dynamic = False

    # ========== 3. One BLIP-2 request for every value-map channel ==========
    # The same RGB embedding supports all HSVM layers and the IG map. A second
    # BLIP-2 replica is therefore only a throughput optimization, not a
    # functional requirement for multiple value maps.
    t0 = time.time()
    combined_prompts = prompts_list + list(active_ig_prompts)
    combined_scores = itmclient.cosine_batch(rgb_image, combined_prompts)
    t_combined_inference = time.time() - t0

    semantic_count = len(prompts_list)
    semantic_scores = combined_scores[:semantic_count]
    ig_scores = combined_scores[semantic_count:]

    hypotheses_data = []
    for metadata, score in zip(metadata_list, semantic_scores):
        metadata["score"] = score
        hypotheses_data.append(metadata)

    ig_score_avg = sum(ig_scores) / len(ig_scores) if ig_scores else 0.0

    # Build ig_data
    ig_data = {
        "ig_score": ig_score_avg,
        "individual_scores": ig_scores,
        "prompts": active_ig_prompts,
        "is_dynamic": is_dynamic
    }

    # For backward compatibility, include fixed keys if using default prompts
    if not is_dynamic and len(ig_scores) >= 3:
        ig_data["corridor_score"] = ig_scores[0]
        ig_data["doorway_score"] = ig_scores[1]
        ig_data["passage_score"] = ig_scores[2]

    t_total = time.time() - t_start_total

    # Log timing
    try:
        from basic_utils.logging import get_log_manager
        logger = get_log_manager()
        logger.log_vlm_detail(
            f"  [VLM] Total: {t_total:.3f}s | "
            f"Combined BLIP2: {t_combined_inference:.3f}s "
            f"({len(prompts_list)} HSVM + {len(active_ig_prompts)} IG prompts) | "
            f"IG: ({len(active_ig_prompts)} prompts, "
            f"{'dynamic' if is_dynamic else 'default'}, score: {ig_score_avg:.3f})"
        )
    except ImportError:
        pass

    return hypotheses_data, ig_data
