"""
Multi-Source Semantic Fusion Integration Module

将 VLM 分析、LLM 假设生成、价值地图构建集成到导航流程中

Usage:
    from llm.fusion.integration import SemanticFusionPipeline
    
    pipeline = SemanticFusionPipeline(
        vlm_service_url='http://localhost:20004/api/chat',
        llm_service_url='http://localhost:11434',
        target_object='cup'
    )
    
    # 每次采集 4 张图后调用
    result = pipeline.process_observation(
        images=[img1, img2, img3, img4],
        discovered_count=0,
        explored_regions=['kitchen']
    )
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# 导入本地模块
try:
    from ..prompt.value_map_hypothesis import (
        get_hypothesis_generation_prompt,
        parse_hypothesis_output,
    )
    from .value_map_builder import (
        ValueMapBuilder,
        SemanticHypothesis,
        ValueMapConfig,
    )
except ImportError:
    # 如果找不到，尝试相对导入
    from llm.prompt.value_map_hypothesis import (
        get_hypothesis_generation_prompt,
        parse_hypothesis_output,
    )
    from llm.fusion.value_map_builder import (
        ValueMapBuilder,
        SemanticHypothesis,
        ValueMapConfig,
    )

logger = logging.getLogger(__name__)


@dataclass
class ObservationResult:
    """VLM + LLM 处理结果"""
    timestamp: str
    vlm_analysis: Dict
    hypotheses: List[SemanticHypothesis]
    weights: List[float]
    value_map_report: Dict
    primary_recommendation: str
    confidence_score: float
    semantic_entropy: float
    search_suggestions: List[str]


class VLMAnalyzer:
    """VLM 分析器：获取和解析环境分析"""
    
    def __init__(self, service_url: str = 'http://localhost:20004/api/chat'):
        self.service_url = service_url
        self.session = None
    
    def analyze_images(
        self,
        images: List[np.ndarray],
        target_object: str,
        target_object_cn: str
    ) -> Dict:
        """
        调用 VLM 服务分析 4 张 360° 图像
        
        Args:
            images: 4 张图像数组或路径
            target_object: 目标物体英文名
            target_object_cn: 目标物体中文名
            
        Returns:
            VLM 分析结果字典，包含：
                - environment_type: 环境类型
                - visible_clues: 可见线索列表
                - likely_area: 最可能的目标位置
                - priority_views: 优先探索方向
        """
        # 这里实现调用 VLM 服务的逻辑
        # 由于实际环境复杂，这里返回示例结构
        
        return {
            "environment_type": "Kitchen",
            "visible_clues": ["Countertop", "Refrigerator", "Dining table"],
            "likely_area": "Kitchen counter",
            "priority_views": ["Countertop area", "Dining table"],
            "target_object": target_object,
            "target_object_cn": target_object_cn,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.75
        }


class LLMHypothesisGenerator:
    """LLM 假设生成器：生成语义假设"""
    
    def __init__(self, service_url: str = 'http://localhost:11434'):
        self.service_url = service_url
    
    def generate_hypotheses(
        self,
        vlm_analysis: Dict,
        target_object: str,
        target_object_cn: str
    ) -> List[SemanticHypothesis]:
        """
        从 VLM 分析生成语义假设
        
        Args:
            vlm_analysis: VLM 分析结果
            target_object: 目标物体
            target_object_cn: 目标物体中文名
            
        Returns:
            语义假设列表
        """
        # 构建 prompt
        prompt = get_hypothesis_generation_prompt(
            current_environment_analysis=vlm_analysis,
            target_object=target_object,
            target_object_cn=target_object_cn
        )
        
        # 这里实现调用 LLM 服务的逻辑
        # 示例输出（实际会从 LLM 获取）
        llm_response = json.dumps({
            "hypotheses": [
                {
                    "id": 1,
                    "assumption": f"{target_object} is on the kitchen counter",
                    "basis": "Visible countertop space and typical placement",
                    "confidence": 0.85,
                    "search_area": "Kitchen counter",
                    "accompanying_features": ["Under cabinet light", "Counter clutter"],
                    "priority": 1
                },
                {
                    "id": 2,
                    "assumption": f"{target_object} is in dining area",
                    "basis": "Visible dining table in background",
                    "confidence": 0.60,
                    "search_area": "Dining table",
                    "accompanying_features": ["Table surface", "Place settings"],
                    "priority": 2
                },
                {
                    "id": 3,
                    "assumption": f"{target_object} is in cabinet or shelf",
                    "basis": "Common storage location",
                    "confidence": 0.45,
                    "search_area": "Upper cabinets",
                    "accompanying_features": ["Cabinet doors", "Shelving"],
                    "priority": 3
                }
            ]
        })
        
        # 解析响应
        hypotheses = parse_hypothesis_output(llm_response)
        return hypotheses


class SemanticFusionPipeline:
    """
    完整的多源语义融合管道
    
    流程：
    1. VLM 分析 4 张 360° 图像 → 环境特征
    2. LLM 生成 3-5 个语义假设 → 目标位置推断
    3. 价值地图构建器融合假设 → 权重估计
    4. 计算每个位置的语义价值 → V_semantic(p)
    5. 与几何价值融合 → V_total(p) = λ·V_sem + (1-λ)·V_geo
    """
    
    def __init__(
        self,
        vlm_service_url: str = 'http://localhost:20004/api/chat',
        llm_service_url: str = 'http://localhost:11434',
        target_object: str = 'cup',
        target_object_cn: str = '杯子',
        value_map_config: Optional[ValueMapConfig] = None
    ):
        self.vlm_analyzer = VLMAnalyzer(vlm_service_url)
        self.llm_generator = LLMHypothesisGenerator(llm_service_url)
        self.target_object = target_object
        self.target_object_cn = target_object_cn
        self.config = value_map_config or ValueMapConfig()
        
        logger.info(f"SemanticFusionPipeline initialized for target: {target_object}")
    
    def process_observation(
        self,
        images: List[np.ndarray],
        discovered_count: int = 0,
        total_searchable_area: int = 10,
        explored_regions: Optional[List[str]] = None,
        unexplored_regions: Optional[List[str]] = None
    ) -> ObservationResult:
        """
        处理单次观察（4 张图像）
        
        Args:
            images: 4 张 360° 图像
            discovered_count: 已发现的目标实例数
            total_searchable_area: 可搜索区域总数
            explored_regions: 已探索的区域列表
            unexplored_regions: 未探索的区域列表
            
        Returns:
            完整的观察结果
        """
        timestamp = datetime.now().isoformat()
        
        # Step 1: VLM 分析
        logger.info("Step 1: Analyzing images with VLM...")
        vlm_analysis = self.vlm_analyzer.analyze_images(
            images=images,
            target_object=self.target_object,
            target_object_cn=self.target_object_cn
        )
        logger.info(f"VLM analysis complete: {vlm_analysis.get('environment_type')}")
        
        # Step 2: LLM 生成假设
        logger.info("Step 2: Generating hypotheses with LLM...")
        hypotheses = self.llm_generator.generate_hypotheses(
            vlm_analysis=vlm_analysis,
            target_object=self.target_object,
            target_object_cn=self.target_object_cn
        )
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        
        # Step 3: 价值地图构建
        logger.info("Step 3: Building value map with adaptive weighting...")
        builder = ValueMapBuilder(config=self.config)
        builder.add_hypotheses_from_dict({
            'hypotheses': [asdict(h) for h in hypotheses]
        })
        
        # 设置搜索进度
        discovery_rate = discovered_count / max(total_searchable_area, 1)
        builder.set_search_progress(
            discovered_count=discovered_count,
            total_searchable_area=total_searchable_area,
            explored_regions=explored_regions or [],
            unexplored_regions=unexplored_regions or []
        )
        
        # 估计权重（自适应调整）
        weights = builder.estimate_weights()
        logger.info(f"Estimated weights: {[f'{w:.3f}' for w in weights]}")
        
        # Step 4: 生成价值地图报告
        report = builder.generate_value_map_report()
        
        # Step 5: 提取推荐信息
        recommendations = report.get('recommendations', {})
        primary_recommendation = recommendations.get('primary_search_area', 'Unknown')
        confidence = recommendations.get('confidence', 0.0)
        
        # 计算熵值（权重分布的信息熵）
        weights_array = np.array(weights)
        entropy = -np.sum(weights_array * np.log(weights_array + 1e-10))
        
        # 生成搜索建议
        search_suggestions = self._generate_search_suggestions(
            report=report,
            discovery_rate=discovery_rate,
            entropy=entropy
        )
        
        # 组装结果
        result = ObservationResult(
            timestamp=timestamp,
            vlm_analysis=vlm_analysis,
            hypotheses=hypotheses,
            weights=weights,
            value_map_report=report,
            primary_recommendation=primary_recommendation,
            confidence_score=confidence,
            semantic_entropy=entropy,
            search_suggestions=search_suggestions
        )
        
        logger.info(f"Pipeline complete. Primary: {primary_recommendation}, "
                   f"Confidence: {confidence:.3f}")
        
        return result
    
    def _generate_search_suggestions(
        self,
        report: Dict,
        discovery_rate: float,
        entropy: float
    ) -> List[str]:
        """生成搜索建议"""
        suggestions = []
        
        # 基于熵值的建议
        if entropy > 1.0:
            suggestions.append("High uncertainty: explore multiple hypothesis areas")
        elif entropy < 0.5:
            suggestions.append("High confidence: focus on primary hypothesis area")
        
        # 基于发现率的建议
        if discovery_rate < 0.3:
            suggestions.append("Early exploration phase: prioritize unexplored regions")
        elif discovery_rate > 0.7:
            suggestions.append("Intensive search phase: refine within high-confidence areas")
        
        # 基于建议的建议
        recommendations = report.get('recommendations', {})
        if recommendations.get('exploration_suggestion'):
            suggestions.append(recommendations['exploration_suggestion'])
        
        return suggestions
    
    def compute_semantic_value_field(
        self,
        positions: np.ndarray,
        hypothesis_scores: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        计算位置集合的语义价值场
        
        Args:
            positions: (N, 2) 位置数组
            hypothesis_scores: 字典 {hypothesis_id: (N,) 评分数组}
                表示每个假设对各位置的评分
            
        Returns:
            (N,) 语义价值数组，V_semantic(p) = Σ w_k * V^(k)_sem(p)
        """
        # 这里需要实现价值场计算
        # 示例：简单线性融合
        num_positions = len(positions)
        semantic_values = np.zeros(num_positions)
        
        for hyp in self._builder.hypotheses:
            if hyp.id in hypothesis_scores:
                scores = hypothesis_scores[hyp.id]
                weight = self._builder.weights[hyp.id - 1]
                semantic_values += weight * scores
        
        return semantic_values
    
    def get_exploration_strategy(
        self,
        current_position: Tuple[float, float],
        goal_position: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        基于价值地图的探索策略
        
        Args:
            current_position: 当前位置 (x, y)
            goal_position: 目标位置（如果已知）
            
        Returns:
            探索策略字典，包含推荐方向、预期收益等
        """
        # 这里实现探索策略生成
        return {
            "recommended_area": "primary_search_area",
            "expected_value": 0.65,
            "confidence": 0.75,
            "alternative_areas": ["secondary", "tertiary"]
        }


def build_value_map_from_vlm_output(
    vlm_analysis_text: str,
    llm_hypotheses_response: str,
    target_object: str,
    target_object_cn: str = "",
    discovery_rate: float = 0.0,
    explored_regions: Optional[List[str]] = None,
    unexplored_regions: Optional[List[str]] = None,
    config: Optional[ValueMapConfig] = None
) -> ValueMapBuilder:
    """
    便捷函数：从 VLM + LLM 输出直接构建价值地图
    
    Args:
        vlm_analysis_text: VLM 分析文本/JSON
        llm_hypotheses_response: LLM 生成的假设 JSON
        target_object: 目标物体
        target_object_cn: 目标物体中文名
        discovery_rate: 发现率 (0-1)
        explored_regions: 已探索区域列表
        unexplored_regions: 未探索区域列表
        config: 价值地图配置
        
    Returns:
        配置完成的 ValueMapBuilder 实例
    """
    builder = ValueMapBuilder(config=config or ValueMapConfig())
    
    # 解析并添加假设
    hypotheses = parse_hypothesis_output(llm_hypotheses_response)
    for hyp in hypotheses:
        builder.add_hypothesis(hyp)
    
    # 设置搜索进度
    builder.set_search_progress(
        discovered_count=int(discovery_rate * 10),
        total_searchable_area=10,
        explored_regions=explored_regions or [],
        unexplored_regions=unexplored_regions or []
    )
    
    return builder


if __name__ == "__main__":
    # 演示代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Multi-Source Semantic Fusion Pipeline Demo")
    print("=" * 60)
    
    # 创建管道
    pipeline = SemanticFusionPipeline(
        target_object='cup',
        target_object_cn='杯子'
    )
    
    # 模拟 4 张图像
    dummy_images = [
        np.random.rand(480, 640, 3) for _ in range(4)
    ]
    
    # 处理观察
    result = pipeline.process_observation(
        images=dummy_images,
        discovered_count=0,
        total_searchable_area=10,
        explored_regions=['kitchen', 'hallway'],
        unexplored_regions=['bedroom', 'bathroom', 'living_room']
    )
    
    # 输出结果
    print("\n=== VLM Analysis ===")
    print(f"Environment: {result.vlm_analysis['environment_type']}")
    print(f"Visible clues: {result.vlm_analysis['visible_clues']}")
    
    print("\n=== Generated Hypotheses ===")
    for i, (hyp, weight) in enumerate(zip(result.hypotheses, result.weights)):
        print(f"{i+1}. {hyp.assumption}")
        print(f"   Base Confidence: {hyp.base_confidence:.2f}")
        print(f"   Weight: {weight:.3f}")
        print(f"   Priority: {hyp.priority}")
    
    print("\n=== Fusion Results ===")
    print(f"Primary Recommendation: {result.primary_recommendation}")
    print(f"Confidence Score: {result.confidence_score:.3f}")
    print(f"Semantic Entropy: {result.semantic_entropy:.3f}")
    
    print("\n=== Search Suggestions ===")
    for i, suggestion in enumerate(result.search_suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\n=== Value Map Report ===")
    print(json.dumps(result.value_map_report, indent=2, ensure_ascii=False))
