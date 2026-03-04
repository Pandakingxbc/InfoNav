"""
Value Map Construction Module
为未来探索构建多源语义价值地图

实现公式:
V_semantic(p) = Σ(k=1 to K) w_k * V^(k)_sem(p)

其中:
- V^(k)_sem(p): 第 k 个语义假设对位置 p 的价值评分
- w_k: 第 k 个假设的融合权重（Σw_k = 1）
- K: 假设总数（通常 3-5）
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class SemanticHypothesis:
    """语义假设数据结构"""
    id: int
    assumption: str  # 位置假设
    basis: str  # 逻辑依据
    base_confidence: float  # 基础置信度 (0-1)
    search_area: str  # 推荐搜索区域
    accompanying_features: List[str] = field(default_factory=list)
    priority: int = 1  # 优先级（1=最高）
    value_map_type: str = "semantic"  # semantic|context_based|spatial_correlation
    
    def __post_init__(self):
        assert 0 <= self.base_confidence <= 1, "Confidence must be in [0, 1]"
        assert self.priority >= 1, "Priority must be >= 1"


@dataclass
class ValueMapConfig:
    """价值地图配置"""
    semantic_weight_range: Tuple[float, float] = (0.1, 0.5)  # 单个假设权重范围
    min_confidence: float = 0.2  # 最小置信度约束
    max_confidence: float = 0.95  # 最大置信度约束
    discovery_rate_threshold: float = 0.7  # 发现率阈值，超过此值开始降低权重
    exploration_bias: float = 0.1  # 对未探索区域的偏向


class ValueMapBuilder:
    """价值地图构建器"""
    
    def __init__(self, config: Optional[ValueMapConfig] = None):
        self.config = config or ValueMapConfig()
        self.hypotheses: List[SemanticHypothesis] = []
        self.weights: np.ndarray = None
        self.discovery_rate: float = 0.0
        self.explored_regions: List[str] = []
        self.unexplored_regions: List[str] = []
        
    def add_hypothesis(self, hypothesis: SemanticHypothesis) -> None:
        """添加一个语义假设"""
        self.hypotheses.append(hypothesis)
        
    def add_hypotheses_from_dict(self, hypotheses_data: Dict[str, Any]) -> None:
        """从 LLM 输出的字典添加假设列表"""
        if "hypotheses" in hypotheses_data:
            for h in hypotheses_data["hypotheses"]:
                hyp = SemanticHypothesis(
                    id=h.get("id", len(self.hypotheses) + 1),
                    assumption=h["assumption"],
                    basis=h.get("basis", ""),
                    base_confidence=float(h["confidence"]),
                    search_area=h.get("search_area", ""),
                    accompanying_features=h.get("accompanying_features", []),
                    priority=int(h.get("priority", 1)),
                    value_map_type=h.get("value_map_type", "semantic")
                )
                self.add_hypothesis(hyp)
    
    def set_search_progress(self, 
                          discovered_count: int, 
                          total_searchable_area: int,
                          explored_regions: List[str] = None,
                          unexplored_regions: List[str] = None) -> None:
        """设置搜索进度信息"""
        self.discovery_rate = discovered_count / max(1, total_searchable_area)
        if explored_regions:
            self.explored_regions = explored_regions
        if unexplored_regions:
            self.unexplored_regions = unexplored_regions
    
    def calculate_adaptive_confidence(self, hypothesis: SemanticHypothesis) -> float:
        """
        计算自适应置信度
        
        策略:
        1. 基础置信度为起点
        2. 根据优先级调整（高优先级置信度更稳定）
        3. 根据搜索进度调整：
           - 发现率低 → 鼓励探索其他假设（低优先级置信度上升）
           - 发现率高 → 倾向已验证的假设（低优先级置信度下降）
        """
        base_conf = hypothesis.base_confidence
        
        # 优先级因子（优先级越高，置信度衰减越少）
        priority_factor = 1.0 - (hypothesis.priority - 1) * 0.15
        
        # 搜索进度因子
        # 发现率低：激励低优先级假设（exploration bonus）
        # 发现率高：压低低优先级假设（exploitation focus）
        if self.discovery_rate < self.config.discovery_rate_threshold:
            # 搜索早期：给低优先级更多机会
            discovery_factor = 1.0 + (1 - self.discovery_rate) * 0.2
        else:
            # 搜索后期：聚焦高优先级
            discovery_factor = 0.8 - (self.discovery_rate - self.config.discovery_rate_threshold) * 0.3
        
        # 未探索区域偏向
        unexplored_bonus = self.config.exploration_bias if self.unexplored_regions else 0
        
        adjusted_conf = base_conf * priority_factor * discovery_factor + unexplored_bonus
        
        # 约束在允许范围内
        adjusted_conf = np.clip(
            adjusted_conf,
            self.config.min_confidence,
            self.config.max_confidence
        )
        
        return float(adjusted_conf)
    
    def estimate_weights(self) -> np.ndarray:
        """
        估计多源融合的权重
        
        Returns:
            标准化后的权重数组，长度 = 假设数量
        """
        if not self.hypotheses:
            return np.array([])
        
        # 计算每个假设的自适应置信度
        adjusted_confidences = np.array([
            self.calculate_adaptive_confidence(h) for h in self.hypotheses
        ])
        
        # 基于置信度的权重（较高置信度 → 较高权重）
        raw_weights = adjusted_confidences ** 1.5  # 幂次放大差异
        
        # 约束权重范围
        min_w, max_w = self.config.semantic_weight_range
        raw_weights = np.clip(raw_weights / raw_weights.max(), min_w, max_w)
        
        # 标准化
        self.weights = raw_weights / raw_weights.sum()
        
        return self.weights
    
    def compute_semantic_value(self, 
                              position: np.ndarray,
                              hypothesis_scores: List[float]) -> float:
        """
        计算位置 p 的语义价值评分
        
        Args:
            position: 3D 位置坐标
            hypothesis_scores: 每个假设对该位置的评分列表 (0-1)
        
        Returns:
            综合语义价值 V_semantic(p)
        """
        if self.weights is None:
            self.estimate_weights()
        
        if len(hypothesis_scores) != len(self.hypotheses):
            raise ValueError("hypothesis_scores length mismatch")
        
        value = float(np.dot(self.weights, hypothesis_scores))
        return np.clip(value, 0, 1)
    
    def generate_value_map_report(self) -> Dict[str, Any]:
        """生成价值地图构建的详细报告"""
        if self.weights is None:
            self.estimate_weights()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_hypotheses": len(self.hypotheses),
            "discovery_rate": float(self.discovery_rate),
            "explored_regions": self.explored_regions,
            "unexplored_regions": self.unexplored_regions,
            "hypotheses_detail": [
                {
                    "id": h.id,
                    "assumption": h.assumption,
                    "base_confidence": float(h.base_confidence),
                    "adjusted_confidence": float(self.calculate_adaptive_confidence(h)),
                    "weight": float(self.weights[i]) if self.weights is not None else 0,
                    "priority": h.priority,
                    "value_map_type": h.value_map_type,
                    "accompanying_features": h.accompanying_features
                }
                for i, h in enumerate(self.hypotheses)
            ],
            "weights_summary": {
                "weights": [float(w) for w in self.weights] if self.weights is not None else [],
                "sum": float(self.weights.sum()) if self.weights is not None else 0,
                "top_hypothesis": self.hypotheses[np.argmax(self.weights)].assumption if self.weights is not None else None,
                "entropy": float(-np.sum(self.weights * np.log(self.weights + 1e-10))) if self.weights is not None else 0
            },
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """生成搜索建议"""
        if not self.hypotheses:
            return {}
        
        weights = self.weights if self.weights is not None else np.ones(len(self.hypotheses))
        top_idx = np.argmax(weights)
        top_hyp = self.hypotheses[top_idx]
        
        recommendations = {
            "primary_search_area": top_hyp.search_area,
            "primary_assumption": top_hyp.assumption,
            "confidence": float(weights[top_idx]),
            "alternative_areas": [
                self.hypotheses[i].search_area 
                for i in np.argsort(weights)[::-1][1:3]
            ],
            "exploration_suggestion": "Focus on unexplored regions" 
                if self.discovery_rate < 0.5 
                else "Refine search in high-confidence areas"
        }
        
        return recommendations
    
    def save_report(self, output_path: Path) -> None:
        """保存报告到文件"""
        report = self.generate_value_map_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================================
# 集成函数：从 VLM 分析到 Value Map
# ============================================================================

def build_value_map_from_vlm_output(
    vlm_analysis_text: str,
    llm_hypotheses_response: str,
    target_object: str,
    discovery_rate: float = 0.0,
    explored_regions: List[str] = None,
    unexplored_regions: List[str] = None
) -> ValueMapBuilder:
    """
    端到端函数：从 VLM 分析和 LLM 假设生成完整的价值地图
    
    Args:
        vlm_analysis_text: VLM 分析结果文本
        llm_hypotheses_response: LLM 返回的假设 JSON 字符串
        target_object: 目标物体名
        discovery_rate: 当前发现率 (0-1)
        explored_regions: 已探索区域列表
        unexplored_regions: 未探索区域列表
    
    Returns:
        构建好的 ValueMapBuilder 实例
    """
    # 解析 LLM 假设响应
    try:
        hypotheses_data = json.loads(llm_hypotheses_response)
    except json.JSONDecodeError:
        # 如果 JSON 解析失败，尝试提取
        import re
        json_match = re.search(r'\{.*\}', llm_hypotheses_response, re.DOTALL)
        if json_match:
            hypotheses_data = json.loads(json_match.group())
        else:
            raise ValueError("Cannot parse LLM hypotheses response")
    
    # 创建构建器
    builder = ValueMapBuilder()
    builder.add_hypotheses_from_dict(hypotheses_data)
    builder.set_search_progress(
        discovered_count=int(discovery_rate * 100),
        total_searchable_area=100,
        explored_regions=explored_regions or [],
        unexplored_regions=unexplored_regions or []
    )
    
    # 估计权重
    builder.estimate_weights()
    
    return builder


# ============================================================================
# 示例和测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Value Map Builder - Demonstration")
    print("=" * 80)
    
    # 创建示例假设
    builder = ValueMapBuilder()
    
    hypotheses = [
        SemanticHypothesis(
            id=1,
            assumption="杯子在厨房台面上",
            basis="厨房是杯子最常见的位置",
            base_confidence=0.85,
            search_area="kitchen countertop",
            accompanying_features=["water tap", "dishes"],
            priority=1,
            value_map_type="semantic"
        ),
        SemanticHypothesis(
            id=2,
            assumption="杯子在饭桌上",
            basis="人类活动中心",
            base_confidence=0.72,
            search_area="dining table",
            accompanying_features=["chairs", "food"],
            priority=2,
            value_map_type="spatial_correlation"
        ),
        SemanticHypothesis(
            id=3,
            assumption="杯子在柜子或架子上",
            basis="存储位置",
            base_confidence=0.58,
            search_area="upper cabinets",
            accompanying_features=["shelving"],
            priority=3,
            value_map_type="context_based"
        ),
    ]
    
    for h in hypotheses:
        builder.add_hypothesis(h)
    
    # 设置搜索进度（未探索）
    builder.set_search_progress(
        discovered_count=0,
        total_searchable_area=10,
        explored_regions=["kitchen", "entry"],
        unexplored_regions=["living_room", "bedroom"]
    )
    
    # 估计权重
    weights = builder.estimate_weights()
    print("\n【权重估计 - 早期探索阶段】")
    print(f"发现率: {builder.discovery_rate:.1%}")
    for i, (h, w) in enumerate(zip(builder.hypotheses, weights)):
        adj_conf = builder.calculate_adaptive_confidence(h)
        print(f"  假设 {h.id}: {h.assumption}")
        print(f"    基础置信度: {h.base_confidence:.3f} → 调整后: {adj_conf:.3f}")
        print(f"    权重: {w:.3f}")
    
    # 模拟搜索进度，重新估计
    print("\n【权重重估 - 搜索进度 60%】")
    builder.set_search_progress(
        discovered_count=6,
        total_searchable_area=10,
        explored_regions=["kitchen", "entry", "dining_area", "living_room"],
        unexplored_regions=["bedroom", "bathroom"]
    )
    weights = builder.estimate_weights()
    for i, (h, w) in enumerate(zip(builder.hypotheses, weights)):
        adj_conf = builder.calculate_adaptive_confidence(h)
        print(f"  假设 {h.id}: {h.assumption}")
        print(f"    调整后置信度: {adj_conf:.3f}")
        print(f"    权重: {w:.3f}")
    
    # 生成报告
    print("\n【价值地图报告】")
    report = builder.generate_value_map_report()
    print(json.dumps(report, ensure_ascii=False, indent=2)[:800] + "...\n")
    
    print("=" * 80)
