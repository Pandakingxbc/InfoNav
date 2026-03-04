#!/usr/bin/env python3
"""
LLM 语义假说分析工具 (Unified Version)
=====================================

整合了单任务分析、批量分析和重试缺失任务功能，基于VLM分析结果调用LLM生成导航假说和方向引导。
输出结果用于多源Value Map构建。

功能模式:
---------
1. 单任务模式 (single): 分析单个任务
2. 批量模式 (batch): 批量分析多个任务
3. 重试模式 (retry): 重试缺失或失败的任务

使用示例:
---------
# 单任务分析
python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0
python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0 --api-url http://localhost:11434/api/chat

# 批量分析
python llm_hypothesis_analyzer.py batch --env hm3dv2
python llm_hypothesis_analyzer.py batch --env hm3dv2 --max-tasks 10
python llm_hypothesis_analyzer.py batch --env hm3dv2 --start 5 --end 15
python llm_hypothesis_analyzer.py batch --env hm3dv1 hm3dv2 --model qwen2:32b

# 重试缺失任务
python llm_hypothesis_analyzer.py retry --env hm3dv2                    # 自动扫描缺失任务
python llm_hypothesis_analyzer.py retry --env hm3dv2 --tasks 100 200 300  # 指定任务列表
python llm_hypothesis_analyzer.py retry --env hm3dv2 --start 100 --end 200  # 指定范围

输出文件:
---------
- llm_hypothesis_analysis.json: 结构化JSON结果
- llm_hypothesis_analysis.txt: 可读文本报告
- batch_llm_hypothesis_summary.txt: 批量分析摘要（仅批量模式）
- retry_log_YYYYMMDD_HHMMSS.json: 重试日志（仅重试模式）

依赖:
-----
- 需要先运行VLM分析生成 vlm_analysis.txt
- 需要LLM服务（如Ollama）运行在指定端口

作者: yangzhi
日期: 2024-12
"""

import json
import requests
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re


# ==================== 数据类定义 ====================

@dataclass
class EnvironmentAnalysis:
    """环境分析"""
    building_type: str  # residential/office/museum/hospital/school/other
    current_area: str
    global_layout_inference: str


@dataclass
class TargetLocationReasoning:
    """目标位置推理"""
    typical_locations: str
    likely_path: str
    key_landmarks: str


@dataclass
class SemanticHypothesis:
    """
    语义假说 - 对应 HSVM (Hierarchical Semantic Value Map) 的四层结构

    hypothesis_type 类型说明 (对应 main.tex Section III-C):
    - room_type: 房间类型层 (如 bedroom, kitchen)
    - target_object: 目标物体层 (目标物体可能出现的位置描述)
    - co_occurrence: 共现物体层 (与目标共现的物体)
    - object_part: 物体部件层 (目标物体的部件/组件)
    """
    id: int
    hypothesis_type: str  # room_type/target_object/co_occurrence/object_part
    description: str  # 用于 CLIP/BLIP2 匹配的文本描述
    reasoning: str
    confidence: float  # Co-occurrence Confidence C_l (0.0-1.0)
    navigation_value: float  # Spatial Influence Weight W_l (0.0-1.0)

    @property
    def weight(self) -> float:
        """假说权重 = C_l × W_l"""
        return self.confidence * self.navigation_value

    def to_habitat_dict(self) -> dict:
        """
        转换为 habitat_evaluation.py 兼容的字典格式

        Returns:
            dict: 包含 'type', 'prompt', 'weight' 等兼容字段
        """
        return {
            'id': self.id,
            'type': self.hypothesis_type,  # habitat_evaluation 期望 'type' 而非 'hypothesis_type'
            'hypothesis_type': self.hypothesis_type,  # 保持向后兼容
            'prompt': self.description,  # habitat_evaluation 期望 'prompt' 而非 'description'
            'description': self.description,  # 保持向后兼容
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'navigation_value': self.navigation_value,
            'weight': self.weight  # 预计算 weight 供 habitat_evaluation 直接使用
        }


@dataclass
class NavigationAnalysis:
    """完整的导航分析结果"""
    environment_analysis: EnvironmentAnalysis
    target_location_reasoning: TargetLocationReasoning
    semantic_hypotheses: List[SemanticHypothesis]
    exploration_strategy: Optional[str]

    def to_dict(self) -> dict:
        """
        转换为字典 (使用 habitat_evaluation 兼容格式)

        输出的 semantic_hypotheses 包含以下兼容字段:
        - type: hypothesis_type 的别名 (habitat_evaluation 期望)
        - prompt: description 的别名 (habitat_evaluation 期望)
        - weight: 预计算的 confidence × navigation_value
        """
        return {
            'environment_analysis': asdict(self.environment_analysis),
            'target_location_reasoning': asdict(self.target_location_reasoning),
            'semantic_hypotheses': [h.to_habitat_dict() for h in self.semantic_hypotheses],
            'exploration_strategy': self.exploration_strategy
        }

    @classmethod
    def from_json(cls, data: dict) -> 'NavigationAnalysis':
        """从JSON解析"""
        # 解析环境分析
        env_data = data.get('environment_analysis', {})
        environment = EnvironmentAnalysis(
            building_type=env_data.get('building_type', 'unknown'),
            current_area=env_data.get('current_area', ''),
            global_layout_inference=env_data.get('global_layout_inference', '')
        )

        # 解析目标位置推理
        target_data = data.get('target_location_reasoning', {})
        target_reasoning = TargetLocationReasoning(
            typical_locations=target_data.get('typical_locations', ''),
            likely_path=target_data.get('likely_path', ''),
            key_landmarks=target_data.get('key_landmarks', '')
        )

        # 解析假说列表
        hypotheses = [
            SemanticHypothesis(
                id=h.get('id', i+1),
                hypothesis_type=h.get('hypothesis_type', 'unknown'),
                description=h.get('description', ''),
                reasoning=h.get('reasoning', ''),
                confidence=h.get('confidence', 0.5),
                navigation_value=h.get('navigation_value', 0.5)
            )
            for i, h in enumerate(data.get('semantic_hypotheses', []))
        ]

        return cls(
            environment_analysis=environment,
            target_location_reasoning=target_reasoning,
            semantic_hypotheses=hypotheses,
            exploration_strategy=data.get('exploration_strategy')
        )


# ==================== 目标物体配置 ====================

TARGET_CONFIGS = {
    "tv": {
        "part_examples": "black/dark screen, TV stand, thin rectangular display, remote control, cable box",
        "cooccur_examples": "sofa, couch, coffee table, TV cabinet, entertainment center, speakers",
        "room_examples": "living room, bedroom, family room"
    },
    "toilet": {
        "part_examples": "white porcelain bowl, toilet seat, flush handle/button, water tank",
        "cooccur_examples": "sink, bathtub, shower, towel rack, toilet paper holder, bathroom tiles",
        "room_examples": "bathroom, restroom"
    },
    "bed": {
        "part_examples": "mattress, pillows, bedframe, headboard, bedsheets, blanket, bed frame",
        "cooccur_examples": "nightstand, dresser, wardrobe, bedside lamp, alarm clock, window",
        "room_examples": "bedroom, guest room, master bedroom"
    },
    "chair": {
        "part_examples": "seat, backrest, legs, armrest, cushion, seat pad",
        "cooccur_examples": "desk, table, computer, bookshelf, office",
        "room_examples": "office, living room, dining room"
    },
    "refrigerator": {
        "part_examples": "large rectangular body, door handle, freezer compartment, stainless steel",
        "cooccur_examples": "kitchen counter, stove, microwave, cabinets, sink",
        "room_examples": "kitchen"
    },
    "sofa": {
        "part_examples": "cushions, armrests, backrest, fabric/leather surface, seat cushions",
        "cooccur_examples": "coffee table, TV, rug, floor lamp, pillows",
        "room_examples": "living room, family room"
    },
    "couch": {
        "part_examples": "cushions, armrests, backrest, fabric/leather surface, seat cushions",
        "cooccur_examples": "coffee table, TV, rug, floor lamp, pillows",
        "room_examples": "living room, family room"
    },
    "sink": {
        "part_examples": "basin, faucet, drain, countertop, white ceramic",
        "cooccur_examples": "toilet, bathtub, shower, mirror, soap dispenser, towel rack",
        "room_examples": "bathroom, kitchen"
    },
    "table": {
        "part_examples": "flat top surface, legs, wooden/metal frame",
        "cooccur_examples": "chairs, place settings, decorations, lamp",
        "room_examples": "dining room, living room, bedroom"
    }
}


# ==================== LLM Prompt ====================
# 对应 main.tex Section III-C: Hierarchical Semantic Value Map via Micro-Perception (HSVM)
#
# HSVM 四层结构:
# 1. Room Type (房间类型): 当前所在房间类型，空间影响范围最大
# 2. Target Object (目标物体): 目标物体可能出现的位置
# 3. Co-occurrence Objects (共现物体): 与目标物体经常同时出现的物体
# 4. Object Parts (物体部件): 目标物体的部件/组件，空间影响范围最小
#
# 公式: V_HSVM(p) = Σ_{l=1}^{4} C_l × W_l(p) × V_l(p)
# - C_l: Co-occurrence Confidence (confidence 字段)
# - W_l: Spatial Influence Weight (navigation_value 字段)

NAVIGATION_HYPOTHESIS_PROMPT = """You are an expert indoor navigation planner helping a robot find **{target_object}**.

## Current Scene Observation (from VLM):
{vlm_perception}

---

## Your Task: Generate Hierarchical Semantic Hypotheses

Based on the VLM scene description, generate **exactly 4 semantic hypotheses** at different hierarchical levels to help the robot find {target_object}. These hypotheses will be used to build a Hierarchical Semantic Value Map (HSVM) for navigation.

**Key Knowledge about {target_object}:**
- Typical room locations: {room_examples}
- Often found near: {cooccur_examples}
- Visual features/parts: {part_examples}

---

## REQUIRED Hypothesis Types (exactly 4, one for each level):

### Level 1: Room Type (房间类型) - LARGEST spatial influence
The type of room where {target_object} is typically found.
- Examples: "bedroom", "bathroom", "kitchen", "living room"
- This hypothesis has the LARGEST spatial propagation range
- Confidence: How likely is {target_object} to be in this room type?
- Navigation Value: Should be HIGH (0.8-1.0) as rooms are large areas

### Level 2: Target Object (目标物体) - MEDIUM-HIGH spatial influence
The target object itself that the robot is searching for.
- This should be the {target_object} directly (e.g., "bed", "toilet", "TV")
- Confidence: MUST be 1.0 (this IS the target we are looking for)
- Navigation Value: MEDIUM-HIGH (0.6-0.8)

### Level 3: Co-occurrence Objects (共现物体) - MEDIUM spatial influence
Objects that frequently appear together with {target_object}.
- Examples for bed: "nightstand", "dresser", "wardrobe", "bedside lamp"
- Examples for toilet: "sink", "bathtub", "towel rack", "bathroom mirror"
- Seeing these objects suggests {target_object} is nearby
- Confidence: How often does this object appear with {target_object}?
- Navigation Value: MEDIUM (0.4-0.7)

### Level 4: Object Parts (物体部件) - SMALLEST spatial influence
Distinctive parts or features of {target_object} itself.
- Examples for bed: "mattress", "pillows", "headboard", "bed frame"
- Examples for toilet: "toilet seat", "flush handle", "porcelain bowl"
- Only visible when very close to the target
- Confidence: How distinctive is this part?
- Navigation Value: LOW (0.2-0.4) as parts indicate immediate proximity

---

## Scoring Guidelines:

### Confidence (C_l, 0.0-1.0): Co-occurrence likelihood
- 0.8-1.0: Very strong association (e.g., beds are almost always in bedrooms)
- 0.5-0.7: Moderate association (e.g., nightstands are often near beds)
- 0.2-0.4: Weak but possible association

### Navigation Value (W_l, 0.0-1.0): Spatial influence range
- 0.8-1.0: Room-level or larger (room_type)
- 0.6-0.8: Area within room (target_object location)
- 0.4-0.6: Nearby objects (co_occurrence)
- 0.2-0.4: Immediate proximity (object_part)

---

## Output Format (MUST be valid JSON, no markdown):

{{
  "environment_analysis": {{
    "building_type": "residential/office/museum/hospital/school/other",
    "current_area": "Brief description of where robot currently is",
    "global_layout_inference": "Your reasoning about the overall building layout"
  }},

  "target_location_reasoning": {{
    "typical_locations": "Where {target_object} is usually found in this building type",
    "likely_path": "How to navigate from current location to target",
    "key_landmarks": "What to look for along the way"
  }},

  "semantic_hypotheses": [
    {{
      "id": 1,
      "hypothesis_type": "room_type",
      "description": "room name (2-3 words max)",
      "reasoning": "why {target_object} would be in this room",
      "confidence": 0.85,
      "navigation_value": 0.9
    }},
    {{
      "id": 2,
      "hypothesis_type": "target_object",
      "description": "{target_object}",
      "reasoning": "this is the target object we are searching for",
      "confidence": 1.0,
      "navigation_value": 0.7
    }},
    {{
      "id": 3,
      "hypothesis_type": "co_occurrence",
      "description": "co-occurring object name (2-4 words)",
      "reasoning": "this object often appears with {target_object}",
      "confidence": 0.65,
      "navigation_value": 0.5
    }},
    {{
      "id": 4,
      "hypothesis_type": "object_part",
      "description": "part/feature name (2-3 words)",
      "reasoning": "distinctive part of {target_object}",
      "confidence": 0.6,
      "navigation_value": 0.3
    }}
  ],

  "exploration_strategy": "If target not found, describe systematic search approach"
}}

---

## Important Rules:
1. Generate EXACTLY 4 hypotheses, one for each type (room_type, target_object, co_occurrence, object_part)
2. Descriptions must be SHORT (2-5 words) for effective BLIP2/CLIP matching
3. Navigation Value should DECREASE from Level 1 to Level 4 (room > target > co-occur > part)
4. Base your analysis on the VLM scene description provided
5. Output valid JSON only - NO markdown code blocks

Generate your analysis:
"""


# ==================== 核心分析类 ====================

class LLMHypothesisAnalyzer:
    """LLM假说分析器 - 支持 Ollama 和 DeepSeek API"""

    # DeepSeek API 配置
    DEEPSEEK_API_KEY = "sk-c012459659cb402db7a3e9bdfa2a7c62"
    DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self,
                 api_url: str = "https://api.deepseek.com/chat/completions",
                 model: str = "deepseek-chat",
                 env_base_dir: str = "/home/yangz/Nav/ApexNav/env",
                 api_key: Optional[str] = None):
        """
        Args:
            api_url: LLM API地址
            model: 模型名称 (deepseek-chat, deepseek-reasoner, 或 ollama模型名)
            env_base_dir: 环境数据集根目录
            api_key: API密钥 (DeepSeek需要)
        """
        self.api_url = api_url
        self.model = model
        self.env_base_dir = Path(env_base_dir)
        self.api_key = api_key or self.DEEPSEEK_API_KEY

        # 自动检测API类型
        self.is_deepseek = "deepseek" in api_url.lower()

    def load_vlm_analysis(self, vlm_file: Path) -> str:
        """读取VLM分析文件"""
        if not vlm_file.exists():
            raise FileNotFoundError(f"VLM analysis file not found: {vlm_file}")

        with open(vlm_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return content

    def build_prompt(self, target_object: str, vlm_perception: str) -> str:
        """构建LLM prompt"""
        target_lower = target_object.lower()
        config = TARGET_CONFIGS.get(target_lower, {
            "part_examples": f"distinctive parts of {target_object}",
            "cooccur_examples": f"objects commonly found with {target_object}",
            "room_examples": f"rooms where {target_object} is typically found"
        })

        return NAVIGATION_HYPOTHESIS_PROMPT.format(
            target_object=target_object,
            vlm_perception=vlm_perception,
            **config
        )

    def call_llm(self, prompt: str, temperature: float = 0.3, timeout: int = 120) -> str:
        """调用LLM生成假说 - 支持DeepSeek和Ollama"""

        print(f"  调用 LLM: {self.model}")
        print(f"  API: {self.api_url}")
        print(f"  温度: {temperature}")

        if self.is_deepseek:
            return self._call_deepseek(prompt, temperature, timeout)
        else:
            return self._call_ollama(prompt, temperature, timeout)

    def _call_deepseek(self, prompt: str, temperature: float, timeout: int) -> str:
        """调用 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert indoor navigation planner. Always respond with valid JSON only, no markdown formatting."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 2048,
            "stream": False
        }

        try:
            resp = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            if resp.status_code != 200:
                raise RuntimeError(f"DeepSeek API returned {resp.status_code}: {resp.text}")

            result = resp.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            if not content:
                raise RuntimeError("Empty response from DeepSeek")

            return content

        except requests.Timeout:
            raise TimeoutError(f"DeepSeek request timed out after {timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"DeepSeek request failed: {e}")

    def _call_ollama(self, prompt: str, temperature: float, timeout: int) -> str:
        """调用 Ollama API"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_k": 40,
                "top_p": 0.9
            }
        }

        try:
            resp = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout
            )

            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API returned {resp.status_code}: {resp.text}")

            result = resp.json()
            content = result.get('message', {}).get('content', '')

            if not content:
                raise RuntimeError("Empty response from Ollama")

            return content

        except requests.Timeout:
            raise TimeoutError(f"Ollama request timed out after {timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"LLM request failed: {e}")

    def parse_response(self, response: str) -> NavigationAnalysis:
        """解析LLM响应为结构化数据"""
        # 清理可能的markdown代码块标记
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        # 尝试提取JSON
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, cleaned_response)

        json_str = None
        for match in matches:
            try:
                data = json.loads(match)
                # 检查必要的字段 - 新格式
                if 'semantic_hypotheses' in data:
                    json_str = match
                    break
            except json.JSONDecodeError:
                continue

        if not json_str:
            # 最后尝试全文
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to extract valid JSON from LLM response: {cleaned_response[:200]}...")
        else:
            data = json.loads(json_str)

        # 验证必要字段 - 新格式
        required_fields = ['semantic_hypotheses']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return NavigationAnalysis.from_json(data)

    def get_target_object(self, task_dir: Path) -> Optional[str]:
        """从task_info.txt获取目标物体"""
        task_info_file = task_dir / "task_info.txt"

        if not task_info_file.exists():
            return None

        with open(task_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'target object' in line.lower():
                    return line.split(':', 1)[1].strip()

        return None

    def analyze_single(self,
                       env_name: str,
                       task_num: int,
                       temperature: float = 0.3) -> Tuple[bool, Optional[NavigationAnalysis], Optional[str]]:
        """
        分析单个任务

        Args:
            env_name: 环境名称
            task_num: 任务编号
            temperature: 生成温度（0-1）

        Returns:
            (success, analysis, error_msg)
        """
        task_dir = self.env_base_dir / f"env_{env_name}" / f"task{task_num}"

        print(f"\n{'='*70}")
        print(f"LLM 假说分析")
        print(f"{'='*70}")
        print(f"任务目录: {task_dir}")

        # 检查任务目录
        if not task_dir.exists():
            return False, None, f"任务目录不存在: {task_dir}"

        # 获取目标物体
        target_object = self.get_target_object(task_dir)
        if not target_object:
            return False, None, "无法从task_info.txt获取目标物体"

        print(f"目标物体: {target_object}")

        try:
            # 1. 加载VLM分析
            print(f"\n[1/4] 加载VLM分析...")
            vlm_file = task_dir / "vlm_analysis.txt"
            vlm_perception = self.load_vlm_analysis(vlm_file)
            print(f"  已加载VLM分析 ({len(vlm_perception)} 字符)")

            # 2. 构建Prompt
            print(f"\n[2/4] 构建分析Prompt...")
            prompt = self.build_prompt(target_object, vlm_perception)
            print(f"  Prompt长度: {len(prompt)} 字符")

            # 3. 调用LLM
            print(f"\n[3/4] 调用LLM生成假说...")
            t0 = time.time()
            response = self.call_llm(prompt, temperature=temperature)
            t1 = time.time()
            print(f"  耗时: {t1-t0:.2f}秒")
            print(f"  响应长度: {len(response)} 字符")

            # 4. 解析响应
            print(f"\n[4/4] 解析LLM响应...")
            analysis = self.parse_response(response)
            print(f"  环境类型: {analysis.environment_analysis.building_type}")
            print(f"  生成假说数: {len(analysis.semantic_hypotheses)}")

            return True, analysis, None

        except Exception as e:
            return False, None, str(e)


# ==================== 输出格式化 ====================

class AnalysisReportWriter:
    """分析结果报告编写器"""

    @staticmethod
    def write_json(output_file: Path, analysis: NavigationAnalysis) -> None:
        """写入JSON格式结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def write_text_report(output_file: Path, analysis: NavigationAnalysis) -> None:
        """写入文本格式报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("全局导航假说分析报告\n")
            f.write("="*70 + "\n\n")

            # 环境分析
            f.write("【环境分析】\n")
            env = analysis.environment_analysis
            f.write(f"建筑类型: {env.building_type}\n")
            f.write(f"当前位置: {env.current_area}\n")
            f.write(f"全局布局推理: {env.global_layout_inference}\n\n")

            # 目标位置推理
            f.write("【目标位置推理】\n")
            target = analysis.target_location_reasoning
            f.write(f"典型位置: {target.typical_locations}\n")
            f.write(f"可能路径: {target.likely_path}\n")
            f.write(f"关键路标: {target.key_landmarks}\n\n")

            # 语义假说
            f.write("【语义假说】\n")
            f.write(f"{'ID':<3} {'类型':<18} {'描述':<25} {'C':<6} {'V':<6} {'权重':<8}\n")
            f.write("-"*70 + "\n")

            # HSVM 四层结构类型映射 (对应 main.tex Section III-C)
            type_map = {
                # 新格式 (HSVM 四层结构)
                'room_type': 'L1-房间类型',
                'target_object': 'L2-目标物体',
                'co_occurrence': 'L3-共现物体',
                'object_part': 'L4-物体部件',
                # 旧格式兼容
                'building_layout': '建筑布局',
                'target_room': '目标房间',
                'adjacent_area': '相邻区域',
                'indicator_object': '指示物体',
                'target_feature': '目标特征',
                'spatial_navigation': '空间导航',
                'room_context': '房间类型',
                'part_attribute': '部件属性'
            }

            for h in analysis.semantic_hypotheses:
                type_str = type_map.get(h.hypothesis_type, h.hypothesis_type)
                desc = h.description[:25].ljust(25)
                f.write(f"{h.id:<3} {type_str:<18} {desc} "
                       f"{h.confidence:<6.2f} {h.navigation_value:<6.2f} {h.weight:<8.4f}\n")
                f.write(f"    推理: {h.reasoning}\n")

            f.write("\n")

            # 探索策略
            if analysis.exploration_strategy:
                f.write("【探索策略】\n")
                f.write(analysis.exploration_strategy + "\n\n")

            # 统计信息
            f.write("【统计信息】\n")
            if analysis.semantic_hypotheses:
                avg_conf = sum(h.confidence for h in analysis.semantic_hypotheses) / len(analysis.semantic_hypotheses)
                avg_nav = sum(h.navigation_value for h in analysis.semantic_hypotheses) / len(analysis.semantic_hypotheses)
                avg_weight = sum(h.weight for h in analysis.semantic_hypotheses) / len(analysis.semantic_hypotheses)
                f.write(f"假说数量: {len(analysis.semantic_hypotheses)}\n")
                f.write(f"平均置信度: {avg_conf:.3f}\n")
                f.write(f"平均导航价值: {avg_nav:.3f}\n")
                f.write(f"平均权重(C×V): {avg_weight:.4f}\n")

            f.write("\n" + "="*70 + "\n")


# ==================== 批量处理类 ====================

class BatchAnalyzer:
    """批量分析器"""

    def __init__(self, analyzer: LLMHypothesisAnalyzer, delay: int = 10):
        """
        Args:
            analyzer: LLM假说分析器实例
            delay: 任务间隔时间（秒）
        """
        self.analyzer = analyzer
        self.delay = delay

    def discover_tasks(self, env_name: str) -> List[int]:
        """发现指定环境下所有有VLM分析的任务"""
        env_dir = self.analyzer.env_base_dir / f"env_{env_name}"

        if not env_dir.exists():
            print(f"警告: 环境目录不存在: {env_dir}")
            return []

        tasks = []
        for task_dir in sorted(env_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith('task'):
                try:
                    task_num = int(task_dir.name[4:])
                    # 检查是否已有VLM分析
                    if (task_dir / "vlm_analysis.txt").exists():
                        tasks.append(task_num)
                except ValueError:
                    continue

        return sorted(tasks)

    def analyze_batch(self,
                      env_name: str,
                      start_task: Optional[int] = None,
                      end_task: Optional[int] = None,
                      max_tasks: Optional[int] = None,
                      temperature: float = 0.3,
                      output_format: str = 'both') -> Dict:
        """
        批量分析环境数据集

        Args:
            env_name: 环境名称
            start_task: 起始任务编号（包含）
            end_task: 结束任务编号（包含）
            max_tasks: 最大任务数量限制
            temperature: 生成温度
            output_format: 输出格式 ('json', 'text', 'both')

        Returns:
            统计信息字典
        """
        print(f"\n{'#'*70}")
        print(f"# 批量假说分析: env_{env_name}")
        print(f"# LLM模型: {self.analyzer.model}")
        print(f"# API: {self.analyzer.api_url}")
        print(f"# 任务间隔: {self.delay} 秒")
        print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")

        # 发现所有任务
        all_tasks = self.discover_tasks(env_name)

        if not all_tasks:
            print(f"错误: 在 env_{env_name} 中未找到任何有VLM分析的任务")
            return None

        print(f"发现 {len(all_tasks)} 个任务")

        # 应用任务范围过滤
        if start_task is not None:
            all_tasks = [t for t in all_tasks if t >= start_task]
        if end_task is not None:
            all_tasks = [t for t in all_tasks if t <= end_task]
        if max_tasks is not None:
            all_tasks = all_tasks[:max_tasks]

        print(f"将分析 {len(all_tasks)} 个任务: {all_tasks[:10]}{'...' if len(all_tasks) > 10 else ''}\n")

        # 统计信息
        stats = {
            'total': len(all_tasks),
            'success': 0,
            'failed': 0,
            'total_time': 0,
            'results': []
        }

        writer = AnalysisReportWriter()

        # 逐个分析任务
        for idx, task_num in enumerate(all_tasks, 1):
            print(f"\n进度: [{idx}/{len(all_tasks)}]")

            t0 = time.time()
            success, analysis, error_msg = self.analyzer.analyze_single(
                env_name, task_num, temperature=temperature
            )
            duration = time.time() - t0

            result_file = None
            if success and analysis:
                task_dir = self.analyzer.env_base_dir / f"env_{env_name}" / f"task{task_num}"

                if output_format in ['json', 'both']:
                    json_file = task_dir / "llm_hypothesis_analysis.json"
                    writer.write_json(json_file, analysis)
                    result_file = str(json_file)
                    print(f"  JSON结果: {json_file}")

                if output_format in ['text', 'both']:
                    text_file = task_dir / "llm_hypothesis_analysis.txt"
                    writer.write_text_report(text_file, analysis)
                    print(f"  文本报告: {text_file}")

            stats['results'].append({
                'task': task_num,
                'success': success,
                'duration': duration,
                'error': error_msg,
                'result_file': result_file
            })

            stats['total_time'] += duration

            if success:
                stats['success'] += 1
                print(f"  成功 (耗时: {duration:.2f}秒)")
            else:
                stats['failed'] += 1
                print(f"  失败: {error_msg}")

            # 如果不是最后一个任务，等待指定时间
            if idx < len(all_tasks):
                print(f"\n等待 {self.delay} 秒后继续...")
                for remaining in range(self.delay, 0, -1):
                    print(f"  剩余: {remaining} 秒", end='\r')
                    time.sleep(1)
                print()

        return stats

    def save_summary(self, env_name: str, stats: Dict) -> None:
        """保存批量分析摘要"""
        summary_file = self.analyzer.env_base_dir / f"env_{env_name}" / "batch_llm_hypothesis_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"批量LLM假说分析摘要: env_{env_name}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LLM模型: {self.analyzer.model}\n")
            f.write(f"总任务数: {stats['total']}\n")
            f.write(f"成功: {stats['success']}\n")
            f.write(f"失败: {stats['failed']}\n")
            f.write(f"成功率: {stats['success']/stats['total']*100:.2f}%\n")
            f.write(f"总耗时: {stats['total_time']:.2f} 秒 ({stats['total_time']/60:.2f} 分钟)\n")
            f.write(f"平均耗时: {stats['total_time']/stats['total']:.2f} 秒/任务\n\n")

            f.write(f"详细结果:\n")
            f.write(f"{'-'*70}\n")

            for result in stats['results']:
                status = "OK" if result['success'] else "FAIL"
                f.write(f"[{status}] task{result['task']}: {result['duration']:.2f}s")
                if result['error']:
                    f.write(f" - {result['error']}")
                f.write("\n")

        print(f"\n摘要已保存到: {summary_file}")

    def print_summary(self, env_name: str, stats: Dict) -> None:
        """打印分析摘要"""
        print(f"\n{'#'*70}")
        print(f"# 分析完成: env_{env_name}")
        print(f"{'#'*70}\n")

        print(f"总任务数: {stats['total']}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"成功率: {stats['success']/stats['total']*100:.2f}%")
        print(f"总耗时: {stats['total_time']:.2f} 秒 ({stats['total_time']/60:.2f} 分钟)")
        print(f"平均耗时: {stats['total_time']/stats['total']:.2f} 秒/任务")

        if stats['failed'] > 0:
            print(f"\n失败的任务:")
            for result in stats['results']:
                if not result['success']:
                    print(f"  - task{result['task']}: {result['error']}")


# ==================== 重试分析类 ====================

class RetryAnalyzer:
    """重试缺失任务分析器"""

    def __init__(self, analyzer: LLMHypothesisAnalyzer, delay: int = 40):
        """
        Args:
            analyzer: LLM假说分析器实例
            delay: 任务间隔时间（秒）
        """
        self.analyzer = analyzer
        self.delay = delay

    def find_missing_tasks(self, env_name: str,
                           start_task: Optional[int] = None,
                           end_task: Optional[int] = None) -> List[int]:
        """
        查找缺失LLM分析结果的任务

        Args:
            env_name: 环境名称
            start_task: 起始任务编号
            end_task: 结束任务编号

        Returns:
            缺失任务编号列表
        """
        env_dir = self.analyzer.env_base_dir / f"env_{env_name}"

        if not env_dir.exists():
            print(f"警告: 环境目录不存在: {env_dir}")
            return []

        missing_tasks = []
        for task_dir in sorted(env_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith('task'):
                try:
                    task_num = int(task_dir.name[4:])

                    # 应用范围过滤
                    if start_task is not None and task_num < start_task:
                        continue
                    if end_task is not None and task_num > end_task:
                        continue

                    # 检查是否有VLM分析但缺少LLM分析
                    vlm_file = task_dir / "vlm_analysis.txt"
                    json_file = task_dir / "llm_hypothesis_analysis.json"

                    if vlm_file.exists() and not json_file.exists():
                        missing_tasks.append(task_num)

                except ValueError:
                    continue

        return sorted(missing_tasks)

    def retry_tasks(self,
                    env_name: str,
                    tasks: List[int],
                    temperature: float = 0.3,
                    output_format: str = 'both',
                    skip_exists: bool = False) -> Dict:
        """
        重试指定的任务列表

        Args:
            env_name: 环境名称
            tasks: 任务编号列表
            temperature: 生成温度
            output_format: 输出格式
            skip_exists: 如果结果已存在是否跳过

        Returns:
            统计信息字典
        """
        print("=" * 80)
        print(f"重试缺失的 LLM 分析任务")
        print("=" * 80)
        print(f"环境: {env_name}")
        print(f"需要处理的任务数: {len(tasks)}")
        if tasks:
            print(f"任务范围: task{min(tasks)} ~ task{max(tasks)}")
        print(f"LLM API: {self.analyzer.api_url}")
        print(f"模型: {self.analyzer.model}")
        print(f"任务间隔: {self.delay} 秒")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        if not tasks:
            print("没有找到需要处理的任务")
            return None

        # 统计信息
        stats = {
            'total': len(tasks),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0,
            'results': [],
            'failed_tasks': []
        }

        writer = AnalysisReportWriter()

        for idx, task_num in enumerate(tasks, 1):
            task_dir = self.analyzer.env_base_dir / f"env_{env_name}" / f"task{task_num}"
            json_file = task_dir / "llm_hypothesis_analysis.json"

            # 检查是否应该跳过
            if skip_exists and json_file.exists():
                print(f"[{idx}/{len(tasks)}] SKIP task{task_num}: 已存在")
                stats['skipped'] += 1
                if idx < len(tasks):
                    time.sleep(self.delay)
                continue

            print(f"[{idx}/{len(tasks)}] 处理 task{task_num}...", end=' ', flush=True)

            t0 = time.time()
            success, analysis, error_msg = self.analyzer.analyze_single(
                env_name, task_num, temperature=temperature
            )
            duration = time.time() - t0

            stats['total_time'] += duration

            if success and analysis:
                if output_format in ['json', 'both']:
                    writer.write_json(json_file, analysis)

                if output_format in ['text', 'both']:
                    text_file = task_dir / "llm_hypothesis_analysis.txt"
                    writer.write_text_report(text_file, analysis)

                stats['success'] += 1
                print(f"OK ({duration:.2f}s)")
            else:
                stats['failed'] += 1
                stats['failed_tasks'].append((task_num, error_msg))
                print(f"FAIL ({duration:.2f}s) - {error_msg[:60] if error_msg else 'Unknown'}")

            stats['results'].append({
                'task': task_num,
                'success': success,
                'duration': duration,
                'error': error_msg
            })

            # 任务间隔
            if idx < len(tasks):
                print(f"  等待 {self.delay} 秒...", end=' ', flush=True)
                time.sleep(self.delay)
                print("完成")

        return stats

    def save_retry_log(self, env_name: str, stats: Dict) -> Path:
        """保存重试日志"""
        log_file = self.analyzer.env_base_dir / f"env_{env_name}" / f"retry_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'env': env_name,
            'model': self.analyzer.model,
            'api_url': self.analyzer.api_url,
            'total_tasks': stats['total'],
            'successful': stats['success'],
            'failed': stats['failed'],
            'skipped': stats['skipped'],
            'total_time': stats['total_time'],
            'failed_tasks': [{'task_id': tid, 'error': err} for tid, err in stats['failed_tasks']],
            'results': stats['results']
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"\n日志已保存: {log_file}")
        return log_file

    def print_retry_summary(self, stats: Dict) -> None:
        """打印重试摘要"""
        print("\n" + "=" * 80)
        print("处理完成")
        print("=" * 80)
        print(f"总任务数: {stats['total']}")
        print(f"成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "成功: 0")
        print(f"失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "失败: 0")
        print(f"跳过: {stats['skipped']}")
        print(f"总耗时: {stats['total_time']:.1f} 秒")
        if stats['total'] > 0:
            print(f"平均耗时: {stats['total_time']/stats['total']:.1f} 秒/任务")

        if stats['failed_tasks']:
            print(f"\n失败的任务 ({len(stats['failed_tasks'])}):")
            for task_id, error in stats['failed_tasks'][:10]:
                print(f"  task{task_id}: {error[:60] if error else 'Unknown error'}")
            if len(stats['failed_tasks']) > 10:
                print(f"  ... 还有 {len(stats['failed_tasks']) - 10} 个失败任务")

        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


# ==================== 命令行接口 ====================

def cmd_single(args):
    """单任务分析命令"""
    analyzer = LLMHypothesisAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir
    )

    success, analysis, error_msg = analyzer.analyze_single(
        args.env, args.task, temperature=args.temperature
    )

    if not success:
        print(f"\n错误: {error_msg}")
        return 1

    # 保存结果
    task_dir = analyzer.env_base_dir / f"env_{args.env}" / f"task{args.task}"
    writer = AnalysisReportWriter()

    print(f"\n{'='*70}")
    print("保存分析结果...")
    print(f"{'='*70}")

    if args.output_format in ['json', 'both']:
        json_file = task_dir / "llm_hypothesis_analysis.json"
        writer.write_json(json_file, analysis)
        print(f"JSON结果: {json_file}")

    if args.output_format in ['text', 'both']:
        text_file = task_dir / "llm_hypothesis_analysis.txt"
        writer.write_text_report(text_file, analysis)
        print(f"文本报告: {text_file}")

    print(f"\n{'='*70}")
    print("分析完成!")
    print(f"{'='*70}\n")

    return 0


def cmd_batch(args):
    """批量分析命令"""
    analyzer = LLMHypothesisAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir
    )

    batch = BatchAnalyzer(analyzer, delay=args.delay)

    all_stats = {}

    for env_name in args.env:
        stats = batch.analyze_batch(
            env_name,
            start_task=args.start,
            end_task=args.end,
            max_tasks=args.max_tasks,
            temperature=args.temperature,
            output_format=args.output_format
        )

        if stats:
            all_stats[env_name] = stats
            batch.print_summary(env_name, stats)
            batch.save_summary(env_name, stats)

            # 如果不是最后一个环境，等待
            if env_name != args.env[-1]:
                print(f"\n{'='*70}")
                print(f"准备分析下一个环境...")
                print(f"等待 {args.delay} 秒...")
                print(f"{'='*70}\n")
                time.sleep(args.delay)

    # 打印总体摘要
    if len(all_stats) > 1:
        print(f"\n{'#'*70}")
        print(f"# 总体摘要")
        print(f"{'#'*70}\n")

        total_tasks = sum(s['total'] for s in all_stats.values())
        total_success = sum(s['success'] for s in all_stats.values())
        total_failed = sum(s['failed'] for s in all_stats.values())
        total_time = sum(s['total_time'] for s in all_stats.values())

        print(f"分析环境数: {len(all_stats)}")
        print(f"总任务数: {total_tasks}")
        print(f"总成功: {total_success}")
        print(f"总失败: {total_failed}")
        print(f"总成功率: {total_success/total_tasks*100:.2f}%")
        print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print()

        for env_name, stats in all_stats.items():
            print(f"  {env_name}: {stats['success']}/{stats['total']} "
                  f"({stats['success']/stats['total']*100:.2f}%)")

    print(f"\n{'='*70}")
    print("所有分析任务完成!")
    print(f"{'='*70}\n")

    return 0


def cmd_retry(args):
    """重试缺失任务命令"""
    analyzer = LLMHypothesisAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir
    )

    retry = RetryAnalyzer(analyzer, delay=args.delay)

    # 确定要处理的任务
    if args.tasks:
        # 使用指定的任务列表
        tasks_to_process = sorted(args.tasks)
    else:
        # 自动扫描缺失的任务
        print(f"扫描 env_{args.env} 中缺失LLM分析的任务...")
        tasks_to_process = retry.find_missing_tasks(
            args.env,
            start_task=args.start,
            end_task=args.end
        )

        if not tasks_to_process:
            print("没有找到缺失LLM分析的任务，所有任务都已完成！")
            return 0

        print(f"找到 {len(tasks_to_process)} 个缺失任务: {tasks_to_process[:20]}{'...' if len(tasks_to_process) > 20 else ''}")

    # 执行重试
    stats = retry.retry_tasks(
        args.env,
        tasks_to_process,
        temperature=args.temperature,
        output_format=args.output_format,
        skip_exists=args.skip_exists
    )

    if stats:
        retry.print_retry_summary(stats)
        retry.save_retry_log(args.env, stats)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="LLM 语义假说分析工具 - 支持单任务、批量分析和重试缺失任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
=========

单任务分析:
  python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0
  python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0 --api-url http://localhost:11434/api/chat

批量分析:
  python llm_hypothesis_analyzer.py batch --env hm3dv2
  python llm_hypothesis_analyzer.py batch --env hm3dv2 --max-tasks 10
  python llm_hypothesis_analyzer.py batch --env hm3dv2 --start 5 --end 15
  python llm_hypothesis_analyzer.py batch --env hm3dv1 hm3dv2 --model qwen2:32b

重试缺失任务:
  python llm_hypothesis_analyzer.py retry --env hm3dv2                      # 自动扫描缺失任务
  python llm_hypothesis_analyzer.py retry --env hm3dv2 --tasks 100 200 300  # 指定任务列表
  python llm_hypothesis_analyzer.py retry --env hm3dv2 --start 100 --end 200  # 指定范围
  python llm_hypothesis_analyzer.py retry --env hm3dv2 --skip-exists        # 跳过已存在的
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='运行模式')

    # 单任务模式
    single_parser = subparsers.add_parser('single', help='单任务分析模式')
    single_parser.add_argument('--env', type=str, required=True,
                               choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                               help='环境类别')
    single_parser.add_argument('--task', type=int, required=True,
                               help='任务编号')
    single_parser.add_argument('--api-url', type=str, default='https://api.deepseek.com/chat/completions',
                               help='LLM API 地址 (默认: DeepSeek API)')
    single_parser.add_argument('--model', type=str, default='deepseek-chat',
                               help='LLM 模型名称 (默认: deepseek-chat)')
    single_parser.add_argument('--temperature', type=float, default=0.3,
                               help='生成温度 0.0-1.0 (默认: 0.3)')
    single_parser.add_argument('--env-base-dir', type=str, default='/home/yangz/Nav/InfoNav/env',
                               help='环境数据集根目录')
    single_parser.add_argument('--output-format', type=str, default='both',
                               choices=['json', 'text', 'both'],
                               help='输出格式 (默认: both)')

    # 批量模式
    batch_parser = subparsers.add_parser('batch', help='批量分析模式')
    batch_parser.add_argument('--env', type=str, nargs='+', required=True,
                              choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                              help='要分析的环境（可指定多个）')
    batch_parser.add_argument('--start', type=int, default=None,
                              help='起始任务编号（包含）')
    batch_parser.add_argument('--end', type=int, default=None,
                              help='结束任务编号（包含）')
    batch_parser.add_argument('--max-tasks', type=int, default=None,
                              help='最大任务数量限制')
    batch_parser.add_argument('--delay', type=int, default=5,
                              help='任务间隔时间（秒），默认5秒')
    batch_parser.add_argument('--api-url', type=str, default='https://api.deepseek.com/chat/completions',
                              help='LLM API 地址 (默认: DeepSeek API)')
    batch_parser.add_argument('--model', type=str, default='deepseek-chat',
                              help='LLM 模型名称 (默认: deepseek-chat)')
    batch_parser.add_argument('--temperature', type=float, default=0.2,
                              help='生成温度 0.0-1.0 (默认: 0.2)')
    batch_parser.add_argument('--env-base-dir', type=str, default='/home/yangz/Nav/InfoNav/env',
                              help='环境数据集根目录')
    batch_parser.add_argument('--output-format', type=str, default='both',
                              choices=['json', 'text', 'both'],
                              help='输出格式 (默认: both)')

    # 重试模式
    retry_parser = subparsers.add_parser('retry', help='重试缺失任务模式')
    retry_parser.add_argument('--env', type=str, required=True,
                              choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                              help='环境类别')
    retry_parser.add_argument('--tasks', type=int, nargs='+', default=None,
                              help='指定要重试的任务编号列表（可选，不指定则自动扫描缺失任务）')
    retry_parser.add_argument('--start', type=int, default=None,
                              help='起始任务编号（包含）')
    retry_parser.add_argument('--end', type=int, default=None,
                              help='结束任务编号（包含）')
    retry_parser.add_argument('--delay', type=int, default=5,
                              help='任务间隔时间（秒），默认5秒')
    retry_parser.add_argument('--api-url', type=str, default='https://api.deepseek.com/chat/completions',
                              help='LLM API 地址 (默认: DeepSeek API)')
    retry_parser.add_argument('--model', type=str, default='deepseek-chat',
                              help='LLM 模型名称 (默认: deepseek-chat)')
    retry_parser.add_argument('--temperature', type=float, default=0.3,
                              help='生成温度 0.0-1.0 (默认: 0.3)')
    retry_parser.add_argument('--env-base-dir', type=str, default='/home/yangz/Nav/InfoNav/env',
                              help='环境数据集根目录')
    retry_parser.add_argument('--output-format', type=str, default='both',
                              choices=['json', 'text', 'both'],
                              help='输出格式 (默认: both)')
    retry_parser.add_argument('--skip-exists', action='store_true',
                              help='如果结果已存在则跳过 (默认: 覆盖)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'single':
        return cmd_single(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'retry':
        return cmd_retry(args)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断分析")
        sys.exit(1)
