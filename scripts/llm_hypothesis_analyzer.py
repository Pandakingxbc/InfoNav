#!/usr/bin/env python3
"""
LLM Semantic Hypothesis Analysis Tool (Unified Version)
========================================================

Integrates single-task analysis, batch analysis, and retry-missing-tasks functionality.
Calls LLM based on VLM analysis results to generate navigation hypotheses and directional guidance.
Output is used for multi-source Value Map construction.

Modes:
------
1. Single mode (single): Analyze a single task
2. Batch mode (batch): Batch analyze multiple tasks
3. Retry mode (retry): Retry missing or failed tasks

Usage Examples:
---------------
# Single task analysis
python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0
python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0 --api-url http://localhost:11434/api/chat

# Batch analysis
python llm_hypothesis_analyzer.py batch --env hm3dv2
python llm_hypothesis_analyzer.py batch --env hm3dv2 --max-tasks 10
python llm_hypothesis_analyzer.py batch --env hm3dv2 --start 5 --end 15
python llm_hypothesis_analyzer.py batch --env hm3dv1 hm3dv2 --model qwen2:32b

# Retry missing tasks
python llm_hypothesis_analyzer.py retry --env hm3dv2                      # Auto-scan missing tasks
python llm_hypothesis_analyzer.py retry --env hm3dv2 --tasks 100 200 300  # Specify task list
python llm_hypothesis_analyzer.py retry --env hm3dv2 --start 100 --end 200  # Specify range

Output Files:
-------------
- llm_hypothesis_analysis.json: Structured JSON result
- llm_hypothesis_analysis.txt: Human-readable text report
- batch_llm_hypothesis_summary.txt: Batch analysis summary (batch mode only)
- retry_log_YYYYMMDD_HHMMSS.json: Retry log (retry mode only)

Dependencies:
-------------
- Requires VLM analysis (vlm_analysis.txt) to be generated first
- Requires LLM service (e.g., Ollama) running on specified port

"""

import json
import os
import requests
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re


# ==================== Data Class Definitions ====================

@dataclass
class EnvironmentAnalysis:
    """Environment Analysis"""
    building_type: str  # residential/office/museum/hospital/school/other
    current_area: str
    global_layout_inference: str


@dataclass
class TargetLocationReasoning:
    """Target Location Reasoning"""
    typical_locations: str
    likely_path: str
    key_landmarks: str


@dataclass
class SemanticHypothesis:
    """
    Semantic Hypothesis - Corresponds to the 4-layer structure of HSVM (Hierarchical Semantic Value Map)

    hypothesis_type descriptions (corresponding to main.tex Section III-C):
    - room_type: Room type layer (e.g., bedroom, kitchen)
    - target_object: Target object layer (description of where target object may appear)
    - co_occurrence: Co-occurrence object layer (objects that co-occur with target)
    - object_part: Object part layer (parts/components of target object)
    """
    id: int
    hypothesis_type: str  # room_type/target_object/co_occurrence/object_part
    description: str  # 用于 CLIP/BLIP2 匹配的文本描述
    reasoning: str
    confidence: float  # Co-occurrence Confidence C_l: P(A | B_l) (0.0-1.0)
    navigation_value: float  # Spatial Footprint a_l: area ratio (0.0-1.0), kept as navigation_value for backward compatibility

    @property
    def weight(self) -> float:
        """Hypothesis weight = C_l × a_l (confidence × spatial_footprint)"""
        return self.confidence * self.navigation_value

    def to_habitat_dict(self) -> dict:
        """
        Convert to habitat_evaluation.py compatible dictionary format

        Returns:
            dict: Contains 'type', 'prompt', 'weight' and other compatible fields
        """
        return {
            'id': self.id,
            'type': self.hypothesis_type,  # habitat_evaluation expects 'type' not 'hypothesis_type'
            'hypothesis_type': self.hypothesis_type,  # Keep for backward compatibility
            'prompt': self.description,  # habitat_evaluation expects 'prompt' not 'description'
            'description': self.description,  # Keep for backward compatibility
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'spatial_footprint': self.navigation_value,  # New field name aligned with paper (a_l)
            'navigation_value': self.navigation_value,  # Keep for backward compatibility
            'weight': self.weight  # Pre-computed weight C_l × a_l for direct use by habitat_evaluation
        }


@dataclass
class IGHypothesis:
    """
    Information Gain Hypothesis - Corresponds to main.tex Section III-D / Appendix A.3

    IG hypotheses differ from HSVM hypotheses:
    - HSVM hypotheses: target-dependent (depend on target object)
    - IG hypotheses: target-agnostic (independent of target, focus on connectivity structures)

    structure_type descriptions:
    - corridor: Corridor/hallway
    - doorway: Doorway
    - passage: Passage
    - archway: Archway
    - opening: Opening
    - other: Other
    """
    id: int
    prompt: str  # Text description for BLIP2 matching (2-6 words)
    structure_type: str  # corridor/doorway/passage/archway/opening/other
    reasoning: str

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'prompt': self.prompt,
            'structure_type': self.structure_type,
            'reasoning': self.reasoning
        }


@dataclass
class IGAnalysis:
    """IG Hypothesis Analysis Result"""
    ig_hypotheses: List[IGHypothesis]
    connectivity_summary: str

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'ig_hypotheses': [h.to_dict() for h in self.ig_hypotheses],
            'connectivity_summary': self.connectivity_summary
        }

    @classmethod
    def from_json(cls, data: dict) -> 'IGAnalysis':
        """Parse from JSON"""
        hypotheses = [
            IGHypothesis(
                id=h.get('id', i+1),
                prompt=h.get('prompt', ''),
                structure_type=h.get('structure_type', 'other'),
                reasoning=h.get('reasoning', '')
            )
            for i, h in enumerate(data.get('ig_hypotheses', []))
        ]

        return cls(
            ig_hypotheses=hypotheses,
            connectivity_summary=data.get('connectivity_summary', '')
        )


@dataclass
class NavigationAnalysis:
    """Complete Navigation Analysis Result"""
    environment_analysis: EnvironmentAnalysis
    target_location_reasoning: TargetLocationReasoning
    semantic_hypotheses: List[SemanticHypothesis]
    exploration_strategy: Optional[str]
    ig_hypotheses: Optional[List[IGHypothesis]] = None  # IG hypotheses (optional)
    connectivity_summary: Optional[str] = None  # IG connectivity summary (optional)

    def to_dict(self) -> dict:
        """
        Convert to dictionary (using habitat_evaluation compatible format)

        Output semantic_hypotheses contains the following compatible fields:
        - type: alias for hypothesis_type (expected by habitat_evaluation)
        - prompt: alias for description (expected by habitat_evaluation)
        - weight: pre-computed confidence × navigation_value
        """
        result = {
            'environment_analysis': asdict(self.environment_analysis),
            'target_location_reasoning': asdict(self.target_location_reasoning),
            'semantic_hypotheses': [h.to_habitat_dict() for h in self.semantic_hypotheses],
            'exploration_strategy': self.exploration_strategy
        }

        # Add IG hypotheses (if present)
        if self.ig_hypotheses:
            result['ig_hypotheses'] = [h.to_dict() for h in self.ig_hypotheses]
        if self.connectivity_summary:
            result['connectivity_summary'] = self.connectivity_summary

        return result

    @classmethod
    def from_json(cls, data: dict) -> 'NavigationAnalysis':
        """Parse from JSON"""
        # Parse environment analysis
        env_data = data.get('environment_analysis', {})
        environment = EnvironmentAnalysis(
            building_type=env_data.get('building_type', 'unknown'),
            current_area=env_data.get('current_area', ''),
            global_layout_inference=env_data.get('global_layout_inference', '')
        )

        # Parse target location reasoning
        target_data = data.get('target_location_reasoning', {})
        target_reasoning = TargetLocationReasoning(
            typical_locations=target_data.get('typical_locations', ''),
            likely_path=target_data.get('likely_path', ''),
            key_landmarks=target_data.get('key_landmarks', '')
        )

        # Parse HSVM hypothesis list
        # Support both 'spatial_footprint' (new) and 'navigation_value' (legacy) field names
        hypotheses = [
            SemanticHypothesis(
                id=h.get('id', i+1),
                hypothesis_type=h.get('hypothesis_type', 'unknown'),
                description=h.get('description', ''),
                reasoning=h.get('reasoning', ''),
                confidence=h.get('confidence', 0.5),
                navigation_value=h.get('spatial_footprint', h.get('navigation_value', 0.5))
            )
            for i, h in enumerate(data.get('semantic_hypotheses', []))
        ]

        # Parse IG hypothesis list (optional)
        ig_hypotheses = None
        if 'ig_hypotheses' in data:
            ig_hypotheses = [
                IGHypothesis(
                    id=h.get('id', i+1),
                    prompt=h.get('prompt', ''),
                    structure_type=h.get('structure_type', 'other'),
                    reasoning=h.get('reasoning', '')
                )
                for i, h in enumerate(data.get('ig_hypotheses', []))
            ]

        return cls(
            environment_analysis=environment,
            target_location_reasoning=target_reasoning,
            semantic_hypotheses=hypotheses,
            exploration_strategy=data.get('exploration_strategy'),
            ig_hypotheses=ig_hypotheses,
            connectivity_summary=data.get('connectivity_summary')
        )


# ==================== Target Object Configuration ====================

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
# Corresponds to main.tex Section III-C: Hierarchical Semantic Value Map via Micro-Perception (HSVM)
#
# HSVM 4-layer structure:
# 1. Room Type: Current room type, largest spatial influence range
# 2. Target Object: Locations where target object may appear
# 3. Co-occurrence Objects: Objects that frequently appear with target object
# 4. Object Parts: Parts/components of target object, smallest spatial influence range
#
# Formula: V_l(p) = C_l × a_l × s_l^new(p), then V_HSVM(p) = (1/|H|) × Σ V_l(p)
# - C_l: Co-occurrence Confidence P(A | B_l) (confidence field)
# - a_l: Spatial Footprint / Area Ratio (navigation_value field, renamed to spatial_footprint in prompt)

# ==================== IG Hypothesis Prompt ====================
# Corresponds to main.tex Section III-D / Appendix A.3: Information Gain Estimation
#
# IG hypotheses differ from HSVM hypotheses:
# - HSVM hypotheses: target-dependent (depend on target object)
# - IG hypotheses: target-agnostic (independent of target, focus on connectivity structures)
#
# IG hypotheses describe spatial connectivity structures (corridors, doorways, passages, openings)
# Used to estimate long-horizon information gain
#
# Important: Generated prompts must follow the style of these baseline prompts:
# - "This view shows a corridor or hallway leading to other areas"
# - "There is a doorway or opening that leads to another room"
# - "This passage connects to multiple rooms or spaces"

IG_HYPOTHESIS_PROMPT = """Based on the following scene description, generate 3-5 COMPLETE SENTENCES describing spatial connectivity structures that may lead to unexplored areas.

These sentences will be used for BLIP-2 Image-Text Matching to estimate exploration potential (Information Gain).

## Scene Description (from VLM):
{vlm_perception}

---

## CRITICAL: Prompt Format Requirements

Your generated prompts MUST follow this exact sentence structure pattern (similar to our baseline prompts):

**Baseline prompts for reference:**
1. "This view shows a corridor or hallway leading to other areas"
2. "There is a doorway or opening that leads to another room"
3. "This passage connects to multiple rooms or spaces"

**Required sentence patterns:**
- "This view shows [connectivity structure] leading to [destination]"
- "There is [connectivity structure] that leads to [destination]"
- "This [structure] connects to [destination/other areas]"

**Key requirements:**
1. Use COMPLETE SENTENCES (8-15 words), NOT short phrases
2. Start with "This view shows...", "There is...", or "This [structure]..."
3. Include connectivity verbs: "leading to", "leads to", "connects to", "opens to"
4. Describe WHERE the passage leads (other areas, rooms, spaces)
5. Focus on spatial transitions, NOT objects or room types

## Output Format (MUST be valid JSON, no markdown):

{{
  "ig_hypotheses": [
    {{
      "id": 1,
      "prompt": "This view shows [structure] leading to [destination]",
      "structure_type": "corridor/doorway/passage/archway/opening/other",
      "reasoning": "why this suggests high exploration potential"
    }},
    ...
  ],
  "connectivity_summary": "Brief summary of observed connectivity features"
}}

---

## Good Examples (follow this style):
- "This view shows a corridor leading to private rooms"
- "There is a doorway that leads to the kitchen area"
- "This hallway connects to multiple bedrooms"
- "There is an opening that leads to the dining space"
- "This passage connects to unexplored living areas"

## Bad Examples (DO NOT use this style):
- "corridor to kitchen" (too short, missing sentence structure)
- "bedroom" (this is a room type, not connectivity)
- "sofa area" (this is an object, not connectivity)
- "door" (too vague, missing destination)

## Important Rules:
1. Generate 3-5 hypotheses describing CONNECTIVITY STRUCTURES only
2. Each prompt MUST be a COMPLETE SENTENCE (8-15 words)
3. Use patterns: "This view shows...", "There is...", "This [structure]..."
4. Include WHERE the passage leads to
5. Do NOT mention target objects (those belong to HSVM)
6. Output valid JSON only - NO markdown code blocks

Generate your analysis:
"""

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

### Level 1: Room Type - LARGEST spatial footprint
The type of room where {target_object} is typically found.
- Examples: "bedroom", "bathroom", "kitchen", "living room"
- This hypothesis has the LARGEST spatial propagation range
- Confidence: How likely is {target_object} to be in this room type?
- Spatial Footprint: Should be HIGH (0.8-1.0) as rooms cover large areas

### Level 2: Target Object - MEDIUM-HIGH spatial footprint
The target object itself that the robot is searching for.
- This should be the {target_object} directly (e.g., "bed", "toilet", "TV")
- Confidence: MUST be 1.0 (this IS the target we are looking for)
- Spatial Footprint: MEDIUM-HIGH (0.6-0.8)

### Level 3: Co-occurrence Objects - MEDIUM spatial footprint
Objects that frequently appear together with {target_object}.
- Examples for bed: "nightstand", "dresser", "wardrobe", "bedside lamp"
- Examples for toilet: "sink", "bathtub", "towel rack", "bathroom mirror"
- Seeing these objects suggests {target_object} is nearby
- Confidence: How often does this object appear with {target_object}?
- Spatial Footprint: MEDIUM (0.4-0.7)

### Level 4: Object Parts - SMALLEST spatial footprint
Distinctive parts or features of {target_object} itself.
- Examples for bed: "mattress", "pillows", "headboard", "bed frame"
- Examples for toilet: "toilet seat", "flush handle", "porcelain bowl"
- Only visible when very close to the target
- Confidence: How distinctive is this part?
- Spatial Footprint: LOW (0.2-0.4) as parts indicate immediate proximity

---

## Scoring Guidelines:

### Confidence (C_l, 0.0-1.0): Co-occurrence likelihood
How likely is target A given semantic cue B_l? P(A | B_l)
- 0.8-1.0: Very strong association (e.g., beds are almost always in bedrooms)
- 0.5-0.7: Moderate association (e.g., nightstands are often near beds)
- 0.2-0.4: Weak but possible association

### Spatial Footprint (a_l, 0.0-1.0): Area ratio / spatial influence range
Relative physical extent of this entity compared to others. Larger entities occupy more sampling space.
- 0.8-1.0: Room-level footprint (room_type covers large area)
- 0.6-0.8: Object-level footprint (target_object location)
- 0.4-0.6: Nearby object footprint (co_occurrence objects)
- 0.2-0.4: Small part footprint (object_part, immediate proximity)

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
      "spatial_footprint": 0.9
    }},
    {{
      "id": 2,
      "hypothesis_type": "target_object",
      "description": "{target_object}",
      "reasoning": "this is the target object we are searching for",
      "confidence": 1.0,
      "spatial_footprint": 0.7
    }},
    {{
      "id": 3,
      "hypothesis_type": "co_occurrence",
      "description": "co-occurring object name (2-4 words)",
      "reasoning": "this object often appears with {target_object}",
      "confidence": 0.65,
      "spatial_footprint": 0.5
    }},
    {{
      "id": 4,
      "hypothesis_type": "object_part",
      "description": "part/feature name (2-3 words)",
      "reasoning": "distinctive part of {target_object}",
      "confidence": 0.6,
      "spatial_footprint": 0.3
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


# ==================== Core Analysis Class ====================

class LLMHypothesisAnalyzer:
    """LLM Hypothesis Analyzer - Supports Ollama and DeepSeek API"""

    # DeepSeek API Configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self,
                 api_url: str = "https://api.deepseek.com/chat/completions",
                 model: str = "deepseek-chat",
                 env_base_dir: str = "env",
                 api_key: Optional[str] = None):
        """
        Args:
            api_url: LLM API address
            model: Model name (deepseek-chat, deepseek-reasoner, or ollama model name)
            env_base_dir: Environment dataset root directory
            api_key: API key (required for DeepSeek)
        """
        self.api_url = api_url
        self.model = model
        self.env_base_dir = Path(env_base_dir)
        self.api_key = api_key or self.DEEPSEEK_API_KEY

        # Auto-detect API type
        self.is_deepseek = "deepseek" in api_url.lower()

    def load_vlm_analysis(self, vlm_file: Path) -> str:
        """Load VLM analysis file"""
        if not vlm_file.exists():
            raise FileNotFoundError(f"VLM analysis file not found: {vlm_file}")

        with open(vlm_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return content

    def build_prompt(self, target_object: str, vlm_perception: str) -> str:
        """Build LLM prompt"""
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
        """Call LLM to generate hypotheses - Supports DeepSeek and Ollama"""

        print(f"  Calling LLM: {self.model}")
        print(f"  API: {self.api_url}")
        print(f"  Temperature: {temperature}")

        if self.is_deepseek:
            return self._call_deepseek(prompt, temperature, timeout)
        else:
            return self._call_ollama(prompt, temperature, timeout)

    def _call_deepseek(self, prompt: str, temperature: float, timeout: int) -> str:
        """Call DeepSeek API"""
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
        """Call Ollama API"""
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
        """Parse LLM response into structured data"""
        # Clean possible markdown code block markers
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        # Try to extract JSON
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, cleaned_response)

        json_str = None
        for match in matches:
            try:
                data = json.loads(match)
                # Check required fields - new format
                if 'semantic_hypotheses' in data:
                    json_str = match
                    break
            except json.JSONDecodeError:
                continue

        if not json_str:
            # Last attempt: try full text
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to extract valid JSON from LLM response: {cleaned_response[:200]}...")
        else:
            data = json.loads(json_str)

        # Validate required fields - new format
        required_fields = ['semantic_hypotheses']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return NavigationAnalysis.from_json(data)

    def get_target_object(self, task_dir: Path) -> Optional[str]:
        """Get target object from task_info.txt"""
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
        Analyze a single task

        Args:
            env_name: Environment name
            task_num: Task number
            temperature: Generation temperature (0-1)

        Returns:
            (success, analysis, error_msg)
        """
        task_dir = self.env_base_dir / f"env_{env_name}" / f"task{task_num}"

        print(f"\n{'='*70}")
        print(f"LLM Hypothesis Analysis")
        print(f"{'='*70}")
        print(f"Task directory: {task_dir}")

        # Check task directory
        if not task_dir.exists():
            return False, None, f"Task directory not found: {task_dir}"

        # Get target object
        target_object = self.get_target_object(task_dir)
        if not target_object:
            return False, None, "Unable to get target object from task_info.txt"

        print(f"Target object: {target_object}")

        try:
            # 1. Load VLM analysis
            print(f"\n[1/4] Loading VLM analysis...")
            vlm_file = task_dir / "vlm_analysis.txt"
            vlm_perception = self.load_vlm_analysis(vlm_file)
            print(f"  Loaded VLM analysis ({len(vlm_perception)} chars)")

            # 2. Build Prompt
            print(f"\n[2/4] Building analysis prompt...")
            prompt = self.build_prompt(target_object, vlm_perception)
            print(f"  Prompt length: {len(prompt)} chars")

            # 3. Call LLM
            print(f"\n[3/4] Calling LLM to generate hypotheses...")
            t0 = time.time()
            response = self.call_llm(prompt, temperature=temperature)
            t1 = time.time()
            print(f"  Duration: {t1-t0:.2f}s")
            print(f"  Response length: {len(response)} chars")

            # 4. Parse response
            print(f"\n[4/4] Parsing LLM response...")
            analysis = self.parse_response(response)
            print(f"  Environment type: {analysis.environment_analysis.building_type}")
            print(f"  Generated hypotheses: {len(analysis.semantic_hypotheses)}")

            return True, analysis, None

        except Exception as e:
            return False, None, str(e)

    def build_ig_prompt(self, vlm_perception: str) -> str:
        """Build IG hypothesis generation prompt (target-agnostic)"""
        return IG_HYPOTHESIS_PROMPT.format(vlm_perception=vlm_perception)

    def parse_ig_response(self, response: str) -> IGAnalysis:
        """Parse IG hypothesis generation response"""
        # Clean possible markdown code block markers
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
                if 'ig_hypotheses' in data:
                    json_str = match
                    break
            except json.JSONDecodeError:
                continue

        if not json_str:
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to extract valid JSON from IG response: {cleaned_response[:200]}...")
        else:
            data = json.loads(json_str)

        # Validate required fields
        if 'ig_hypotheses' not in data:
            raise ValueError("Missing required field: ig_hypotheses")

        return IGAnalysis.from_json(data)

    def generate_ig_hypotheses(self,
                               vlm_perception: str,
                               temperature: float = 0.3) -> Tuple[bool, Optional[IGAnalysis], Optional[str]]:
        """
        Generate IG hypotheses (independent of target object)

        According to paper Section III-D / Appendix A.3:
        IG hypotheses are target-agnostic, focusing only on spatial connectivity structures

        Args:
            vlm_perception: VLM environment analysis result
            temperature: Generation temperature

        Returns:
            (success, ig_analysis, error_msg)
        """
        try:
            print(f"\n[IG] Generating IG hypotheses...")
            prompt = self.build_ig_prompt(vlm_perception)
            print(f"  IG Prompt length: {len(prompt)} chars")

            t0 = time.time()
            response = self.call_llm(prompt, temperature=temperature)
            t1 = time.time()
            print(f"  IG LLM duration: {t1-t0:.2f}s")
            print(f"  Response length: {len(response)} chars")

            ig_analysis = self.parse_ig_response(response)
            print(f"  Generated IG hypotheses: {len(ig_analysis.ig_hypotheses)}")

            return True, ig_analysis, None

        except Exception as e:
            return False, None, str(e)

    def analyze_single_with_ig(self,
                               env_name: str,
                               task_num: int,
                               temperature: float = 0.3,
                               generate_ig: bool = True) -> Tuple[bool, Optional[NavigationAnalysis], Optional[str]]:
        """
        Analyze a single task, including HSVM and IG hypothesis generation

        Args:
            env_name: Environment name
            task_num: Task number
            temperature: Generation temperature (0-1)
            generate_ig: Whether to generate IG hypotheses

        Returns:
            (success, analysis, error_msg)
        """
        task_dir = self.env_base_dir / f"env_{env_name}" / f"task{task_num}"

        print(f"\n{'='*70}")
        print(f"LLM Hypothesis Analysis (HSVM + IG)")
        print(f"{'='*70}")
        print(f"Task directory: {task_dir}")

        # Check task directory
        if not task_dir.exists():
            return False, None, f"Task directory not found: {task_dir}"

        # Get target object
        target_object = self.get_target_object(task_dir)
        if not target_object:
            return False, None, "Unable to get target object from task_info.txt"

        print(f"Target object: {target_object}")

        try:
            # 1. Load VLM analysis
            print(f"\n[1/5] Loading VLM analysis...")
            vlm_file = task_dir / "vlm_analysis.txt"
            vlm_perception = self.load_vlm_analysis(vlm_file)
            print(f"  Loaded VLM analysis ({len(vlm_perception)} chars)")

            # 2. Build HSVM Prompt
            print(f"\n[2/5] Building HSVM analysis prompt...")
            prompt = self.build_prompt(target_object, vlm_perception)
            print(f"  Prompt length: {len(prompt)} chars")

            # 3. Call LLM to generate HSVM hypotheses
            print(f"\n[3/5] Calling LLM to generate HSVM hypotheses...")
            t0 = time.time()
            response = self.call_llm(prompt, temperature=temperature)
            t1 = time.time()
            print(f"  Duration: {t1-t0:.2f}s")
            print(f"  Response length: {len(response)} chars")

            # 4. Parse HSVM response
            print(f"\n[4/5] Parsing HSVM response...")
            analysis = self.parse_response(response)
            print(f"  Environment type: {analysis.environment_analysis.building_type}")
            print(f"  Generated HSVM hypotheses: {len(analysis.semantic_hypotheses)}")

            # 5. Generate IG hypotheses (optional)
            if generate_ig:
                print(f"\n[5/5] Generating IG hypotheses...")
                ig_success, ig_analysis, ig_error = self.generate_ig_hypotheses(
                    vlm_perception, temperature=temperature
                )

                if ig_success and ig_analysis:
                    # Merge IG hypotheses into NavigationAnalysis
                    analysis.ig_hypotheses = ig_analysis.ig_hypotheses
                    analysis.connectivity_summary = ig_analysis.connectivity_summary
                    print(f"  IG hypotheses merged: {len(ig_analysis.ig_hypotheses)}")
                else:
                    print(f"  IG hypothesis generation failed: {ig_error}")
                    # Does not affect HSVM hypotheses, continue returning
            else:
                print(f"\n[5/5] Skipping IG hypothesis generation")

            return True, analysis, None

        except Exception as e:
            return False, None, str(e)


# ==================== Output Formatting ====================

class AnalysisReportWriter:
    """Analysis Result Report Writer"""

    @staticmethod
    def write_json(output_file: Path, analysis: NavigationAnalysis) -> None:
        """Write JSON format result"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def write_text_report(output_file: Path, analysis: NavigationAnalysis) -> None:
        """Write text format report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Global Navigation Hypothesis Analysis Report\n")
            f.write("="*70 + "\n\n")

            # Environment Analysis
            f.write("[Environment Analysis]\n")
            env = analysis.environment_analysis
            f.write(f"Building Type: {env.building_type}\n")
            f.write(f"Current Location: {env.current_area}\n")
            f.write(f"Global Layout Inference: {env.global_layout_inference}\n\n")

            # Target Location Reasoning
            f.write("[Target Location Reasoning]\n")
            target = analysis.target_location_reasoning
            f.write(f"Typical Locations: {target.typical_locations}\n")
            f.write(f"Likely Path: {target.likely_path}\n")
            f.write(f"Key Landmarks: {target.key_landmarks}\n\n")

            # Semantic Hypotheses
            f.write("[Semantic Hypotheses]\n")
            f.write(f"{'ID':<3} {'Type':<18} {'Description':<25} {'C_l':<6} {'a_l':<6} {'Weight':<8}\n")
            f.write("-"*70 + "\n")

            # HSVM 4-layer structure type mapping (corresponding to main.tex Section III-C)
            type_map = {
                # New format (HSVM 4-layer structure)
                'room_type': 'L1-Room Type',
                'target_object': 'L2-Target Object',
                'co_occurrence': 'L3-Co-occurrence',
                'object_part': 'L4-Object Part',
                # Legacy format compatibility
                'building_layout': 'Building Layout',
                'target_room': 'Target Room',
                'adjacent_area': 'Adjacent Area',
                'indicator_object': 'Indicator Object',
                'target_feature': 'Target Feature',
                'spatial_navigation': 'Spatial Navigation',
                'room_context': 'Room Context',
                'part_attribute': 'Part Attribute'
            }

            for h in analysis.semantic_hypotheses:
                type_str = type_map.get(h.hypothesis_type, h.hypothesis_type)
                desc = h.description[:25].ljust(25)
                f.write(f"{h.id:<3} {type_str:<18} {desc} "
                       f"{h.confidence:<6.2f} {h.navigation_value:<6.2f} {h.weight:<8.4f}\n")
                f.write(f"    Reasoning: {h.reasoning}\n")

            f.write("\n")

            # IG Hypotheses (if present)
            if analysis.ig_hypotheses:
                f.write("[IG Hypotheses (Information Gain)]\n")
                f.write(f"{'ID':<3} {'Type':<12} {'Prompt':<40}\n")
                f.write("-"*70 + "\n")

                for h in analysis.ig_hypotheses:
                    prompt_str = h.prompt[:40].ljust(40)
                    f.write(f"{h.id:<3} {h.structure_type:<12} {prompt_str}\n")
                    f.write(f"    Reasoning: {h.reasoning}\n")

                f.write("\n")

                if analysis.connectivity_summary:
                    f.write(f"Connectivity Summary: {analysis.connectivity_summary}\n\n")

            # Exploration Strategy
            if analysis.exploration_strategy:
                f.write("[Exploration Strategy]\n")
                f.write(analysis.exploration_strategy + "\n\n")

            # Statistics
            f.write("[Statistics]\n")
            if analysis.semantic_hypotheses:
                avg_conf = sum(h.confidence for h in analysis.semantic_hypotheses) / len(analysis.semantic_hypotheses)
                avg_nav = sum(h.navigation_value for h in analysis.semantic_hypotheses) / len(analysis.semantic_hypotheses)
                avg_weight = sum(h.weight for h in analysis.semantic_hypotheses) / len(analysis.semantic_hypotheses)
                f.write(f"HSVM Hypothesis Count: {len(analysis.semantic_hypotheses)}\n")
                f.write(f"Average Confidence (C_l): {avg_conf:.3f}\n")
                f.write(f"Average Spatial Footprint (a_l): {avg_nav:.3f}\n")
                f.write(f"Average Weight (C_l × a_l): {avg_weight:.4f}\n")

            if analysis.ig_hypotheses:
                f.write(f"IG Hypothesis Count: {len(analysis.ig_hypotheses)}\n")

            f.write("\n" + "="*70 + "\n")


# ==================== Batch Processing Class ====================

class BatchAnalyzer:
    """Batch Analyzer"""

    def __init__(self, analyzer: LLMHypothesisAnalyzer, delay: int = 10):
        """
        Args:
            analyzer: LLM hypothesis analyzer instance
            delay: Delay between tasks (seconds)
        """
        self.analyzer = analyzer
        self.delay = delay

    def discover_tasks(self, env_name: str) -> List[int]:
        """Discover all tasks with VLM analysis in the specified environment"""
        env_dir = self.analyzer.env_base_dir / f"env_{env_name}"

        if not env_dir.exists():
            print(f"Warning: Environment directory not found: {env_dir}")
            return []

        tasks = []
        for task_dir in sorted(env_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith('task'):
                try:
                    task_num = int(task_dir.name[4:])
                    # Check if VLM analysis exists
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
                      output_format: str = 'both',
                      generate_ig: bool = True) -> Dict:
        """
        Batch analyze environment dataset

        Args:
            env_name: Environment name
            start_task: Start task number (inclusive)
            end_task: End task number (inclusive)
            max_tasks: Maximum number of tasks limit
            temperature: Generation temperature
            output_format: Output format ('json', 'text', 'both')
            generate_ig: Whether to generate IG hypotheses (default True)

        Returns:
            Statistics dictionary
        """
        print(f"\n{'#'*70}")
        print(f"# Batch Hypothesis Analysis: env_{env_name}")
        print(f"# LLM Model: {self.analyzer.model}")
        print(f"# API: {self.analyzer.api_url}")
        print(f"# Generate IG Hypotheses: {generate_ig}")
        print(f"# Task Delay: {self.delay} seconds")
        print(f"# Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")

        # Discover all tasks
        all_tasks = self.discover_tasks(env_name)

        if not all_tasks:
            print(f"Error: No tasks with VLM analysis found in env_{env_name}")
            return None

        print(f"Found {len(all_tasks)} tasks")

        # Apply task range filtering
        if start_task is not None:
            all_tasks = [t for t in all_tasks if t >= start_task]
        if end_task is not None:
            all_tasks = [t for t in all_tasks if t <= end_task]
        if max_tasks is not None:
            all_tasks = all_tasks[:max_tasks]

        print(f"Will analyze {len(all_tasks)} tasks: {all_tasks[:10]}{'...' if len(all_tasks) > 10 else ''}\n")

        # Statistics
        stats = {
            'total': len(all_tasks),
            'success': 0,
            'failed': 0,
            'total_time': 0,
            'results': []
        }

        writer = AnalysisReportWriter()

        # Analyze tasks one by one
        for idx, task_num in enumerate(all_tasks, 1):
            print(f"\nProgress: [{idx}/{len(all_tasks)}]")

            t0 = time.time()
            # Use analysis method with IG
            if generate_ig:
                success, analysis, error_msg = self.analyzer.analyze_single_with_ig(
                    env_name, task_num, temperature=temperature, generate_ig=True
                )
            else:
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
                    print(f"  JSON result: {json_file}")

                if output_format in ['text', 'both']:
                    text_file = task_dir / "llm_hypothesis_analysis.txt"
                    writer.write_text_report(text_file, analysis)
                    print(f"  Text report: {text_file}")

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
                print(f"  Success (duration: {duration:.2f}s)")
            else:
                stats['failed'] += 1
                print(f"  Failed: {error_msg}")

            # Wait for specified time if not last task
            if idx < len(all_tasks):
                print(f"\nWaiting {self.delay} seconds before continuing...")
                for remaining in range(self.delay, 0, -1):
                    print(f"  Remaining: {remaining} seconds", end='\r')
                    time.sleep(1)
                print()

        return stats

    def save_summary(self, env_name: str, stats: Dict) -> None:
        """Save batch analysis summary"""
        summary_file = self.analyzer.env_base_dir / f"env_{env_name}" / "batch_llm_hypothesis_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Batch LLM Hypothesis Analysis Summary: env_{env_name}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LLM Model: {self.analyzer.model}\n")
            f.write(f"Total Tasks: {stats['total']}\n")
            f.write(f"Success: {stats['success']}\n")
            f.write(f"Failed: {stats['failed']}\n")
            f.write(f"Success Rate: {stats['success']/stats['total']*100:.2f}%\n")
            f.write(f"Total Time: {stats['total_time']:.2f} seconds ({stats['total_time']/60:.2f} minutes)\n")
            f.write(f"Average Time: {stats['total_time']/stats['total']:.2f} seconds/task\n\n")

            f.write(f"Detailed Results:\n")
            f.write(f"{'-'*70}\n")

            for result in stats['results']:
                status = "OK" if result['success'] else "FAIL"
                f.write(f"[{status}] task{result['task']}: {result['duration']:.2f}s")
                if result['error']:
                    f.write(f" - {result['error']}")
                f.write("\n")

        print(f"\nSummary saved to: {summary_file}")

    def print_summary(self, env_name: str, stats: Dict) -> None:
        """Print analysis summary"""
        print(f"\n{'#'*70}")
        print(f"# Analysis Complete: env_{env_name}")
        print(f"{'#'*70}\n")

        print(f"Total Tasks: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success Rate: {stats['success']/stats['total']*100:.2f}%")
        print(f"Total Time: {stats['total_time']:.2f} seconds ({stats['total_time']/60:.2f} minutes)")
        print(f"Average Time: {stats['total_time']/stats['total']:.2f} seconds/task")

        if stats['failed'] > 0:
            print(f"\nFailed Tasks:")
            for result in stats['results']:
                if not result['success']:
                    print(f"  - task{result['task']}: {result['error']}")


# ==================== Retry Analysis Class ====================

class RetryAnalyzer:
    """Retry Missing Tasks Analyzer"""

    def __init__(self, analyzer: LLMHypothesisAnalyzer, delay: int = 40):
        """
        Args:
            analyzer: LLM hypothesis analyzer instance
            delay: Delay between tasks (seconds)
        """
        self.analyzer = analyzer
        self.delay = delay

    def find_missing_tasks(self, env_name: str,
                           start_task: Optional[int] = None,
                           end_task: Optional[int] = None) -> List[int]:
        """
        Find tasks missing LLM analysis results

        Args:
            env_name: Environment name
            start_task: Start task number
            end_task: End task number

        Returns:
            List of missing task numbers
        """
        env_dir = self.analyzer.env_base_dir / f"env_{env_name}"

        if not env_dir.exists():
            print(f"Warning: Environment directory not found: {env_dir}")
            return []

        missing_tasks = []
        for task_dir in sorted(env_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith('task'):
                try:
                    task_num = int(task_dir.name[4:])

                    # Apply range filtering
                    if start_task is not None and task_num < start_task:
                        continue
                    if end_task is not None and task_num > end_task:
                        continue

                    # Check if VLM analysis exists but LLM analysis is missing
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
        Retry specified task list

        Args:
            env_name: Environment name
            tasks: List of task numbers
            temperature: Generation temperature
            output_format: Output format
            skip_exists: Whether to skip if result already exists

        Returns:
            Statistics dictionary
        """
        print("=" * 80)
        print(f"Retry Missing LLM Analysis Tasks")
        print("=" * 80)
        print(f"Environment: {env_name}")
        print(f"Tasks to process: {len(tasks)}")
        if tasks:
            print(f"Task range: task{min(tasks)} ~ task{max(tasks)}")
        print(f"LLM API: {self.analyzer.api_url}")
        print(f"Model: {self.analyzer.model}")
        print(f"Task delay: {self.delay} seconds")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        if not tasks:
            print("No tasks found to process")
            return None

        # Statistics
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

            # Check if should skip
            if skip_exists and json_file.exists():
                print(f"[{idx}/{len(tasks)}] SKIP task{task_num}: already exists")
                stats['skipped'] += 1
                if idx < len(tasks):
                    time.sleep(self.delay)
                continue

            print(f"[{idx}/{len(tasks)}] Processing task{task_num}...", end=' ', flush=True)

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

            # Task delay
            if idx < len(tasks):
                print(f"  Waiting {self.delay} seconds...", end=' ', flush=True)
                time.sleep(self.delay)
                print("done")

        return stats

    def save_retry_log(self, env_name: str, stats: Dict) -> Path:
        """Save retry log"""
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

        print(f"\nLog saved to: {log_file}")
        return log_file

    def print_retry_summary(self, stats: Dict) -> None:
        """Print retry summary"""
        print("\n" + "=" * 80)
        print("Processing Complete")
        print("=" * 80)
        print(f"Total Tasks: {stats['total']}")
        print(f"Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "Success: 0")
        print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "Failed: 0")
        print(f"Skipped: {stats['skipped']}")
        print(f"Total Time: {stats['total_time']:.1f} seconds")
        if stats['total'] > 0:
            print(f"Average Time: {stats['total_time']/stats['total']:.1f} seconds/task")

        if stats['failed_tasks']:
            print(f"\nFailed Tasks ({len(stats['failed_tasks'])}):")
            for task_id, error in stats['failed_tasks'][:10]:
                print(f"  task{task_id}: {error[:60] if error else 'Unknown error'}")
            if len(stats['failed_tasks']) > 10:
                print(f"  ... and {len(stats['failed_tasks']) - 10} more failed tasks")

        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


# ==================== Command Line Interface ====================

def cmd_single(args):
    """Single task analysis command"""
    analyzer = LLMHypothesisAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir
    )

    # Select analysis method based on whether IG hypotheses are needed
    generate_ig = getattr(args, 'generate_ig', True)
    if generate_ig:
        success, analysis, error_msg = analyzer.analyze_single_with_ig(
            args.env, args.task, temperature=args.temperature, generate_ig=True
        )
    else:
        success, analysis, error_msg = analyzer.analyze_single(
            args.env, args.task, temperature=args.temperature
        )

    if not success:
        print(f"\nError: {error_msg}")
        return 1

    # Save results
    task_dir = analyzer.env_base_dir / f"env_{args.env}" / f"task{args.task}"
    writer = AnalysisReportWriter()

    print(f"\n{'='*70}")
    print("Saving analysis results...")
    print(f"{'='*70}")

    if args.output_format in ['json', 'both']:
        json_file = task_dir / "llm_hypothesis_analysis.json"
        writer.write_json(json_file, analysis)
        print(f"JSON result: {json_file}")

    if args.output_format in ['text', 'both']:
        text_file = task_dir / "llm_hypothesis_analysis.txt"
        writer.write_text_report(text_file, analysis)
        print(f"Text report: {text_file}")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")

    return 0


def cmd_batch(args):
    """Batch analysis command"""
    analyzer = LLMHypothesisAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir
    )

    batch = BatchAnalyzer(analyzer, delay=args.delay)

    all_stats = {}

    # Get whether to generate IG hypotheses option
    generate_ig = getattr(args, 'generate_ig', True)

    for env_name in args.env:
        stats = batch.analyze_batch(
            env_name,
            start_task=args.start,
            end_task=args.end,
            max_tasks=args.max_tasks,
            temperature=args.temperature,
            output_format=args.output_format,
            generate_ig=generate_ig
        )

        if stats:
            all_stats[env_name] = stats
            batch.print_summary(env_name, stats)
            batch.save_summary(env_name, stats)

            # Wait if not the last environment
            if env_name != args.env[-1]:
                print(f"\n{'='*70}")
                print(f"Preparing to analyze next environment...")
                print(f"Waiting {args.delay} seconds...")
                print(f"{'='*70}\n")
                time.sleep(args.delay)

    # Print overall summary
    if len(all_stats) > 1:
        print(f"\n{'#'*70}")
        print(f"# Overall Summary")
        print(f"{'#'*70}\n")

        total_tasks = sum(s['total'] for s in all_stats.values())
        total_success = sum(s['success'] for s in all_stats.values())
        total_failed = sum(s['failed'] for s in all_stats.values())
        total_time = sum(s['total_time'] for s in all_stats.values())

        print(f"Environments Analyzed: {len(all_stats)}")
        print(f"Total Tasks: {total_tasks}")
        print(f"Total Success: {total_success}")
        print(f"Total Failed: {total_failed}")
        print(f"Overall Success Rate: {total_success/total_tasks*100:.2f}%")
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print()

        for env_name, stats in all_stats.items():
            print(f"  {env_name}: {stats['success']}/{stats['total']} "
                  f"({stats['success']/stats['total']*100:.2f}%)")

    print(f"\n{'='*70}")
    print("All analysis tasks complete!")
    print(f"{'='*70}\n")

    return 0


def cmd_retry(args):
    """Retry missing tasks command"""
    analyzer = LLMHypothesisAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir
    )

    retry = RetryAnalyzer(analyzer, delay=args.delay)

    # Determine tasks to process
    if args.tasks:
        # Use specified task list
        tasks_to_process = sorted(args.tasks)
    else:
        # Auto-scan for missing tasks
        print(f"Scanning env_{args.env} for tasks missing LLM analysis...")
        tasks_to_process = retry.find_missing_tasks(
            args.env,
            start_task=args.start,
            end_task=args.end
        )

        if not tasks_to_process:
            print("No tasks missing LLM analysis found, all tasks are complete!")
            return 0

        print(f"Found {len(tasks_to_process)} missing tasks: {tasks_to_process[:20]}{'...' if len(tasks_to_process) > 20 else ''}")

    # Execute retry
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
        description="LLM Semantic Hypothesis Analysis Tool - Supports single task, batch analysis, and retry missing tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
===============

Single task analysis:
  python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0
  python llm_hypothesis_analyzer.py single --env hm3dv2 --task 0 --api-url http://localhost:11434/api/chat

Batch analysis:
  python llm_hypothesis_analyzer.py batch --env hm3dv2
  python llm_hypothesis_analyzer.py batch --env hm3dv2 --max-tasks 10
  python llm_hypothesis_analyzer.py batch --env hm3dv2 --start 5 --end 15
  python llm_hypothesis_analyzer.py batch --env hm3dv1 hm3dv2 --model qwen2:32b

Retry missing tasks:
  python llm_hypothesis_analyzer.py retry --env hm3dv2                      # Auto-scan missing tasks
  python llm_hypothesis_analyzer.py retry --env hm3dv2 --tasks 100 200 300  # Specify task list
  python llm_hypothesis_analyzer.py retry --env hm3dv2 --start 100 --end 200  # Specify range
  python llm_hypothesis_analyzer.py retry --env hm3dv2 --skip-exists        # Skip existing results
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Execution mode')

    # Single task mode
    single_parser = subparsers.add_parser('single', help='Single task analysis mode')
    single_parser.add_argument('--env', type=str, required=True,
                               choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                               help='Environment category')
    single_parser.add_argument('--task', type=int, required=True,
                               help='Task number')
    single_parser.add_argument('--api-url', type=str, default='https://api.deepseek.com/chat/completions',
                               help='LLM API address (default: DeepSeek API)')
    single_parser.add_argument('--model', type=str, default='deepseek-chat',
                               help='LLM model name (default: deepseek-chat)')
    single_parser.add_argument('--temperature', type=float, default=0.3,
                               help='Generation temperature 0.0-1.0 (default: 0.3)')
    single_parser.add_argument('--env-base-dir', type=str, default='env',
                               help='Environment dataset root directory')
    single_parser.add_argument('--output-format', type=str, default='both',
                               choices=['json', 'text', 'both'],
                               help='Output format (default: both)')
    single_parser.add_argument('--generate-ig', action='store_true', default=True,
                               help='Generate IG hypotheses (default: True)')
    single_parser.add_argument('--no-ig', action='store_false', dest='generate_ig',
                               help='Do not generate IG hypotheses')

    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Batch analysis mode')
    batch_parser.add_argument('--env', type=str, nargs='+', required=True,
                              choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                              help='Environments to analyze (can specify multiple)')
    batch_parser.add_argument('--start', type=int, default=None,
                              help='Start task number (inclusive)')
    batch_parser.add_argument('--end', type=int, default=None,
                              help='End task number (inclusive)')
    batch_parser.add_argument('--max-tasks', type=int, default=None,
                              help='Maximum number of tasks limit')
    batch_parser.add_argument('--delay', type=int, default=5,
                              help='Delay between tasks (seconds), default 5s')
    batch_parser.add_argument('--api-url', type=str, default='https://api.deepseek.com/chat/completions',
                              help='LLM API address (default: DeepSeek API)')
    batch_parser.add_argument('--model', type=str, default='deepseek-chat',
                              help='LLM model name (default: deepseek-chat)')
    batch_parser.add_argument('--temperature', type=float, default=0.2,
                              help='Generation temperature 0.0-1.0 (default: 0.2)')
    batch_parser.add_argument('--env-base-dir', type=str, default='env',
                              help='Environment dataset root directory')
    batch_parser.add_argument('--output-format', type=str, default='both',
                              choices=['json', 'text', 'both'],
                              help='Output format (default: both)')
    batch_parser.add_argument('--generate-ig', action='store_true', default=True,
                              help='Generate IG hypotheses (default: True)')
    batch_parser.add_argument('--no-ig', action='store_false', dest='generate_ig',
                              help='Do not generate IG hypotheses')

    # Retry mode
    retry_parser = subparsers.add_parser('retry', help='Retry missing tasks mode')
    retry_parser.add_argument('--env', type=str, required=True,
                              choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                              help='Environment category')
    retry_parser.add_argument('--tasks', type=int, nargs='+', default=None,
                              help='Task numbers to retry (optional, auto-scans if not specified)')
    retry_parser.add_argument('--start', type=int, default=None,
                              help='Start task number (inclusive)')
    retry_parser.add_argument('--end', type=int, default=None,
                              help='End task number (inclusive)')
    retry_parser.add_argument('--delay', type=int, default=5,
                              help='Delay between tasks (seconds), default 5s')
    retry_parser.add_argument('--api-url', type=str, default='https://api.deepseek.com/chat/completions',
                              help='LLM API address (default: DeepSeek API)')
    retry_parser.add_argument('--model', type=str, default='deepseek-chat',
                              help='LLM model name (default: deepseek-chat)')
    retry_parser.add_argument('--temperature', type=float, default=0.3,
                              help='Generation temperature 0.0-1.0 (default: 0.3)')
    retry_parser.add_argument('--env-base-dir', type=str, default='env',
                              help='Environment dataset root directory')
    retry_parser.add_argument('--output-format', type=str, default='both',
                              choices=['json', 'text', 'both'],
                              help='Output format (default: both)')
    retry_parser.add_argument('--skip-exists', action='store_true',
                              help='Skip if result already exists (default: overwrite)')

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
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
