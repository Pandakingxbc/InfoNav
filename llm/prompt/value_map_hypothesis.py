import json
from typing import List, Dict, Any

# yangzhi_12_24: 拟采用的LLM引导词prompt模板，需要输入target object 还有vlm_perception
prompt = f"""
    You are an indoor navigation agent searching for **{target_object}**.

    ## Current Perception (from VLM):
    {vlm_perception}

    ## Your Task:
    Generate TWO types of outputs:

    ### 1. Direction Guidance (方向引导)
    Based on overall scene analysis, which direction should the robot prioritize?
    This will be used to build a Curiosity Map.

    ### 2. Semantic Hypotheses (语义假说)
    Generate descriptive hypotheses about objects/features/rooms related to finding {target_object}.
    Each hypothesis should have a **text description** that can be matched against images using CLIP.
    These will be used to compute similarity with observations and build Value Map.

    ---

    ## Hypothesis Types:

    **1. Part/Attribute (部件/属性)**
    Text description of {target_object}'s visual features:
    - Examples: "black rectangular screen", "TV stand with shelves", "remote control"

    **2. Co-occurrence (共现物体)**
    Text description of objects commonly found with {target_object}:
    - Examples: "sofa", "coffee table", "entertainment center"

    **3. Room Context (房间类型)**
    Text description of room types where {target_object} is typically found:
    - Examples: "living room", "bedroom", "family room"

    ---

    ## Scoring:

    ### Confidence (置信度): 0.0-1.0
    How reliable is this hypothesis based on current evidence?

    ### Navigation Value (导航价值): 0.0-1.0  
    How large is the spatial scope this hypothesis guides to?

    | Score | Scope | Example |
    |-------|-------|---------|
    | 0.8-1.0 | Room-level | "living room" - large area |
    | 0.5-0.7 | Zone-level | "sofa area" - medium area |
    | 0.2-0.4 | Point-level | "black screen" - specific spot |

    ---

    ## Output Format:
    ```json
    {{
    "scene_summary": "Brief description of current environment",
    "target_in_view": true/false,
    
    "direction_guidance": {{
        "recommended_direction": <0-3>,
        "reasoning": "Why this direction is recommended",
        "confidence": <0.0-1.0>
    }},
    
    "semantic_hypotheses": [
        {{
        "id": <int>,
        "hypothesis_type": "part_attribute" | "co_occurrence" | "room_context",
        "description": "Text description for CLIP matching",
        "reasoning": "Why this relates to {target_object}",
        "confidence": <0.0-1.0>,
        "navigation_value": <0.0-1.0>
        }}
    ],
    
    "exploration_fallback": "Suggestion if no strong hypotheses"
    }}
    ```

    ## Direction Mapping:
    - 0 = front
    - 1 = left
    - 2 = back  
    - 3 = right

    ## Rules:
    1. Direction guidance: ONE recommended direction based on overall analysis
    2. Semantic hypotheses: Multiple (1-6) descriptive text for CLIP matching
    3. Description should be concise and visually matchable (suitable for CLIP)
    4. Generate hypotheses you have evidence for, quality over quantity

    Generate your analysis:
    """