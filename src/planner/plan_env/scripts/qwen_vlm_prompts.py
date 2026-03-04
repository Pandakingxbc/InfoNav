"""
Qwen-VL Object Validation Prompts

优化的提示词模板，用于高精度目标物体校验。

核心设计原则：
1. 明确要求 VLM 进行多步骤推理
2. 提供物体特征描述，减少混淆
3. 要求 VLM 输出结构化结果
4. 采用"先验证、再确认"的两级策略
"""

def get_validation_prompt_v1(target_object: str, 
                             target_features: str = None,
                             fallback_objects: list = None) -> str:
    """
    极简版：单步验证（快速、低成本、最少提示词）
    
    Args:
        target_object: 目标物体名称（如 "potted plant"）
        target_features: 可选的物体特征（如 "green leaves, ceramic pot"）
        fallback_objects: 可能被混淆的相似物体列表（如 ["flower", "vase"]）
    
    Returns:
        str: 极简提示词
    """
    
    prompt = f"Is there a {target_object} in this image? Answer YES or NO."
    
    if fallback_objects:
        fallback_str = ", ".join(fallback_objects)
        prompt += f" (It's NOT {fallback_str})"
    
    return prompt


def get_validation_prompt_v2_detailed(target_object: str,
                                     target_features: str = None,
                                     fallback_objects: list = None) -> str:
    """
    详细版：多步骤推理（准确、高成本）
    不限制物体位置，只要有就算成功
    适用于关键校验，或不确定情况下的二次确认
    
    Args:
        target_object: 目标物体名称
        target_features: 物体特征描述
        fallback_objects: 易混淆的相似物体
    
    Returns:
        str: 详细的多步骤验证提示词
    """
    
    prompt = f"""You are an expert vision verification system for robotic object navigation.

**PRIMARY TASK**: Is '{target_object}' present anywhere in this image?

**TARGET OBJECT DETAILS**:
- Name: {target_object}"""
    
    if target_features:
        prompt += f"\n- Characteristics: {target_features}"
    
    if fallback_objects:
        fallback_str = ", ".join(fallback_objects)
        prompt += f"\n- Common confusions: {fallback_str}"
    
    prompt += """

**MULTI-STEP VERIFICATION**:

Step 1 - FULL SCAN: 
  • Look at entire image: center, edges, corners, background
  • List all objects you can identify

Step 2 - LOCATE TARGET:
  • Is the target object present anywhere in the image?
  • Where exactly is it? (position is only for reference, not a limiting factor)

Step 3 - CONFIRM IDENTITY:
  • Verify the object matches the target's characteristics
  • If confused objects listed, confirm this is NOT one of those

Step 4 - MAKE DECISION**:
  • Only answer: YES (target is present) or NO (target is absent)
  • No UNCERTAIN - make a choice

**RESPONSE FORMAT** (JSON):
{
  "target_found": true/false,
  "confidence": 0.0-1.0,
  "location_in_image": "center/left/right/top/bottom/background/multiple/edge",
  "identifying_features": ["feature1", "feature2"],
  "confusion_ruled_out": ["object1", "object2"] or null,
  "reasoning": "Your brief explanation"
}

Now analyze this image:"""
    
    return prompt


def get_comparison_prompt(target_object: str,
                         current_view_description: str,
                         reference_features: str = None) -> str:
    """
    比较版：与参考特征对比（用于多角度验证）
    
    当机器人从不同角度看同一个物体时，可用此提示词确保一致性
    
    Args:
        target_object: 目标物体
        current_view_description: 当前视角的描述
        reference_features: 参考特征（从其他视角看到的）
    
    Returns:
        str: 比较验证提示词
    """
    
    prompt = f"""You are verifying object identity across multiple viewpoints.

**OBJECT**: {target_object}
**CURRENT VIEW**: {current_view_description}"""
    
    if reference_features:
        prompt += f"\n**REFERENCE FEATURES** (from other angles): {reference_features}"
    
    prompt += """

**TASK**:
1. Identify the target object in the current view
2. Verify it has consistent features with the reference features
3. Assess if it's the SAME object from a different angle

**OUTPUT**:
- SAME_OBJECT: true/false
- CONFIDENCE: 0.0-1.0
- CONSISTENCY: High/Medium/Low
- NOTES: Brief explanation

Now analyze:"""
    
    return prompt


# ============================================================================
# 响应解析优化
# ============================================================================

def parse_vlm_response_v2(response_text: str, target_object: str) -> tuple:
    """
    改进的响应解析逻辑（比原有的更严格）
    
    原有逻辑问题：
    - 匹配模式太多，容易产生假正例
    - 置信度计算过于乐观
    
    新逻辑：
    - 优先查找结构化输出（JSON 格式）
    - 如果是自然语言，使用更严格的匹配规则
    - 对"不确定"情况，默认保守判断
    
    Args:
        response_text: VLM 原始响应
        target_object: 目标物体
    
    Returns:
        tuple: (is_valid: bool, confidence: float, reasoning: str)
    """
    
    import json
    import re
    
    response_text = response_text.strip()
    
    # ========== 尝试解析 JSON 格式 ==========
    if '{' in response_text and '}' in response_text:
        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # 优先使用 JSON 中的结构化信息
                if 'target_found' in result:
                    is_valid = result.get('target_found', False)
                    confidence = float(result.get('confidence', 0.5))
                    reasoning = result.get('reasoning', 'No reasoning provided')
                    return is_valid, confidence, reasoning
                
                elif 'DECISION' in result:
                    decision = result.get('DECISION', '').upper()
                    confidence = float(result.get('CONFIDENCE', 0.5))
                    reasoning = result.get('REASON', '')
                    is_valid = decision == 'YES'
                    return is_valid, confidence, reasoning
        
        except (json.JSONDecodeError, ValueError):
            pass  # JSON 解析失败，继续用自然语言处理
    
    # ========== 自然语言解析（更严格的规则） ==========
    response_lower = response_text.lower()
    
    # 1. 寻找明确的"是"表述
    yes_patterns = [
        r'\bdecision:\s*yes\b',
        r'\btarget_found:\s*true\b',
        r'\bYES\b',
        r'^yes\b',
        r'\b(yes,|yes\.)\b',
        r'(是的|确实|看到了|有的)',
    ]
    
    # 2. 寻找明确的"否"表述  
    no_patterns = [
        r'\bdecision:\s*no\b',
        r'\btarget_found:\s*false\b',
        r'\bNO\b',
        r'^no\b',
        r'\b(no,|no\.)\b',
        r'(没有|没看到|不存在|没有看到)',
    ]
    
    # 3. 寻找不确定表述
    uncertain_patterns = [
        r'\bUNCERTAIN\b',
        r'\b(uncertain|unsure|ambiguous|unclear)\b',
        r'(不确定|不清楚)',
    ]
    
    # 4. 计数（但不加权）- 只计算是否存在
    has_yes = any(re.search(pattern, response_lower) for pattern in yes_patterns)
    has_no = any(re.search(pattern, response_lower) for pattern in no_patterns)
    has_uncertain = any(re.search(pattern, response_lower) for pattern in uncertain_patterns)
    
    # 5. 判决逻辑（保守策略）
    if has_uncertain:
        # 不确定时，默认保守判断
        return False, 0.3, "VLM returned uncertain response"
    
    elif has_yes and not has_no:
        # 只有肯定，没有否定 → 倾向相信
        # 但置信度上限 0.85，避免过于自信
        confidence = min(0.85, 0.7 + 0.15)  # 基础 0.7 + 偏移 0.15
        return True, confidence, "VLM confirmed target object presence"
    
    elif has_no and not has_yes:
        # 只有否定，没有肯定 → 相信否定
        confidence = min(0.85, 0.7 + 0.15)
        return False, confidence, "VLM confirmed target object absent"
    
    elif has_yes and has_no:
        # 同时出现肯定和否定 → 这是矛盾的，返回不确定
        return False, 0.4, "VLM response contains contradictory statements"
    
    else:
        # 找不到明确的判断词 → 默认否定（安全策略）
        return False, 0.2, "VLM response is ambiguous, no clear decision found"


def parse_vlm_response_with_confidence_extraction(response_text: str) -> tuple:
    """
    从 VLM 响应中提取置信度信息
    
    尝试从多种格式中提取置信度数值
    """
    
    import re
    
    # 寻找置信度数值（格式例如：0.85, 85%, 85）
    confidence_patterns = [
        r'confidence:\s*(0\.\d+|\d+%)',
        r'CONFIDENCE:\s*(0\.\d+|\d+%)',
        r'置信度:\s*(0\.\d+|\d+%)',
        r'信心:\s*(0\.\d+|\d+%)',
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, response_text)
        if match:
            conf_str = match.group(1)
            if '%' in conf_str:
                return float(conf_str.rstrip('%')) / 100.0
            else:
                return float(conf_str)
    
    # 如果找不到，返回 None（由调用者处理）
    return None


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    
    print("=" * 80)
    print("测试提示词 v1 - 简单版")
    print("=" * 80)
    prompt1 = get_validation_prompt_v1(
        target_object="potted plant",
        target_features="green leaves, ceramic pot, height ~50cm",
        fallback_objects=["flower", "vase", "decoration"]
    )
    print(prompt1)
    
    print("\n" + "=" * 80)
    print("测试提示词 v2 - 详细版")
    print("=" * 80)
    prompt2 = get_validation_prompt_v2_detailed(
        target_object="potted plant",
        target_features="green leaves, ceramic pot, height ~50cm",
        fallback_objects=["flower", "vase", "decoration"]
    )
    print(prompt2)
    
    print("\n" + "=" * 80)
    print("测试响应解析 - 严格模式")
    print("=" * 80)
    
    test_responses = [
        "DECISION: YES\nCONFIDENCE: 0.95\nREASON: Clear green potted plant visible.",
        "I see some green object but not sure if it's a plant",
        "DECISION: NO\nCONFIDENCE: 0.80\nREASON: No potted plant detected",
        '{"target_found": true, "confidence": 0.85, "reasoning": "Found potted plant on shelf"}',
    ]
    
    for response in test_responses:
        is_valid, conf, reason = parse_vlm_response_v2(response, "potted plant")
        print(f"\n响应: {response[:50]}...")
        print(f"结果: valid={is_valid}, confidence={conf:.2f}, reason={reason}")
