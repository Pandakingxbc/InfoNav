"""
Semantic Prompt Expansion for Multi-Source Value Maps

This module uses LLM to expand a navigation target into multiple semantic hypotheses.
These hypotheses cover different semantic perspectives:
- Target object attributes
- Co-occurring objects
- Room-level context
- Structural/layout priors

Reference: Value_Map_Construction_and_Fusion.md - Semantic Prompt Expansion

Author: Zager-Zhang
"""

from typing import List, Tuple, Dict
import json


# System prompt for semantic expansion
SEMANTIC_EXPANSION_SYSTEM_PROMPT = """You are an intelligent spatial reasoning assistant that helps with object navigation tasks.

Given a target object name, you will generate multiple semantic hypotheses that can help locate the object.
These hypotheses should cover different semantic perspectives:

1. **Target Attributes**: Different ways to describe the target object itself
2. **Co-occurring Objects**: Objects that often appear near the target
3. **Room Context**: Rooms or spaces where the target is likely located
4. **Structural Context**: Architectural or layout features associated with the target

For each hypothesis, provide:
- A descriptive text prompt suitable for vision-language models (e.g., "Is there a kitchen ahead?")
- A confidence weight (0.0 to 1.0) indicating how useful this hypothesis is for finding the target

Return your answer in JSON format with the following structure:
{
  "hypotheses": [
    {"prompt": "text prompt for VLM", "weight": 0.9, "type": "target/cooccur/room/structural"},
    ...
  ]
}

Guidelines:
- Generate 4-8 hypotheses total
- Prioritize diverse perspectives over redundant variations
- Weights should roughly sum to the number of hypotheses (will be normalized)
- Use natural language suitable for vision-language understanding
"""

# Few-shot examples
EXAMPLE_USER_1 = "dining table"

EXAMPLE_ASSISTANT_1 = """{
  "hypotheses": [
    {
      "prompt": "Is there a dining table ahead?",
      "weight": 1.0,
      "type": "target"
    },
    {
      "prompt": "Is there a table ahead?",
      "weight": 0.8,
      "type": "target"
    },
    {
      "prompt": "Is there a dining room ahead?",
      "weight": 0.9,
      "type": "room"
    },
    {
      "prompt": "Are there chairs around a table ahead?",
      "weight": 0.7,
      "type": "cooccur"
    },
    {
      "prompt": "Is there a kitchen or dining area ahead?",
      "weight": 0.6,
      "type": "room"
    },
    {
      "prompt": "Is there tableware or dining furniture ahead?",
      "weight": 0.5,
      "type": "cooccur"
    }
  ]
}"""

EXAMPLE_USER_2 = "bed"

EXAMPLE_ASSISTANT_2 = """{
  "hypotheses": [
    {
      "prompt": "Is there a bed ahead?",
      "weight": 1.0,
      "type": "target"
    },
    {
      "prompt": "Is there a bedroom ahead?",
      "weight": 0.9,
      "type": "room"
    },
    {
      "prompt": "Is there sleeping furniture ahead?",
      "weight": 0.7,
      "type": "target"
    },
    {
      "prompt": "Are there pillows or bedding ahead?",
      "weight": 0.6,
      "type": "cooccur"
    },
    {
      "prompt": "Is there a nightstand or dresser ahead?",
      "weight": 0.5,
      "type": "cooccur"
    },
    {
      "prompt": "Is there a private sleeping area ahead?",
      "weight": 0.7,
      "type": "room"
    }
  ]
}"""

EXAMPLE_USER_3 = "refrigerator"

EXAMPLE_ASSISTANT_3 = """{
  "hypotheses": [
    {
      "prompt": "Is there a refrigerator ahead?",
      "weight": 1.0,
      "type": "target"
    },
    {
      "prompt": "Is there a fridge ahead?",
      "weight": 0.9,
      "type": "target"
    },
    {
      "prompt": "Is there a kitchen ahead?",
      "weight": 0.95,
      "type": "room"
    },
    {
      "prompt": "Are there kitchen appliances ahead?",
      "weight": 0.7,
      "type": "cooccur"
    },
    {
      "prompt": "Is there a stove or oven ahead?",
      "weight": 0.6,
      "type": "cooccur"
    },
    {
      "prompt": "Are there kitchen cabinets or countertops ahead?",
      "weight": 0.65,
      "type": "cooccur"
    },
    {
      "prompt": "Is there a cooking or food preparation area ahead?",
      "weight": 0.8,
      "type": "room"
    }
  ]
}"""


def build_semantic_expansion_messages(target_object: str) -> List[Dict[str, str]]:
    """
    Build message list for LLM API to expand semantic prompts.

    Args:
        target_object: Name of the navigation target object

    Returns:
        List of message dicts in OpenAI chat format
    """
    messages = [
        {"role": "system", "content": SEMANTIC_EXPANSION_SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_USER_1},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT_1},
        {"role": "user", "content": EXAMPLE_USER_2},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT_2},
        {"role": "user", "content": EXAMPLE_USER_3},
        {"role": "assistant", "content": EXAMPLE_ASSISTANT_3},
        {"role": "user", "content": target_object}
    ]
    return messages


def parse_semantic_expansion_response(response_text: str) -> List[Tuple[str, float]]:
    """
    Parse LLM response to extract semantic prompts and weights.

    Args:
        response_text: JSON response from LLM

    Returns:
        List of (prompt, weight) tuples
    """
    try:
        # Parse JSON response
        data = json.loads(response_text)
        hypotheses = data.get("hypotheses", [])

        # Extract prompts and weights
        result = []
        for hyp in hypotheses:
            prompt = hyp.get("prompt", "")
            weight = hyp.get("weight", 1.0)
            if prompt:  # Only include non-empty prompts
                result.append((prompt, weight))

        return result

    except json.JSONDecodeError as e:
        print(f"[SemanticExpansion] JSON decode error: {e}")
        print(f"[SemanticExpansion] Response text: {response_text}")
        return []
    except Exception as e:
        print(f"[SemanticExpansion] Error parsing response: {e}")
        return []


def get_semantic_prompts(target_object: str, llm_client) -> List[Tuple[str, float]]:
    """
    Get expanded semantic prompts for a target object using LLM.

    Args:
        target_object: Name of the navigation target
        llm_client: LLM client instance (must have `chat` or `get_answer` method)

    Returns:
        List of (prompt, weight) tuples
    """
    messages = build_semantic_expansion_messages(target_object)

    # Query LLM
    try:
        # Try different LLM client interfaces
        if hasattr(llm_client, 'chat'):
            response = llm_client.chat(messages)
        elif hasattr(llm_client, 'get_answer'):
            # Convert messages to simple prompt for simpler clients
            prompt = messages[-1]["content"]
            response = llm_client.get_answer(prompt)
        else:
            print("[SemanticExpansion] LLM client has no compatible method")
            return []

        # Parse response
        prompts_and_weights = parse_semantic_expansion_response(response)

        print(f"[SemanticExpansion] Generated {len(prompts_and_weights)} semantic prompts for '{target_object}'")
        for prompt, weight in prompts_and_weights:
            print(f"  - [{weight:.2f}] {prompt}")

        return prompts_and_weights

    except Exception as e:
        print(f"[SemanticExpansion] Error querying LLM: {e}")
        return []


def get_default_semantic_prompts(target_object: str) -> List[Tuple[str, float]]:
    """
    Fallback method: Generate basic semantic prompts without LLM.

    Uses simple heuristics to create target and room-based prompts.

    Args:
        target_object: Name of the navigation target

    Returns:
        List of (prompt, weight) tuples
    """
    # Basic target prompt
    prompts = [
        (f"Is there a {target_object} ahead?", 1.0),
        (f"Is there {target_object} ahead?", 0.8),  # Without article
    ]

    # Simple room associations (could be expanded with a lookup table)
    room_associations = {
        "bed": "bedroom",
        "dining table": "dining room",
        "table": "dining room",
        "refrigerator": "kitchen",
        "fridge": "kitchen",
        "stove": "kitchen",
        "oven": "kitchen",
        "toilet": "bathroom",
        "sink": "bathroom",
        "shower": "bathroom",
        "sofa": "living room",
        "couch": "living room",
        "tv": "living room",
        "television": "living room",
    }

    # Add room context if available
    target_lower = target_object.lower()
    for obj, room in room_associations.items():
        if obj in target_lower:
            prompts.append((f"Is there a {room} ahead?", 0.9))
            break

    print(f"[SemanticExpansion] Using default prompts for '{target_object}' (no LLM)")
    for prompt, weight in prompts:
        print(f"  - [{weight:.2f}] {prompt}")

    return prompts


if __name__ == "__main__":
    # Test the semantic expansion
    print("Testing semantic prompt expansion...")

    # Test default prompts (no LLM)
    print("\n=== Testing default prompts ===")
    test_objects = ["bed", "dining table", "refrigerator", "sofa"]
    for obj in test_objects:
        print(f"\nTarget: {obj}")
        prompts = get_default_semantic_prompts(obj)

    # Test message building
    print("\n=== Testing LLM message building ===")
    messages = build_semantic_expansion_messages("microwave")
    print(f"Built {len(messages)} messages")
    print("Last message:", messages[-1])

    # Test JSON parsing
    print("\n=== Testing JSON parsing ===")
    test_response = EXAMPLE_ASSISTANT_1
    parsed = parse_semantic_expansion_response(test_response)
    print(f"Parsed {len(parsed)} hypotheses:")
    for prompt, weight in parsed:
        print(f"  - [{weight:.2f}] {prompt}")
