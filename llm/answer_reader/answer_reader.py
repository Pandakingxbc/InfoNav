import ast
import re
from llm.answer import get_answer


def _safe_parse_list(raw_str):
    """
    安全解析列表字符串，处理各种格式问题。

    支持:
    - 中文引号 '' "" 转换为英文引号
    - 多余空格处理
    - eval 失败时的 fallback
    """
    # 替换中文引号为英文引号
    normalized = raw_str.replace("'", "'").replace("'", "'")
    normalized = normalized.replace(""", '"').replace(""", '"')
    # 清理多余空格
    normalized = normalized.strip()

    try:
        # 优先使用 ast.literal_eval (更安全)
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError) as e:
        print(f"[Warning] ast.literal_eval failed: {e}, trying regex fallback")
        # Fallback: 使用正则提取列表内容
        match = re.search(r'\[(.*)\]', normalized)
        if match:
            items_str = match.group(1)
            # 分割并清理每个元素
            items = []
            for item in items_str.split(','):
                item = item.strip().strip("'\"")
                if item and not _is_legacy_field(item):
                    items.append(item)
            return items
        return []


def _is_legacy_field(item):
    """检查是否为旧格式的遗留字段 (room 或 fusion_score)"""
    # 检查是否为数字 (fusion_score)
    try:
        float(item)
        return True
    except ValueError:
        pass
    # 检查是否为房间类型
    room_keywords = ['room', 'everywhere', 'bedroom', 'bathroom', 'kitchen', 'living']
    return any(kw in item.lower() for kw in room_keywords)


def _filter_legacy_fields(llm_answer):
    """
    过滤旧格式的遗留字段 (room 和 fusion_score)

    新格式: ['obj1', 'obj2', 'obj3']
    旧格式: ['obj1', 'obj2', 0.6, 'living room']
    """
    if not llm_answer:
        return llm_answer

    filtered = []
    for item in llm_answer:
        if isinstance(item, str) and not _is_legacy_field(item):
            filtered.append(item)
        elif isinstance(item, (int, float)):
            # 跳过数字 (fusion_score)
            continue

    return filtered


def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    """
    读取或生成 LLM 答案用于物体检测。

    NOTE: 移除了 room 和 fusion_threshold 的返回值
    - room: HSVM 的多层级假说已经包含了 room_type 信息
    - fusion_threshold: 现在在 src/exploration_manager 中配置

    Args:
        llm_answer_path: LLM答案缓存文件路径
        llm_response_path: LLM完整响应保存路径
        label: 目标物体类别
        llm_client: LLM客户端类型

    Returns:
        llm_answer: 用于物体检测的相关物体列表 (仅包含物体名称)
    """
    label_existing = False
    llm_answer = []

    with open(llm_answer_path, "a+") as f:
        f.seek(0)
        lines = f.readlines()

        for line in lines:
            if line.startswith(f"{label}:"):
                label_existing = True
                raw_str = line[len(label) + 1:].strip()
                llm_answer = _safe_parse_list(raw_str)
                print(f"Already have Answer for {label}: {llm_answer}")
                break

        if not label_existing:
            llm_answer, response = get_answer(prompt=label, client=llm_client)
            print(llm_answer)
            f.write(f"\n{label}: {llm_answer}")
            print(f"New Answer for {label}: {llm_answer}")
            with open(llm_response_path, "a+") as response_file:
                response_file.write(f"\n{label}: {response}")
                print(f"Response saved to {llm_response_path}: {response}")

    # 过滤旧格式遗留字段
    llm_answer = _filter_legacy_fields(llm_answer)

    return llm_answer