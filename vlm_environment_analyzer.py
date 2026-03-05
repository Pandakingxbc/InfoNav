#!/usr/bin/env python3
"""
使用说明：2026-2-1
#python vlm_environment_analyzer.py single --env hm3dv2 --task 0 --api-type dashscope --model qwen3-vl-flash --env-base-dir /home/yangz/Nav/InfoNav/env

VLM 环境感知分析工具 (Unified Version)
=====================================

整合了单任务VLM分析和批量分析功能，使用VLM（如Qwen3-VL）分析环境图片，
生成环境感知信息供后续LLM假说分析使用。

功能模式:
---------
1. 单任务模式 (single): 分析单个任务的4视角图片
2. 批量模式 (batch): 批量分析多个任务

使用示例:
---------
# 单任务分析 (Ollama 本地模式)
python vlm_environment_analyzer.py single --env hm3dv2 --task 0
python vlm_environment_analyzer.py single --env hm3dv2 --task 0 --api-url http://localhost:20004/api/chat

# 单任务分析 (DashScope 阿里云模式)
export DASHSCOPE_API_KEY="your-api-key"
python vlm_environment_analyzer.py single --env hm3dv2 --task 0 --api-type dashscope --model qwen3-vl-flash

# 批量分析 (Ollama)
python vlm_environment_analyzer.py batch --env hm3dv2
python vlm_environment_analyzer.py batch --env hm3dv2 --max-tasks 10
python vlm_environment_analyzer.py batch --env hm3dv2 --start 5 --end 15
python vlm_environment_analyzer.py batch --env hm3dv1 hm3dv2 --model qwen3-vl:32b

# 批量分析 (DashScope)
python vlm_environment_analyzer.py batch --env hm3dv2 --api-type dashscope --model qwen3-vl-flash

输入文件:
---------
- task_info.txt: 任务信息（目标物体等）
- view0.png ~ view3.png: 4视角环境图片
  - view0.png: 前方 (front)
  - view1.png: 左方 (left)
  - view2.png: 后方 (back)
  - view3.png: 右方 (right)

输出文件:
---------
- vlm_analysis.txt: VLM环境分析结果
- batch_eval_summary.txt: 批量分析摘要（仅批量模式）

依赖:
-----
- Ollama模式: 需要VLM服务（如Ollama + qwen3-vl）运行在指定端口
- DashScope模式: 需要设置 DASHSCOPE_API_KEY 环境变量，安装 openai 库 (pip install openai)
- 需要任务目录包含 task_info.txt 和 4张视角图片

DashScope 支持的模型:
-------------------
- qwen-vl-plus: 通用视觉语言模型
- qwen-vl-max: 高性能视觉语言模型
- qwen3-vl-flash: 快速推理模型 (推荐)

作者: yangzhi
日期: 2024-12
"""

import base64
import requests
import time
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI


# ==================== 工具函数 ====================

def encode_image(path: Path) -> str:
    """将图片编码为 base64 字符串"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_task_info(task_dir: Path) -> Dict[str, str]:
    """从 task_info.txt 读取任务信息"""
    info_file = task_dir / "task_info.txt"
    task_info = {}

    if not info_file.exists():
        return task_info

    with open(info_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                task_info[key] = value

    return task_info


def build_prompt(target_object: str) -> str:
    """构建环境感知 prompt - 为LLM全局推理提供场景上下文"""
    prompt = f"""You are helping a robot find **{target_object}** in an indoor environment.
Describe the scene to help plan a building-wide search:
1. **Scene**: What building type (house/apartment/office/etc)? What room is this?
2. **Layout**: Describe visible doors, hallways, or passages. Where might they lead?
3. **Objects**: Key furniture/items visible. Any clues about nearby rooms?
4. **Global hint**: In this building, where would {target_object} typically be?
"""
    return prompt


# ==================== 核心分析类 ====================

class VLMEnvironmentAnalyzer:
    """VLM环境感知分析器"""

    # 支持的 API 后端类型
    API_OLLAMA = "ollama"
    API_DASHSCOPE = "dashscope"

    def __init__(self,
                 api_url: str = "http://localhost:20004/api/chat",
                 model: str = "qwen3-vl:32b",
                 env_base_dir: str = "env",
                 timeout: int = 300,
                 api_type: str = "ollama"):
        """
        Args:
            api_url: VLM API地址 (Ollama模式) 或 base_url (DashScope模式)
            model: 模型名称
            env_base_dir: 环境数据集根目录
            timeout: 请求超时时间（秒）
            api_type: API类型 ("ollama" 或 "dashscope")
        """
        self.api_url = api_url
        self.model = model
        self.env_base_dir = Path(env_base_dir)
        self.timeout = timeout
        self.api_type = api_type

        # 初始化 DashScope 客户端
        if self.api_type == self.API_DASHSCOPE:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("使用 DashScope API 需要设置 DASHSCOPE_API_KEY 环境变量")
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=api_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def load_images(self, task_dir: Path) -> Tuple[List[Path], List[str]]:
        """
        加载4张视角图片

        Args:
            task_dir: 任务目录

        Returns:
            (image_paths, missing_images)
        """
        image_paths = []
        missing = []

        for i in range(4):
            img_path = task_dir / f"view{i}.png"
            if img_path.exists():
                image_paths.append(img_path)
            else:
                missing.append(f"view{i}.png")

        return image_paths, missing

    def call_vlm(self, prompt: str, image_paths: List[Path], temperature: float = 0.2) -> Tuple[str, float]:
        """
        调用VLM API进行分析

        Args:
            prompt: 分析prompt
            image_paths: 图片路径列表
            temperature: 生成温度

        Returns:
            (response_content, duration)
        """
        if self.api_type == self.API_DASHSCOPE:
            return self._call_dashscope(prompt, image_paths, temperature)
        else:
            return self._call_ollama(prompt, image_paths, temperature)

    def _call_ollama(self, prompt: str, image_paths: List[Path], temperature: float = 0.2) -> Tuple[str, float]:
        """调用 Ollama 本地 API"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encode_image(p) for p in image_paths]
                }
            ],
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        t0 = time.time()
        resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
        duration = time.time() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"VLM API返回错误: {resp.status_code} - {resp.text}")

        result = resp.json()
        content = result.get("message", {}).get("content", "")

        if not content:
            raise RuntimeError("VLM返回空响应")

        return content, duration

    def _call_dashscope(self, prompt: str, image_paths: List[Path], temperature: float = 0.2) -> Tuple[str, float]:
        """调用 DashScope API (阿里云通义千问)"""
        # 构建多模态消息内容
        content = []

        # 添加图片 (使用 base64 data URL 格式)
        for img_path in image_paths:
            img_base64 = encode_image(img_path)
            # 根据文件扩展名确定 MIME 类型
            suffix = img_path.suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(suffix, 'image/png')

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
            })

        # 添加文本 prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        t0 = time.time()
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
        )
        duration = time.time() - t0

        response_content = completion.choices[0].message.content

        if not response_content:
            raise RuntimeError("VLM返回空响应")

        return response_content, duration

    def analyze_single(self,
                       env_name: str,
                       task_num: int,
                       temperature: float = 0.2,
                       output_file: Optional[Path] = None) -> Tuple[bool, Optional[str], float, Optional[str]]:
        """
        分析单个任务

        Args:
            env_name: 环境名称
            task_num: 任务编号
            temperature: 生成温度
            output_file: 输出文件路径（可选）

        Returns:
            (success, content, duration, error_msg)
        """
        task_dir = self.env_base_dir / f"env_{env_name}" / f"task{task_num}"

        print("=" * 70)
        print(f"VLM 环境感知分析")
        print(f"环境: {env_name}")
        print(f"任务: {task_num}")
        print(f"目录: {task_dir}")
        print("=" * 70)

        # 检查任务目录
        if not task_dir.exists():
            return False, None, 0, f"任务目录不存在: {task_dir}"

        # 1. 读取任务信息
        print("\n[1/4] 读取任务信息...")
        task_info = load_task_info(task_dir)
        target_object = task_info.get('target_object', 'unknown')

        print(f"  Task Index: {task_info.get('task_index', 'N/A')}")
        print(f"  Episode ID: {task_info.get('episode_id', 'N/A')}")
        print(f"  Target Object: {target_object}")
        print(f"  Scene ID: {task_info.get('scene_id', 'N/A')}")

        # 2. 加载4张视角图片
        print("\n[2/4] 加载环境图片...")
        image_paths, missing = self.load_images(task_dir)

        for path in image_paths:
            print(f"  OK: {path.name}")
        for name in missing:
            print(f"  MISSING: {name}")

        if len(image_paths) != 4:
            return False, None, 0, f"需要4张图片，但只找到 {len(image_paths)} 张"

        # 3. 构建 prompt
        print("\n[3/4] 构建分析 prompt...")
        prompt = build_prompt(target_object)
        print(f"  Prompt 长度: {len(prompt)} 字符")

        # 4. 调用 VLM API
        print(f"\n[4/4] 调用 VLM API 进行分析...")
        print(f"  模型: {self.model}")
        print(f"  API: {self.api_url}")
        print(f"  等待响应中... (超时设置: {self.timeout}秒)")

        try:
            content, duration = self.call_vlm(prompt, image_paths, temperature)

            print(f"  HTTP状态: 200")
            print(f"  耗时: {duration:.2f} 秒")
            print(f"  输出长度: {len(content)} 字符")

        except requests.Timeout:
            return False, None, 0, f"请求超时 (超过{self.timeout}秒)"
        except Exception as e:
            return False, None, 0, str(e)

        # 5. 显示和保存结果
        print("\n" + "=" * 70)
        print("VLM 分析结果")
        print("=" * 70)
        print(content)
        print("=" * 70)

        # 确定输出文件路径
        if output_file is None:
            output_file = task_dir / "vlm_analysis.txt"

        # 保存输出
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"环境: {env_name}\n")
            f.write(f"任务: {task_num}\n")
            f.write(f"目标物体: {target_object}\n")
            f.write(f"耗时: {duration:.2f} 秒\n")
            f.write("=" * 70 + "\n")
            f.write(content)

        print(f"\n结果已保存到: {output_file}")

        return True, content, duration, None


# ==================== 批量处理类 ====================

class BatchEvaluator:
    """批量评估器"""

    def __init__(self, analyzer: VLMEnvironmentAnalyzer, delay: int = 30):
        """
        Args:
            analyzer: VLM分析器实例
            delay: 任务间隔时间（秒）
        """
        self.analyzer = analyzer
        self.delay = delay

    def discover_tasks(self, env_name: str) -> List[int]:
        """发现指定环境下的所有有效任务"""
        env_dir = self.analyzer.env_base_dir / f"env_{env_name}"

        if not env_dir.exists():
            print(f"警告: 环境目录不存在: {env_dir}")
            return []

        tasks = []
        for task_dir in sorted(env_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith('task'):
                try:
                    task_num = int(task_dir.name[4:])
                    # 检查是否包含必要文件
                    if (task_dir / "task_info.txt").exists():
                        tasks.append(task_num)
                except ValueError:
                    continue

        return sorted(tasks)

    def evaluate_batch(self,
                       env_name: str,
                       start_task: Optional[int] = None,
                       end_task: Optional[int] = None,
                       max_tasks: Optional[int] = None,
                       temperature: float = 0.2) -> Dict:
        """
        批量评估环境数据集

        Args:
            env_name: 环境名称
            start_task: 起始任务编号（包含）
            end_task: 结束任务编号（包含）
            max_tasks: 最大任务数量限制
            temperature: 生成温度

        Returns:
            统计信息字典
        """
        print(f"\n{'#'*70}")
        print(f"# 批量VLM评估: env_{env_name}")
        print(f"# VLM模型: {self.analyzer.model}")
        print(f"# API: {self.analyzer.api_url}")
        print(f"# 任务间隔: {self.delay} 秒")
        print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")

        # 发现所有任务
        all_tasks = self.discover_tasks(env_name)

        if not all_tasks:
            print(f"错误: 在 env_{env_name} 中未找到任何有效任务")
            return None

        print(f"发现 {len(all_tasks)} 个任务: {all_tasks[:20]}{'...' if len(all_tasks) > 20 else ''}")

        # 应用任务范围过滤
        if start_task is not None:
            all_tasks = [t for t in all_tasks if t >= start_task]
        if end_task is not None:
            all_tasks = [t for t in all_tasks if t <= end_task]
        if max_tasks is not None:
            all_tasks = all_tasks[:max_tasks]

        print(f"将评估 {len(all_tasks)} 个任务\n")

        # 统计信息
        stats = {
            'total': len(all_tasks),
            'success': 0,
            'failed': 0,
            'total_time': 0,
            'results': []
        }

        # 逐个评估任务
        for idx, task_num in enumerate(all_tasks, 1):
            print(f"\n进度: [{idx}/{len(all_tasks)}]")

            success, content, duration, error_msg = self.analyzer.analyze_single(
                env_name, task_num, temperature=temperature
            )

            stats['results'].append({
                'task': task_num,
                'success': success,
                'duration': duration,
                'error': error_msg
            })

            stats['total_time'] += duration

            if success:
                stats['success'] += 1
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
        """保存评估摘要"""
        summary_file = self.analyzer.env_base_dir / f"env_{env_name}" / "batch_eval_summary.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"批量VLM评估摘要: env_{env_name}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"VLM模型: {self.analyzer.model}\n")
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
        """打印评估摘要"""
        print(f"\n{'#'*70}")
        print(f"# 评估完成: env_{env_name}")
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


# ==================== 命令行接口 ====================

def cmd_single(args):
    """单任务分析命令"""
    analyzer = VLMEnvironmentAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir,
        timeout=args.timeout,
        api_type=args.api_type
    )

    output_file = Path(args.output) if args.output else None

    success, content, duration, error_msg = analyzer.analyze_single(
        args.env, args.task, temperature=args.temperature, output_file=output_file
    )

    if not success:
        print(f"\n错误: {error_msg}")
        return 1

    return 0


def cmd_batch(args):
    """批量分析命令"""
    analyzer = VLMEnvironmentAnalyzer(
        api_url=args.api_url,
        model=args.model,
        env_base_dir=args.env_base_dir,
        timeout=args.timeout,
        api_type=args.api_type
    )

    batch = BatchEvaluator(analyzer, delay=args.delay)

    all_stats = {}

    for env_name in args.env:
        stats = batch.evaluate_batch(
            env_name,
            start_task=args.start,
            end_task=args.end,
            max_tasks=args.max_tasks,
            temperature=args.temperature
        )

        if stats:
            all_stats[env_name] = stats
            batch.print_summary(env_name, stats)
            batch.save_summary(env_name, stats)

            # 如果不是最后一个环境，等待
            if env_name != args.env[-1]:
                print(f"\n{'='*70}")
                print(f"准备评估下一个环境...")
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

        print(f"评估环境数: {len(all_stats)}")
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
    print("所有评估任务完成!")
    print(f"{'='*70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VLM 环境感知分析工具 - 支持单任务和批量分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
=========

单任务分析:
  python vlm_environment_analyzer.py single --env hm3dv2 --task 0
  python vlm_environment_analyzer.py single --env hm3dv2 --task 0 --api-url http://localhost:20004/api/chat
  python vlm_environment_analyzer.py single --env hm3dv2 --task 0 --output result.txt

批量分析:
  python vlm_environment_analyzer.py batch --env hm3dv2
  python vlm_environment_analyzer.py batch --env hm3dv2 --max-tasks 10
  python vlm_environment_analyzer.py batch --env hm3dv2 --start 5 --end 15
  python vlm_environment_analyzer.py batch --env hm3dv1 hm3dv2 --delay 60

工作流程:
=========
1. 先运行本工具生成 vlm_analysis.txt
2. 再运行 llm_hypothesis_analyzer.py 生成假说分析

完整示例:
  # Step 1: VLM环境分析
  python vlm_environment_analyzer.py batch --env hm3dv2 --max-tasks 10

  # Step 2: LLM假说分析
  python llm_hypothesis_analyzer.py batch --env hm3dv2 --max-tasks 10
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
    single_parser.add_argument('--api-type', type=str, default='ollama',
                               choices=['ollama', 'dashscope'],
                               help='API类型: ollama(本地) 或 dashscope(阿里云)')
    single_parser.add_argument('--api-url', type=str, default=None,
                               help='VLM API 地址 (ollama默认: localhost:20004, dashscope默认: 阿里云)')
    single_parser.add_argument('--model', type=str, default=None,
                               help='VLM 模型名称 (ollama默认: qwen3-vl:32b, dashscope默认: qwen-vl-max)')
    single_parser.add_argument('--temperature', type=float, default=0.2,
                               help='生成温度 0.0-1.0 (默认: 0.2)')
    single_parser.add_argument('--timeout', type=int, default=600,
                               help='请求超时时间（秒），默认600秒')
    single_parser.add_argument('--env-base-dir', type=str, default='env',
                               help='环境数据集根目录')
    single_parser.add_argument('--output', type=str, default=None,
                               help='输出文件路径（默认自动保存到任务目录）')

    # 批量模式
    batch_parser = subparsers.add_parser('batch', help='批量分析模式')
    batch_parser.add_argument('--env', type=str, nargs='+', required=True,
                              choices=['hm3dv1', 'hm3dv2', 'mp3d'],
                              help='要评估的环境（可指定多个）')
    batch_parser.add_argument('--start', type=int, default=None,
                              help='起始任务编号（包含）')
    batch_parser.add_argument('--end', type=int, default=None,
                              help='结束任务编号（包含）')
    batch_parser.add_argument('--max-tasks', type=int, default=None,
                              help='最大任务数量限制')
    batch_parser.add_argument('--delay', type=int, default=20,
                              help='任务间隔时间（秒），默认20秒')
    batch_parser.add_argument('--api-type', type=str, default='ollama',
                              choices=['ollama', 'dashscope'],
                              help='API类型: ollama(本地) 或 dashscope(阿里云)')
    batch_parser.add_argument('--api-url', type=str, default=None,
                              help='VLM API 地址 (ollama默认: localhost:20004, dashscope默认: 阿里云)')
    batch_parser.add_argument('--model', type=str, default=None,
                              help='VLM 模型名称 (ollama默认: qwen3-vl:32b, dashscope默认: qwen-vl-max)')
    batch_parser.add_argument('--temperature', type=float, default=0.2,
                              help='生成温度 0.0-1.0 (默认: 0.2)')
    batch_parser.add_argument('--timeout', type=int, default=600,
                              help='请求超时时间（秒），默认600秒')
    batch_parser.add_argument('--env-base-dir', type=str, default='env',
                              help='环境数据集根目录')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # 根据 api_type 设置默认值
    if args.api_type == 'dashscope':
        if args.api_url is None:
            args.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if args.model is None:
            args.model = "qwen-vl-max"  # 可选: qwen-vl-plus, qwen-vl-max, qwen3-vl-flash
    else:  # ollama
        if args.api_url is None:
            args.api_url = "http://localhost:20004/api/chat"
        if args.model is None:
            args.model = "qwen3-vl:32b"

    if args.command == 'single':
        return cmd_single(args)
    elif args.command == 'batch':
        return cmd_batch(args)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断评估")
        sys.exit(1)
