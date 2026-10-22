#!/usr/bin/env python3
"""Print or send a concise status report for the HM3Dv2 greedy-1000 run."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".run_state" / "hm3dv2_greedy1000.env"
CC_DATA_DIR = os.environ.get("CC_CONNECT_DATA_DIR", "")
CC_CONNECT = os.environ.get("CC_CONNECT_BIN", "cc-connect")
FEISHU_SESSION_KEY = os.environ.get("CC_CONNECT_SESSION", "")


def read_state() -> dict[str, str]:
    state: dict[str, str] = {}
    if not STATE_FILE.exists():
        return state
    for raw_line in STATE_FILE.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key.strip()] = value.strip()
    return state


def first_match(text: str, pattern: str, default: str = "N/A") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_continue(output_dir: Path) -> dict[str, str]:
    path = output_dir / "continue.txt"
    if not path.exists():
        return {
            "completed": "0",
            "success": "0",
            "spl": "N/A",
            "soft_spl": "N/A",
            "episode_id": "N/A",
            "target": "N/A",
            "result": "N/A",
            "age": "N/A",
        }

    # write_record prepends the newest result, so the first block is sufficient.
    text = path.read_text(errors="replace")[:12000]
    record_path = output_dir / "record.txt"
    record_text = (
        record_path.read_text(errors="replace")[:12000]
        if record_path.exists()
        else ""
    )
    completed = first_match(text, r"No\.(\d+)\s+task is finished", "0")
    success = first_match(text, r"\|\s*Total Success\s*\|\s*([0-9.]+)\s*\|", "0")
    # record.txt stores the already-computed average with two decimal places;
    # continue.txt stores a rounded sum, which loses precision after division.
    spl = first_match(
        record_text, r"\|\s*Average SPL\s*\|\s*([0-9.]+%)\s*\|"
    )
    soft_spl = first_match(
        record_text, r"\|\s*Average Soft SPL\s*\|\s*([0-9.]+%)\s*\|"
    )

    modified = dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    age_minutes = max(
        0, int((dt.datetime.now().astimezone() - modified).total_seconds() // 60)
    )
    return {
        "completed": completed,
        "success": success,
        "spl": spl,
        "soft_spl": soft_spl,
        "episode_id": first_match(text, r"Episode ID:\s*(\S+)"),
        "target": first_match(text, r"target to find is\s+(.+)"),
        "result": first_match(text, r"success or not:\s*(.+)"),
        "age": f"{age_minutes} 分钟前",
    }


def process_running(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def gpu_status(power_limit_max: float = 220.0) -> tuple[bool, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,power.limit,memory.used,"
        "memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "不可访问"
    if result.returncode != 0 or not result.stdout.strip():
        return False, "不可访问（驱动/PCIe 故障）"
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if len(values) != 6:
        return False, "返回格式异常"
    temp, power, power_limit, used, total, util = values
    summary = (
        f"{temp}°C, {power}W/{power_limit}W, "
        f"显存 {used}/{total}MiB, 利用率 {util}%"
    )
    try:
        if float(power_limit) > power_limit_max:
            return False, f"功耗上限过高：{summary}（要求 ≤{power_limit_max:g}W）"
    except ValueError:
        return False, f"功耗上限格式异常：{summary}"
    return True, summary


def build_report() -> str:
    state = read_state()
    output_value = state.get("OUTPUT_DIR", "")
    output_dir = Path(output_value) if output_value else ROOT / "videos" / "未创建"
    metrics = parse_continue(output_dir)
    total = int(state.get("TOTAL", "1000"))
    completed = int(metrics["completed"])
    success = int(float(metrics["success"]))
    failed = max(0, completed - success)
    sr = f"{success / completed * 100:.2f}%" if completed else "N/A"
    progress = f"{completed / total * 100:.1f}%" if total else "N/A"

    power_limit_max = float(state.get("GPU_POWER_LIMIT_MAX", "220"))
    gpu_ok, gpu = gpu_status(power_limit_max)
    eval_running = process_running(
        r"/python([0-9.]*)? .*habitat_evaluation\.py"
    )
    planner_running = process_running(r"/exploration_node([ ]|$)")
    verifier_running = process_running(
        r"/python([0-9.]*)? .*vlm_approach_verifier\.py"
    )
    services = {
        "BLIP2": port_open(12182),
        "MobileSAM": port_open(12183),
        "D-FINE": port_open(12185),
    }
    services_text = ", ".join(
        f"{name}={'正常' if healthy else '未运行'}"
        for name, healthy in services.items()
    )

    run_status = state.get("STATUS", "not_prepared")
    if completed >= total:
        human_status = "已完成"
    elif not gpu_ok:
        human_status = "暂停：GPU 不可访问"
    elif eval_running and planner_running:
        human_status = "运行中"
    elif run_status == "starting":
        human_status = "服务启动中"
    else:
        human_status = f"未运行（{run_status}）"

    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return "\n".join(
        [
            "[InfoNav HM3Dv2 Greedy-1000 定时播报]",
            f"时间：{now}",
            "策略：policy=1，value-greedy（价值地图贪心）",
            (
                f"负载：{state.get('CONTROL_RATE_HZ', '4')}Hz 控制；"
                "本地仅 BLIP2 + MobileSAM + D-FINE；"
                f"Qwen={state.get('QWEN_VL_MODEL', 'qwen3-vl-plus')}"
            ),
            f"状态：{human_status}",
            f"进度：{completed}/{total}（{progress}）",
            f"结果：成功 {success}，失败 {failed}，SR {sr}",
            f"SPL：{metrics['spl']}；Soft SPL：{metrics['soft_spl']}",
            (
                f"最近：Episode {metrics['episode_id']}，目标 {metrics['target']}，"
                f"结果 {metrics['result']}"
            ),
            f"最后写入：{metrics['age']}",
            (
                "进程："
                f"eval={'运行' if eval_running else '未运行'}，"
                f"planner={'运行' if planner_running else '未运行'}，"
                f"verifier={'运行' if verifier_running else '未运行'}"
            ),
            f"VLM：{services_text}",
            f"GPU：{gpu}",
            f"目录：{output_dir}",
        ]
    )


def send_report(report: str) -> int:
    if not CC_DATA_DIR or not FEISHU_SESSION_KEY:
        print(
            "Set CC_CONNECT_DATA_DIR and CC_CONNECT_SESSION before using --send.",
            file=sys.stderr,
        )
        return 2
    command = [
        CC_CONNECT,
        "send",
        "--data-dir",
        CC_DATA_DIR,
        "--project",
        "claude_bridge",
        "--session",
        FEISHU_SESSION_KEY,
        "--stdin",
    ]
    result = subprocess.run(command, input=report, text=True, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--send", action="store_true", help="send through the active cc-connect session"
    )
    args = parser.parse_args()
    report = build_report()
    if args.send:
        return send_report(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
