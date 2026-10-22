#!/usr/bin/env python3
"""Report the active HM3Dv2 value-greedy/WTRP paired experiment."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "videos"
STATE_DIR = ROOT / ".run_state"
CRON_ID = os.environ.get("CC_CONNECT_CRON_ID", "")
CC_DATA_DIR = os.environ.get("CC_CONNECT_DATA_DIR", "")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(errors="replace").splitlines():
        if "=" not in raw_line or raw_line.lstrip().startswith("#"):
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def latest_pair_root() -> Path | None:
    candidates = []
    for path in VIDEOS.glob("test_hm3dv2_value_wtrp_pair100_*_val"):
        protocol = read_env(path / "value_greedy" / "run_protocol.env")
        if protocol.get("TOTAL_EPISODES") == "100":
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_metrics(path: Path) -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (str(row["scene_id"]), str(row["episode_id"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        rows[key] = row
    return sorted(rows.values(), key=lambda row: int(row.get("episode_num", 0)))


def arm_summary(root: Path, name: str, state: dict[str, str]) -> dict[str, object]:
    metrics_path = root / name / "episode_metrics.jsonl"
    rows = load_metrics(metrics_path)
    completed = len(rows)
    success = sum(int(float(row.get("success", 0))) for row in rows)
    spl = sum(float(row.get("spl", 0)) for row in rows) / completed if completed else None
    soft_spl = (
        sum(float(row.get("soft_spl", 0)) for row in rows) / completed
        if completed
        else None
    )
    age = "N/A"
    if metrics_path.exists():
        modified = dt.datetime.fromtimestamp(metrics_path.stat().st_mtime).astimezone()
        minutes = max(
            0,
            int((dt.datetime.now().astimezone() - modified).total_seconds() // 60),
        )
        age = f"{minutes} 分钟前"
    return {
        "completed": completed,
        "success": success,
        "failed": completed - success,
        "sr": 100.0 * success / completed if completed else None,
        "spl": 100.0 * spl if spl is not None else None,
        "soft_spl": 100.0 * soft_spl if soft_spl is not None else None,
        "latest": rows[-1] if rows else {},
        "age": age,
        "status": state.get("STATUS", "not_started"),
    }


def percent(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


def process_running(pattern: str) -> bool:
    return subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def tmux_running(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def gpu_status() -> str:
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
        return "不可访问"
    values = [item.strip() for item in result.stdout.strip().split(",")]
    if result.returncode != 0 or len(values) != 6:
        return "不可访问（驱动或 PCIe 异常）"
    temp, power, limit, used, total, util = values
    return f"{temp}°C, {power}/{limit}W, 显存 {used}/{total}MiB, 利用率 {util}%"


def latest_text(summary: dict[str, object]) -> str:
    latest = summary["latest"]
    if not latest:
        return "N/A"
    return (
        f"Episode {latest.get('episode_id', 'N/A')}，"
        f"目标 {latest.get('target', 'N/A')}，"
        f"结果 {latest.get('result', 'N/A')}，"
        f"{int(latest.get('steps', 0))} 步"
    )


def disable_after_completion() -> None:
    if not CRON_ID or not CC_DATA_DIR:
        return
    subprocess.run(
        [
            "cc-connect",
            "cron",
            "edit",
            CRON_ID,
            "enabled",
            "false",
            "--data-dir",
            CC_DATA_DIR,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> int:
    pair_root = latest_pair_root()
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    if pair_root is None:
        print(f"[InfoNav HM3Dv2 Value-Greedy vs WTRP 定时播报]\n时间：{now}\n状态：未找到配对实验目录")
        return 1

    match = re.search(r"pair100_(\d{8}_\d{6})_val$", pair_root.name)
    stamp = match.group(1) if match else "unknown"
    greedy_state = read_env(STATE_DIR / f"hm3dv2_value_greedy100_{stamp}.env")
    wtrp_state = read_env(STATE_DIR / f"hm3dv2_wtrp100_{stamp}.env")
    greedy = arm_summary(pair_root, "value_greedy", greedy_state)
    wtrp = arm_summary(pair_root, "wtrp", wtrp_state)

    greedy_n = int(greedy["completed"])
    wtrp_n = int(wtrp["completed"])
    overall = greedy_n + wtrp_n
    if wtrp_n >= 100:
        active = "全部完成"
    elif wtrp_state.get("STATUS") in {"starting", "running"}:
        active = "B：WTRP（policy=4）"
    elif greedy_state.get("STATUS") in {"starting", "running"}:
        active = "A：价值贪心（policy=1）"
    elif greedy_n >= 100:
        active = "A 已完成，正在切换到 B"
    else:
        active = "未运行或异常停止"

    services = {
        "BLIP2": port_open(12182),
        "GroundingDINO": port_open(12181),
        "MobileSAM": port_open(12183),
        "D-FINE": port_open(12185),
    }
    services_text = "，".join(
        f"{name}={'正常' if healthy else '异常'}" for name, healthy in services.items()
    )
    eval_running = process_running(r"habitat_evaluation\.py")
    planner_running = process_running(r"/exploration_node([ ]|$)")
    manager_running = tmux_running("infonav_value_wtrp_pair100")

    lines = [
        "[InfoNav HM3Dv2 Value-Greedy vs WTRP 定时播报]",
        f"时间：{now}",
        "协议：前 100 个相同 episode，seed=100，success distance=1m；归一化多源 HSVM+IG",
        f"当前实验臂：{active}",
        f"总进度：{overall}/200（{overall / 2:.1f}%）",
        (
            f"A 价值贪心：{greedy_n}/100，成功 {greedy['success']}，"
            f"失败 {greedy['failed']}，SR {percent(greedy['sr'])}，"
            f"SPL {percent(greedy['spl'])}，Soft SPL {percent(greedy['soft_spl'])}"
        ),
        (
            f"B WTRP：{wtrp_n}/100，成功 {wtrp['success']}，"
            f"失败 {wtrp['failed']}，SR {percent(wtrp['sr'])}，"
            f"SPL {percent(wtrp['spl'])}，Soft SPL {percent(wtrp['soft_spl'])}"
        ),
        f"最近：{latest_text(wtrp if wtrp_n else greedy)}",
        f"最后写入：{(wtrp if wtrp_n else greedy)['age']}",
        (
            f"进程：eval={'运行' if eval_running else '未运行'}，"
            f"planner={'运行' if planner_running else '未运行'}，"
            f"pair-manager={'运行' if manager_running else '未运行'}"
        ),
        f"VLM：{services_text}",
        f"GPU：{gpu_status()}",
        f"目录：{pair_root}",
    ]
    print("\n".join(lines))
    if greedy_n >= 100 and wtrp_n >= 100:
        disable_after_completion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
