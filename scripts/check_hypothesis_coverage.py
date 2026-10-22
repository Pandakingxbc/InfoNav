#!/usr/bin/env python3
"""Validate the offline per-episode hypotheses used by InfoNav evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


EXPECTED_TYPES = {"room_type", "target_object", "co_occurrence", "object_part"}


def compact_ranges(values: Iterable[int]) -> str:
    numbers = sorted(set(values))
    if not numbers:
        return "none"

    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def validate_file(path: Path, require_ig: bool) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable JSON ({exc})"]

    hypotheses = data.get("semantic_hypotheses")
    if not isinstance(hypotheses, list):
        return ["semantic_hypotheses is not a list"]

    errors: list[str] = []
    types: list[str] = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            errors.append(f"semantic_hypotheses[{index}] is not an object")
            continue
        hypothesis_type = hypothesis.get("type", hypothesis.get("hypothesis_type"))
        types.append(hypothesis_type)
        prompt = hypothesis.get("prompt", hypothesis.get("description"))
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"semantic_hypotheses[{index}] has no prompt")

    if len(hypotheses) != 4:
        errors.append(f"expected 4 semantic hypotheses, found {len(hypotheses)}")
    if set(types) != EXPECTED_TYPES or len(types) != len(set(types)):
        errors.append(
            "semantic hypothesis types must be exactly "
            + ",".join(sorted(EXPECTED_TYPES))
        )

    if require_ig:
        ig_hypotheses = data.get("ig_hypotheses")
        if not isinstance(ig_hypotheses, list) or not ig_hypotheses:
            errors.append("ig_hypotheses is missing or empty")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check task0..taskN offline InfoNav hypothesis coverage"
    )
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=Path("env/env_hm3dv2"),
        help="directory containing task<N> folders",
    )
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument(
        "--require-ig",
        action="store_true",
        help="also require at least one information-gain hypothesis",
    )
    args = parser.parse_args()

    if args.total <= 0:
        parser.error("--total must be positive")

    missing: list[int] = []
    invalid: dict[int, list[str]] = {}
    for task_index in range(args.total):
        path = args.env_dir / f"task{task_index}" / "llm_hypothesis_analysis.json"
        if not path.is_file():
            missing.append(task_index)
            continue
        errors = validate_file(path, args.require_ig)
        if errors:
            invalid[task_index] = errors

    valid = args.total - len(missing) - len(invalid)
    print(
        f"[hypotheses] valid={valid}/{args.total} "
        f"missing={len(missing)} invalid={len(invalid)} dir={args.env_dir}"
    )
    if missing:
        print(f"[hypotheses] missing tasks: {compact_ranges(missing)}")
    if invalid:
        preview = list(sorted(invalid.items()))[:20]
        for task_index, errors in preview:
            print(f"[hypotheses] invalid task{task_index}: {'; '.join(errors)}")
        if len(invalid) > len(preview):
            print(f"[hypotheses] ... {len(invalid) - len(preview)} more invalid tasks")

    return 0 if valid == args.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
