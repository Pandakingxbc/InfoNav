#!/usr/bin/env python3
"""Compare two same-ID HM3D episode_metrics.jsonl files."""

import argparse
import json
import math
from pathlib import Path


def load(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scene_id"]), str(row["episode_id"]))
            if key in rows:
                raise ValueError(f"duplicate episode key in {path}: {key}")
            rows[key] = row
    return rows


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else math.nan


def exact_mcnemar_p(discordant_a, discordant_b):
    """Two-sided exact McNemar p-value (binomial test, p=0.5)."""
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    tail = sum(math.comb(total, k) for k in range(lower + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("value_greedy", type=Path)
    parser.add_argument("wtrp", type=Path)
    args = parser.parse_args()

    greedy = load(args.value_greedy)
    wtrp = load(args.wtrp)
    greedy_keys, wtrp_keys = set(greedy), set(wtrp)
    if greedy_keys != wtrp_keys:
        only_greedy = sorted(greedy_keys - wtrp_keys)
        only_wtrp = sorted(wtrp_keys - greedy_keys)
        raise SystemExit(
            f"episode-ID mismatch: only_greedy={only_greedy[:5]}, only_wtrp={only_wtrp[:5]}"
        )

    keys = sorted(greedy_keys)
    greedy_rows = [greedy[key] for key in keys]
    wtrp_rows = [wtrp[key] for key in keys]
    g_win = w_win = same = 0
    for key in keys:
        g_success = int(greedy[key]["success"])
        w_success = int(wtrp[key]["success"])
        if g_success > w_success:
            g_win += 1
        elif w_success > g_success:
            w_win += 1
        else:
            same += 1

    print("HM3Dv2 value-greedy vs WTRP paired comparison")
    print(f"paired episodes: {len(keys)}")
    print(
        f"value-greedy: SR={100 * mean(greedy_rows, 'success'):.2f}% "
        f"SPL={100 * mean(greedy_rows, 'spl'):.2f}% "
        f"SoftSPL={100 * mean(greedy_rows, 'soft_spl'):.2f}%"
    )
    print(
        f"WTRP:         SR={100 * mean(wtrp_rows, 'success'):.2f}% "
        f"SPL={100 * mean(wtrp_rows, 'spl'):.2f}% "
        f"SoftSPL={100 * mean(wtrp_rows, 'soft_spl'):.2f}%"
    )
    print(f"paired success outcomes: greedy-only={g_win}, WTRP-only={w_win}, same={same}")
    print(f"exact McNemar p={exact_mcnemar_p(g_win, w_win):.6g}")


if __name__ == "__main__":
    main()
