#!/usr/bin/env python3
"""Summarize Kuafu MPC worker success counts without opening live HDF5 files."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


SUCCESS_RE = re.compile(rb"ENV finished with outcome=true_success")
SAVED_RE = re.compile(rb"Total success trajectory saved:\s*(\d+)")
TRACEBACK_RE = re.compile(rb"Traceback", re.IGNORECASE)
HANDLED_MPC_RE = re.compile(rb"Resetting episode and continuing\.", re.IGNORECASE)
FATAL_RE = re.compile(
    rb"CUDA out of memory|Segmentation fault|worker[^\n]*failed",
    re.IGNORECASE,
)


def tail_bytes(path: Path, limit: int = 8 * 1024 * 1024) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read()


def classify(root: Path, log_path: Path) -> tuple[str, str, str, str, int]:
    relative = log_path.relative_to(root)
    parts = relative.parts
    task = parts[0]
    phase = "legacy_tranche_01"
    for part in parts:
        if part.startswith("tranche_"):
            phase = part
            break
    program = next(
        (match.group() for part in parts if (match := re.match(r"program_\d\d", part))),
        "program_??",
    )
    attempt = "legacy"
    if "runs" in parts:
        run_index = parts.index("runs")
        if run_index + 1 < len(parts):
            attempt = parts[run_index + 1]
    match = re.search(r"worker_(\d+)\.txt$", log_path.name)
    worker = int(match.group(1)) if match else -1
    return task, phase, program, attempt, worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()

    groups: dict[
        tuple[str, str, str, str], list[tuple[int, int, int, int]]
    ] = defaultdict(list)
    for log_path in sorted(root.glob("**/logs/worker_*.txt")):
        task, phase, program, attempt, worker = classify(root, log_path)
        content = tail_bytes(log_path)
        saved_matches = SAVED_RE.findall(content)
        saved = (
            int(saved_matches[-1])
            if saved_matches
            else len(SUCCESS_RE.findall(content))
        )
        tracebacks = len(TRACEBACK_RE.findall(content))
        handled_mpc = len(HANDLED_MPC_RE.findall(content))
        fatal_count = len(FATAL_RE.findall(content)) + max(
            0, tracebacks - handled_mpc
        )
        groups[(task, phase, program, attempt)].append(
            (worker, saved, fatal_count, handled_mpc)
        )

    if not groups:
        print("no worker logs found")
    for (task, phase, program, attempt), rows in sorted(groups.items()):
        rows.sort()
        counts = [count for _, count, _, _ in rows]
        fatals = sum(fatal for _, _, fatal, _ in rows)
        handled_mpc = sum(handled for _, _, _, handled in rows)
        print(
            f"{task:14s} {phase:20s} {program} "
            f"success={sum(counts):4d}/{len(rows) * 100:<4d} "
            f"workers={counts} fatals={fatals} handled_mpc={handled_mpc} "
            f"attempt={attempt}"
        )

    completions = sorted(root.glob("*/completion_mpc1600.json"))
    if completions:
        print("completed_mpc1600=" + ",".join(path.parent.name for path in completions))


if __name__ == "__main__":
    main()
