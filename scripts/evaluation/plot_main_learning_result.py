#!/usr/bin/env python3
"""Plot the main learning-result table as grouped success bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from kuafu.paths import resolve_artifact_path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# MPC values are the Raw MPC Mentor true-success rates from the main escalation table.
TASKS = [
    {
        "label": "T1\nStack\n2 cubes",
        "k": 8,
        "flat_il": 0.20,
        "bc": 10.79,
        "rrl": 86.89,
        "rrl_false_success": 0.00,
        "mpc_mentor_success": 96.15,
    },
    {
        "label": "T2\nOpen\ndrawer",
        "k": 6,
        "flat_il": 0.60,
        "bc": 25.11,
        "rrl": 98.80,
        "rrl_false_success": 0.00,
        "mpc_mentor_success": 100.00,
    },
    {
        "label": "T3\nOpen\nE-stop",
        "k": 6,
        "flat_il": 14.80,
        "bc": 29.80,
        "rrl": 83.92,
        "rrl_false_success": 6.34,
        "mpc_mentor_success": 89.29,
    },
    {
        "label": "T4\nRight\ncan",
        "k": 7,
        "flat_il": 4.90,
        "bc": 22.47,
        "rrl": 55.18,
        "rrl_false_success": 0.35,
        "mpc_mentor_success": 62.50,
    },
    {
        "label": "T5\nMarker\ncup",
        "k": 8,
        "flat_il": 0.60,
        "bc": 31.82,
        "rrl": 58.69,
        "rrl_false_success": 5.26,
        "mpc_mentor_success": 58.14,
    },
    {
        "label": "T6\nStack\n3 cubes",
        "k": 15,
        "flat_il": 0.00,
        "bc": 13.81,
        "rrl": 89.40,
        "rrl_false_success": 0.00,
        "mpc_mentor_success": 94.34,
    },
    {
        "label": "T7\nDrawer+\nstrawberry",
        "k": 14,
        "flat_il": 0.00,
        "bc": 53.02,
        "rrl": 76.10,
        "rrl_false_success": 0.00,
        "mpc_mentor_success": 70.92,
    },
    {
        "label": "T8\nHang\ncup",
        "k": 9,
        "flat_il": 0.00,
        "bc": 14.16,
        "rrl": 47.50,
        "rrl_false_success": 1.60,
        "mpc_mentor_success": 49.75,
    },
]

METHODS = [
    ("flat_il", "Flat IL", "#AEB4BC"),
    ("bc", "BC", "#4C78A8"),
    ("rrl", "RRL", "#54A24B"),
]

FALSE_SUCCESS_COLOR = "#D65F5F"
MPC_COLOR = "#2B2B2B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/main_learning_result",
        help="Directory for saved PDF/PNG outputs.",
    )
    parser.add_argument(
        "--stem",
        default="main_learning_success_bars",
        help="Filename stem for generated outputs.",
    )
    parser.add_argument(
        "--hide-flat-il",
        action="store_true",
        help="Only plot BC prior and Bounded RRL bars.",
    )
    parser.add_argument(
        "--hide-mpc",
        action="store_true",
        help="Do not plot hollow-diamond Raw MPC Mentor true-success markers.",
    )
    parser.add_argument(
        "--false-success-layout",
        choices=("overlay", "above", "none"),
        default="overlay",
        help="How to show RRL false-success rates in the main axes.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
        }
    )


def build_offsets(method_keys: list[str]) -> tuple[float, dict[str, float]]:
    width = 0.26 if len(method_keys) == 3 else 0.34
    center = (len(method_keys) - 1) / 2
    offsets = {key: (idx - center) * width for idx, key in enumerate(method_keys)}
    return width, offsets


def annotate_false_success_above(ax: plt.Axes, x: np.ndarray, rrl_offset: float) -> None:
    for idx, task in enumerate(TASKS):
        fs = task["rrl_false_success"]
        if fs <= 0:
            continue

        bar_top = task["rrl"]
        marker_y = min(bar_top + 5.0, 102.0)
        marker_x = x[idx] + rrl_offset
        ax.scatter(
            marker_x,
            marker_y,
            marker="x",
            s=28,
            color=FALSE_SUCCESS_COLOR,
            linewidths=1.1,
            zorder=5,
        )


def draw_false_success_overlay(ax: plt.Axes, x: np.ndarray, rrl_offset: float, bar_width: float) -> None:
    fs_values = np.array([task["rrl_false_success"] for task in TASKS], dtype=float)
    mask = fs_values > 0.0
    if not np.any(mask):
        return

    marker_x = x[mask] + rrl_offset
    marker_y = fs_values[mask]
    ax.vlines(marker_x, 0.0, marker_y, color=FALSE_SUCCESS_COLOR, lw=0.95, zorder=5)
    ax.scatter(
        marker_x,
        marker_y,
        marker="x",
        s=24,
        color=FALSE_SUCCESS_COLOR,
        linewidths=1.05,
        zorder=6,
    )


def plot_main_learning_result(args: argparse.Namespace) -> tuple[Path, Path]:
    configure_matplotlib()

    method_items = METHODS[1:] if args.hide_flat_il else METHODS
    method_keys = [key for key, _, _ in method_items]
    width, offsets = build_offsets(method_keys)
    bar_width = width * 0.94

    x = np.arange(len(TASKS), dtype=float)
    labels = [f"T{idx + 1}" for idx in range(len(TASKS))]

    fig, ax = plt.subplots(figsize=(5.75, 2.05), layout="constrained")

    for key, name, color in method_items:
        values = np.array([task[key] for task in TASKS], dtype=float)
        ax.bar(
            x + offsets[key],
            values,
            width=bar_width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.97,
            zorder=3,
        )

    if args.false_success_layout == "overlay":
        draw_false_success_overlay(ax, x, offsets["rrl"], bar_width)

    if not args.hide_mpc:
        mpc_points = [
            (idx, task["mpc_mentor_success"])
            for idx, task in enumerate(TASKS)
            if task["mpc_mentor_success"] is not None
        ]
        if mpc_points:
            mpc_x = np.array([x[idx] for idx, _ in mpc_points], dtype=float)
            mpc_y = np.array([value for _, value in mpc_points], dtype=float)
            ax.scatter(
                mpc_x,
                mpc_y,
                marker="D",
                s=31,
                facecolors="none",
                edgecolors=MPC_COLOR,
                linewidths=1.0,
                zorder=7,
                label="MPC",
            )

    if args.false_success_layout == "above":
        annotate_false_success_above(ax, x, offsets["rrl"])

    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#E9E9E9", lw=0.55, zorder=0)
    ax.tick_params(axis="both", length=2.3, width=0.65, pad=2, color="#666666")
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.set_axisbelow(True)
    ax.margins(x=0.012)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    legend_handles = [Patch(facecolor=color, edgecolor="white", label=name) for _, name, color in method_items]
    if not args.hide_mpc and any(task["mpc_mentor_success"] is not None for task in TASKS):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=MPC_COLOR,
                markeredgewidth=1.0,
                markersize=5.0,
                label="MPC",
            )
        )
    if args.false_success_layout == "overlay":
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="-",
                color=FALSE_SUCCESS_COLOR,
                linewidth=1.0,
                markersize=5.0,
                label="RRL FS",
            )
        )
    if args.false_success_layout == "above":
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                color=FALSE_SUCCESS_COLOR,
                markersize=4.8,
                label="RRL FS",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0.0,
    )

    output_dir = resolve_artifact_path(args.output_dir, "Main-learning figure directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{args.stem}.pdf"
    png_path = output_dir / f"{args.stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    pdf_path, png_path = plot_main_learning_result(args)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
