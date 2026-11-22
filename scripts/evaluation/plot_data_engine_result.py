#!/usr/bin/env python3
"""Plot autonomous data generation speed, quality, and trajectory metrics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from kuafu.paths import resolve_artifact_path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    short_name: str
    success: float
    quality_pass: float
    norm_time: float
    norm_eef_path: float
    norm_action_path: float
    action_second_diff: float


@dataclass(frozen=True)
class QualityPoint:
    name: str
    short_name: str
    success: float
    quality_pass: float


@dataclass(frozen=True)
class SpeedRow:
    label: str
    raw_mpc_sec: float
    kuafu_sec: float
    sim2real_ratio: float

    @property
    def speedup(self) -> float:
        return self.raw_mpc_sec / self.kuafu_sec


METHODS = [
    MethodMetrics("MimicGen", "MG", 29.3, 17.6, 1.10, 11.79, 1.49, 0.059),
    MethodMetrics("SkillGen", "SG", 30.3, 16.8, 1.50, 26.05, 2.99, 0.063),
    MethodMetrics("Isaac Mimic", "IM", 46.7, 30.5, 0.95, 1.10, 1.25, 0.063),
    MethodMetrics("Isaac SkillGen", "IS", 61.4, 42.7, 0.80, 0.90, 0.92, 0.060),
    MethodMetrics("Kuafu", "K", 72.0, 71.5, 0.61, 1.03, 0.54, 0.028),
]

QUALITY_POINTS = [
    QualityPoint(method.name, method.short_name, method.success, method.quality_pass)
    for method in METHODS
]

SPEED_ROWS = [
    SpeedRow("T1 Stack 2", 34.49, 2.49, 0.609),
    SpeedRow("T2 Drawer", 34.93, 2.15, 0.456),
    SpeedRow("T3 E-stop", 47.70, 2.65, 0.520),
    SpeedRow("T4 Can", 133.04, 6.89, 0.521),
    SpeedRow("T5 Marker", 162.62, 7.70, 0.504),
    SpeedRow("T6 Stack 3", 81.37, 3.87, 0.484),
    SpeedRow("T7 Drawer+berry", 137.43, 6.59, 0.529),
    SpeedRow("T8 Hang cup", 218.67, 12.33, 1.215),
]

METRIC_KEYS = [
    "norm_time",
    "norm_eef_path",
    "norm_action_path",
    "action_second_diff",
]
METRIC_LABELS = [
    "Time $\\downarrow$",
    "EEF path $\\downarrow$",
    "Act. path $\\downarrow$",
    "$\\Delta^2 a$ $\\downarrow$",
]
HIGHER_IS_BETTER = {
    "success": True,
    "quality_pass": True,
    "norm_time": False,
    "norm_eef_path": False,
    "norm_action_path": False,
    "action_second_diff": False,
}


MPC_COLOR = "#8D939B"
KUAFU_COLOR = "#1F8F64"
ACCENT_COLOR = "#C44E52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/data_engine_result",
        help="Directory for saved PDF/PNG outputs.",
    )
    parser.add_argument(
        "--stem",
        default="data_engine_autonomous_generation",
        help="Filename stem for the combined figure.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "legend.fontsize": 13,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def metric_matrix() -> np.ndarray:
    return np.array([[getattr(method, key) for key in METRIC_KEYS] for method in METHODS], dtype=float)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(values, dtype=float)
    for col_idx, key in enumerate(METRIC_KEYS):
        col = values[:, col_idx]
        order = np.argsort(col)
        if HIGHER_IS_BETTER[key]:
            order = order[::-1]
        for rank, row_idx in enumerate(order):
            ranks[row_idx, col_idx] = 1.0 - rank / (len(METHODS) - 1)
    return ranks


def format_metric(value: float, key: str) -> str:
    if key in {"success", "quality_pass"}:
        return f"{value:.1f}"
    if key == "action_second_diff":
        return f"{value:.3f}"
    return f"{value:.2f}"


def draw_realtime_speed_panel(ax: plt.Axes, title: str = "A. Sim-to-Real Speed") -> None:
    y = np.arange(len(SPEED_ROWS), dtype=float)
    ratios = np.array([row.sim2real_ratio for row in SPEED_ROWS], dtype=float)
    factors = 1.0 / ratios

    colors = [KUAFU_COLOR if factor >= 1.0 else ACCENT_COLOR for factor in factors]
    ax.barh(y, factors, height=0.64, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(1.0, color="#666666", lw=0.9, ls=(0, (2, 2)), zorder=2)

    for idx, factor in enumerate(factors):
        ax.text(
            max(factor - 0.06, 0.12),
            idx,
            f"{factor:.1f}x",
            ha="right",
            va="center",
            fontsize=15.8,
            color="white",
            fontweight=900,
            path_effects=[pe.withStroke(linewidth=1.0, foreground="#1F1F1F")],
        )

    ax.set_xlim(0.0, 2.25)
    ax.set_ylim(-0.72, len(SPEED_ROWS) - 0.38)
    ax.invert_yaxis()
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels(["0", "1x", "2x"])
    ax.set_yticks(y)
    ax.set_yticklabels([row.label.split()[0] for row in SPEED_ROWS])
    ax.set_xlabel("Speed / real time")
    ax.grid(axis="x", which="major", color="#E5E5E5", lw=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="center", x=0.5, ha="center", fontweight="bold", pad=4)


def draw_heatmap(ax: plt.Axes, show_ylabel: bool = True, title: str = "C. Remaining Metrics") -> None:
    values = metric_matrix()
    ranks = rank_normalize(values)
    cmap = LinearSegmentedColormap.from_list(
        "paper_rank",
        ["#F7F7F2", "#D9E8D2", "#86BEA6", "#287D8E", "#123C69"],
    )

    ax.imshow(ranks, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([method.name for method in METHODS])
    if show_ylabel:
        ax.set_ylabel("Pipeline")

    for row_idx, method in enumerate(METHODS):
        for col_idx, key in enumerate(METRIC_KEYS):
            value = getattr(method, key)
            color = "white" if ranks[row_idx, col_idx] > 0.62 else "#161616"
            is_best = ranks[row_idx, col_idx] == 1.0
            effects = [pe.withStroke(linewidth=1.35, foreground="#071923")] if is_best else []
            ax.text(
                col_idx,
                row_idx,
                format_metric(value, key),
                ha="center",
                va="center",
                color=color,
                fontsize=18.0 if is_best else 16.2,
                fontweight=900 if is_best else 400,
                path_effects=effects,
            )

    ax.set_xticks(np.arange(-0.5, len(METRIC_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=True, length=0, pad=5)
    ax.tick_params(axis="y", left=False, right=False, length=0, pad=4)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14.2)
        tick.set_fontweight("semibold")
        tick.set_color("#222222")
    for tick, method in zip(ax.get_yticklabels(), METHODS):
        tick.set_color("#222222")
        if method.name == "Kuafu":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="center", x=0.5, ha="center", fontweight="bold", pad=4)


def draw_pareto(ax: plt.Axes, title: str = "B. Success -> Quality Pass") -> None:
    x_success = 0.0
    x_quality = 1.0
    colors = {
        "MimicGen": "#4E79A7",
        "SkillGen": "#E07B39",
        "Isaac Mimic": "#7A8C87",
        "Isaac SkillGen": "#2F8F83",
        "Kuafu": KUAFU_COLOR,
    }
    linestyles = {
        "MimicGen": (0, (2.5, 1.8)),
        "SkillGen": (0, (1.0, 1.4)),
        "Isaac Mimic": "solid",
        "Isaac SkillGen": "solid",
        "Kuafu": "solid",
    }

    for point in QUALITY_POINTS:
        is_kuafu = point.name == "Kuafu"
        color = colors[point.name]
        ax.plot(
            [x_success, x_quality],
            [point.success, point.quality_pass],
            color=color,
            lw=3.0 if is_kuafu else 1.9,
            linestyle=linestyles[point.name],
            alpha=0.96,
            solid_capstyle="round",
            zorder=3 if is_kuafu else 2,
        )
        ax.scatter(
            [x_success, x_quality],
            [point.success, point.quality_pass],
            s=170 if is_kuafu else 115,
            color=color,
            edgecolors="#1E1E1E",
            linewidths=0.75,
            zorder=5 if is_kuafu else 4,
        )

    kuafu = next(point for point in QUALITY_POINTS if point.name == "Kuafu")
    ax.scatter(
        [x_success, x_quality],
        [kuafu.success, kuafu.quality_pass],
        s=275,
        facecolors="none",
        edgecolors=ACCENT_COLOR,
        linewidths=1.55,
        zorder=6,
    )

    right_labels = {
        "MimicGen": (10, 12),
        "SkillGen": (10, -12),
        "Isaac Mimic": (10, 0),
        "Isaac SkillGen": (10, 0),
        "Kuafu": (10, 0),
    }
    for point in QUALITY_POINTS:
        dx, dy = right_labels[point.name]
        is_kuafu = point.name == "Kuafu"
        ax.annotate(
            f"{point.short_name} {point.quality_pass:.1f}",
            xy=(x_quality, point.quality_pass),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=13.2,
            fontweight="bold" if is_kuafu else "normal",
            color=colors[point.name],
            arrowprops={
                "arrowstyle": "-",
                "color": colors[point.name],
                "lw": 0.75,
                "shrinkA": 1,
                "shrinkB": 5,
            } if point.name in {"MimicGen", "SkillGen"} else None,
            path_effects=[pe.withStroke(linewidth=1.0, foreground="white")],
            zorder=7,
        )

    left_values = {
        "MimicGen": (-9, -9),
        "SkillGen": (-9, 9),
        "Isaac Mimic": (-9, 0),
        "Isaac SkillGen": (-9, 0),
        "Kuafu": (-9, 0),
    }
    for point in QUALITY_POINTS:
        dx, dy = left_values[point.name]
        ax.annotate(
            f"{point.success:.1f}",
            xy=(x_success, point.success),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=11.6,
            fontweight="bold" if point.name == "Kuafu" else "normal",
            color=colors[point.name],
            path_effects=[pe.withStroke(linewidth=1.0, foreground="white")],
            zorder=7,
        )

    ax.set_xlim(-0.24, 1.38)
    ax.set_ylim(10, 78)
    ax.set_xticks([x_success, x_quality])
    ax.set_xticklabels(["Success", "Quality pass"])
    ax.set_ylabel("Rate (%)")
    ax.set_yticks(np.arange(20, 81, 10))
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7, zorder=0)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=3)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="center", x=0.5, ha="center", fontweight="bold", pad=4)


def draw_success_quality_grouped_bars(ax: plt.Axes, title: str = "B. Success and Quality Pass") -> None:
    y = np.arange(len(QUALITY_POINTS), dtype=float)
    success = np.array([point.success for point in QUALITY_POINTS], dtype=float)
    quality = np.array([point.quality_pass for point in QUALITY_POINTS], dtype=float)
    success_colors = [KUAFU_COLOR if point.name == "Kuafu" else "#86BEA6" for point in QUALITY_POINTS]
    quality_colors = ["#2F6F63" if point.name == "Kuafu" else "#D9E8D2" for point in QUALITY_POINTS]

    ax.barh(y - 0.16, success, height=0.28, color=success_colors, edgecolor="white", linewidth=0.7, label="Success", zorder=3)
    ax.barh(y + 0.16, quality, height=0.28, color=quality_colors, edgecolor="white", linewidth=0.7, label="Quality pass", zorder=3)

    for idx, point in enumerate(QUALITY_POINTS):
        weight = "bold" if point.name == "Kuafu" else "normal"
        ax.text(success[idx] + 0.9, y[idx] - 0.16, f"{success[idx]:.1f}", va="center", ha="left", fontsize=9.8, fontweight=weight, color="#222222")
        ax.text(quality[idx] + 0.9, y[idx] + 0.16, f"{quality[idx]:.1f}", va="center", ha="left", fontsize=9.8, fontweight=weight, color="#222222")

    ax.set_xlim(0, 80)
    ax.set_xticks(np.arange(0, 81, 20))
    ax.set_yticks(y)
    ax.set_yticklabels([point.short_name for point in QUALITY_POINTS])
    ax.set_xlabel("Rate (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=3)
    for tick, point in zip(ax.get_yticklabels(), QUALITY_POINTS):
        if point.name == "Kuafu":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.35, labelspacing=0.2, borderpad=0.1, fontsize=11.5)
    ax.set_title(title, loc="center", x=0.5, ha="center", fontweight="bold", pad=4)


def draw_success_quality_overlay_bars(ax: plt.Axes, title: str = "B. Quality Within Success") -> None:
    y = np.arange(len(QUALITY_POINTS), dtype=float)
    success = np.array([point.success for point in QUALITY_POINTS], dtype=float)
    quality = np.array([point.quality_pass for point in QUALITY_POINTS], dtype=float)

    ax.barh(y, success, height=0.56, color="#D9E8D2", edgecolor="white", linewidth=0.7, label="Success", zorder=2)
    ax.barh(y, quality, height=0.32, color="#86BEA6", edgecolor="white", linewidth=0.7, label="Quality pass", zorder=3)
    kuafu_idx = next(idx for idx, point in enumerate(QUALITY_POINTS) if point.name == "Kuafu")
    ax.barh(y[kuafu_idx], success[kuafu_idx], height=0.56, color="#A8D8B9", edgecolor="white", linewidth=0.7, zorder=2)
    ax.barh(y[kuafu_idx], quality[kuafu_idx], height=0.32, color=KUAFU_COLOR, edgecolor="white", linewidth=0.7, zorder=3)

    for idx, point in enumerate(QUALITY_POINTS):
        weight = "bold" if point.name == "Kuafu" else "normal"
        ax.text(
            min(success[idx] + 1.2, 77.0),
            y[idx],
            f"{quality[idx]:.1f}/{success[idx]:.1f}",
            va="center",
            ha="left",
            fontsize=10.2,
            fontweight=weight,
            color="#222222",
        )

    ax.set_xlim(0, 80)
    ax.set_xticks(np.arange(0, 81, 20))
    ax.set_yticks(y)
    ax.set_yticklabels([point.short_name for point in QUALITY_POINTS])
    ax.set_xlabel("Rate (%)")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, zorder=0)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=3)
    for tick, point in zip(ax.get_yticklabels(), QUALITY_POINTS):
        if point.name == "Kuafu":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.35, labelspacing=0.2, borderpad=0.1, fontsize=11.5)
    ax.set_title(title, loc="center", x=0.5, ha="center", fontweight="bold", pad=4)


def draw_success_quality_heatmap(ax: plt.Axes, title: str = "B. Success / Quality Pass") -> None:
    values = np.array([[point.success, point.quality_pass] for point in QUALITY_POINTS], dtype=float)
    normalized = values / 100.0
    cmap = LinearSegmentedColormap.from_list("success_quality", ["#F7F7F2", "#D9E8D2", "#86BEA6", KUAFU_COLOR])

    ax.imshow(normalized, cmap=cmap, vmin=0.0, vmax=0.75, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Success", "Quality pass"])
    ax.set_yticks(np.arange(len(QUALITY_POINTS)))
    ax.set_yticklabels([point.short_name for point in QUALITY_POINTS])

    for row_idx, point in enumerate(QUALITY_POINTS):
        for col_idx, value in enumerate(values[row_idx]):
            is_kuafu = point.name == "Kuafu"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=14.8,
                fontweight=900 if is_kuafu else 500,
                color="white" if normalized[row_idx, col_idx] > 0.55 else "#111111",
                path_effects=[pe.withStroke(linewidth=1.1, foreground="#14322B")] if is_kuafu else [],
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(QUALITY_POINTS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", top=False, bottom=False, labeltop=False, labelbottom=True, length=0, pad=5)
    ax.tick_params(axis="y", left=False, right=False, length=0, pad=4)
    for tick, point in zip(ax.get_yticklabels(), QUALITY_POINTS):
        if point.name == "Kuafu":
            tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(13.2)
        tick.set_fontweight("semibold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="center", x=0.5, ha="center", fontweight="bold", pad=4)


PANEL_B_VARIANTS = [
    ("b2_grouped_bars", draw_success_quality_grouped_bars),
    ("b3_overlay_bars", draw_success_quality_overlay_bars),
    ("b4_success_quality_heatmap", draw_success_quality_heatmap),
]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def plot_combined(output_dir: Path, stem: str, dpi: int, panel_b_func=draw_pareto) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(16.6, 3.45), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.006, h_pad=0.004, wspace=0.014, hspace=0.012)
    grid = fig.add_gridspec(nrows=1, ncols=5, width_ratios=[0.9, 0.20, 1.55, 0.02, 2.05], wspace=0.0)
    draw_realtime_speed_panel(fig.add_subplot(grid[0, 0]))
    panel_b_func(fig.add_subplot(grid[0, 2]))
    draw_heatmap(fig.add_subplot(grid[0, 4]), show_ylabel=False)
    return save_figure(fig, output_dir, stem, dpi)


def plot_speed(output_dir: Path, dpi: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(4.05, 3.35), layout="constrained")
    draw_realtime_speed_panel(ax, title="Sim-to-Real Speed")
    return save_figure(fig, output_dir, "data_engine_sim2real_speed", dpi)


def plot_heatmap(output_dir: Path, dpi: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.15, 2.65), layout="constrained")
    draw_heatmap(ax, title="Remaining Metrics")
    return save_figure(fig, output_dir, "data_engine_quality_heatmap", dpi)


def plot_pareto(output_dir: Path, dpi: int) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(3.8, 3.1), layout="constrained")
    draw_pareto(ax, title="Success -> Quality Pass")
    return save_figure(fig, output_dir, "data_engine_success_quality_slope", dpi)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = resolve_artifact_path(args.output_dir, "Data-engine figure directory")
    written = [plot_combined(output_dir, args.stem, args.dpi)]
    written.extend(
        plot_combined(output_dir, f"{args.stem}_{suffix}", args.dpi, panel_b_func)
        for suffix, panel_b_func in PANEL_B_VARIANTS
    )
    written.extend(
        [
            plot_speed(output_dir, args.dpi),
            plot_heatmap(output_dir, args.dpi),
            plot_pareto(output_dir, args.dpi),
        ]
    )
    for pdf_path, png_path in written:
        print(f"Wrote {pdf_path}")
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
