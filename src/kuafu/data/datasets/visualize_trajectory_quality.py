"""
Trajectory quality analysis across all HDF5 datasets.

Plots per dataset:
  1. Trajectory length distribution + CDF
  2. Per-step reward profile: mean ± std over all demos (aligned to start)
  3. Total / final reward distribution
  4. Action smoothness (mean |Δa| per trajectory)
  5. Anomaly flags: short trajs, reward spikes, missing data

Also produces a single summary table figure across all datasets.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from kuafu.paths import external_artifact_root

BASE_DIR = external_artifact_root() / "Trajectories"

DATASETS = {
    "table_drawer":       "fr3_finray_table_drawer_damped_noise.hdf5",
    "pen_cup":            "fr3_finray_pen_cup_damped_noise.hdf5",
    "table_drawer_stack": "fr3_finray_table_drawer_stack_damped_noise.hdf5",
    "table_right_can":    "fr3_finray_table_right_can_spice_damped_noise.hdf5",
    "table_stack2":       "fr3_finray_table_stack2_damped_higher_noise.hdf5",
    "table_stack3":       "fr3_finray_table_stack3_damped_higher_noise.hdf5",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ─── data loading ─────────────────────────────────────────────────────────────

def load_quality_stats(path):
    stats = dict(lengths=[], rewards_total=[], rewards_final=[],
                 rewards_max=[], action_smoothness=[], done_step=[],
                 success=[], reward_curves=[], missing_rewards=0,
                 corrupt=[])
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for d in demos:
            demo = f[f"data/{d}"]
            if "rewards" not in demo:
                stats["missing_rewards"] += 1
                stats["corrupt"].append(d)
                continue
            T = int(demo.attrs["num_samples"])
            r = demo["rewards"][:]
            act = demo["actions/relative_joint"][:]
            dones = demo["dones"][:]

            stats["lengths"].append(T)
            stats["rewards_total"].append(float(r.sum()))
            stats["rewards_final"].append(float(r[-1]))
            stats["rewards_max"].append(float(r.max()))
            stats["success"].append(bool(demo.attrs.get("success", False)))

            done_idx = np.where(dones)[0]
            stats["done_step"].append(int(done_idx[0]) if len(done_idx) > 0 else T - 1)

            if len(act) > 1:
                delta = np.linalg.norm(np.diff(act, axis=0), axis=1)
                stats["action_smoothness"].append(float(delta.mean()))
            else:
                stats["action_smoothness"].append(0.0)

            stats["reward_curves"].append(r)

    for k in ["lengths", "rewards_total", "rewards_final", "rewards_max",
              "action_smoothness", "done_step"]:
        stats[k] = np.array(stats[k])
    stats["success"] = np.array(stats["success"])
    return stats


def reward_profile(curves, percentile=80):
    """Align reward curves at start, compute mean ± std up to percentile-th length."""
    max_T = int(np.percentile([len(c) for c in curves], percentile))
    mat = np.full((len(curves), max_T), np.nan)
    for i, c in enumerate(curves):
        n = min(len(c), max_T)
        mat[i, :n] = c[:n]
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return np.arange(max_T), mean, std


# ─── per-dataset plot ─────────────────────────────────────────────────────────

def plot_dataset_quality(name, filepath, color):
    stats = load_quality_stats(filepath)
    L = stats["lengths"]
    R = stats["rewards_total"]
    Rf = stats["rewards_final"]
    AN = stats["action_smoothness"]
    n = len(L)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Trajectory Quality — {name}   N={n}   success={stats['success'].mean()*100:.1f}%"
        + (f"   missing={stats['missing_rewards']}" if stats["missing_rewards"] else ""),
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # ── (0,0-1): trajectory length histogram + CDF ───────────────────────────
    ax_len = fig.add_subplot(gs[0, :2])
    ax_len.hist(L, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.3, label="count")
    ax_len.set_xlabel("Trajectory length (steps)")
    ax_len.set_ylabel("Count")
    ax_len.set_title("Trajectory Length Distribution")
    ax_len.axvline(L.mean(), color="red", lw=1.2, linestyle="--", label=f"mean={L.mean():.0f}")
    ax_len.axvline(np.median(L), color="orange", lw=1.2, linestyle=":", label=f"median={np.median(L):.0f}")
    ax_len.legend(fontsize=8)
    ax_len.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    ax_cdf = ax_len.twinx()
    sorted_L = np.sort(L)
    cdf = np.arange(1, len(sorted_L) + 1) / len(sorted_L)
    ax_cdf.plot(sorted_L, cdf, color="darkblue", lw=1.2, alpha=0.6)
    ax_cdf.set_ylabel("CDF", color="darkblue")
    ax_cdf.set_ylim(0, 1)

    # ── (0,2-3): total reward distribution ───────────────────────────────────
    ax_rwd = fig.add_subplot(gs[0, 2:])
    ax_rwd.hist(R, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax_rwd.set_xlabel("Total reward per trajectory")
    ax_rwd.set_ylabel("Count")
    ax_rwd.set_title("Total Reward Distribution")
    ax_rwd.axvline(R.mean(), color="red", lw=1.2, linestyle="--", label=f"mean={R.mean():.0f}")
    ax_rwd.legend(fontsize=8)
    ax_rwd.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    # ── (1, 0-3): per-step reward profile ────────────────────────────────────
    ax_prof = fig.add_subplot(gs[1, :])
    t, mean_r, std_r = reward_profile(stats["reward_curves"], percentile=90)

    # clip first step spike for clarity
    plot_mean = mean_r.copy()
    ax_prof.plot(t, plot_mean, color=color, lw=1.2, label="mean reward")
    ax_prof.fill_between(t, plot_mean - std_r, plot_mean + std_r, alpha=0.25, color=color, label="±1 std")

    # mark where most trajs end
    ax_prof.axvline(np.median(stats["done_step"]), color="green", lw=1, linestyle="--",
                    label=f"median done={np.median(stats['done_step']):.0f}")
    ax_prof.set_xlabel("Step")
    ax_prof.set_ylabel("Reward")
    ax_prof.set_title("Per-Step Reward Profile (mean ± std across demos, first-step spike clipped to plot range)")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(True, linewidth=0.4, alpha=0.5)

    # auto-clip y for readability (ignore step-0 spike)
    if len(mean_r) > 1:
        body_max = np.nanpercentile(mean_r[1:] + std_r[1:], 99) * 1.3
        ax_prof.set_ylim(bottom=-0.05 * body_max, top=max(body_max, 0.1))

    # ── (2,0-1): final reward distribution ───────────────────────────────────
    ax_rfin = fig.add_subplot(gs[2, :2])
    ax_rfin.hist(Rf, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax_rfin.set_xlabel("Final step reward")
    ax_rfin.set_ylabel("Count")
    ax_rfin.set_title("Final Reward Distribution")
    ax_rfin.axvline(Rf.mean(), color="red", lw=1.2, linestyle="--", label=f"mean={Rf.mean():.3f}")
    frac_zero = (Rf == 0).mean()
    ax_rfin.text(0.97, 0.92, f"final=0: {frac_zero*100:.1f}%", transform=ax_rfin.transAxes,
                 ha="right", fontsize=8, color="darkred")
    ax_rfin.legend(fontsize=8)
    ax_rfin.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    # ── (2,2-3): action smoothness distribution ───────────────────────────────
    ax_sm = fig.add_subplot(gs[2, 2:])
    ax_sm.hist(AN, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax_sm.set_xlabel("Mean |Δaction| per trajectory (rad)")
    ax_sm.set_ylabel("Count")
    ax_sm.set_title("Action Smoothness (lower = smoother)")
    ax_sm.axvline(AN.mean(), color="red", lw=1.2, linestyle="--", label=f"mean={AN.mean():.4f}")
    # flag jerk outliers
    thresh = AN.mean() + 3 * AN.std()
    n_jerky = (AN > thresh).sum()
    ax_sm.axvline(thresh, color="orange", lw=1, linestyle=":", label=f"3σ ({n_jerky} demos)")
    ax_sm.legend(fontsize=8)
    ax_sm.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    out_path = BASE_DIR / f"quality_{name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return stats


# ─── summary table ────────────────────────────────────────────────────────────

def plot_summary_table(all_stats):
    metrics = ["N", "success%", "len mean", "len std", "len min", "len max",
               "short(<10)", "missing", "total_r mean", "final_r mean",
               "final_r=0%", "smooth mean", "jerk outliers"]
    rows = []
    row_labels = []
    for name, stats in all_stats.items():
        L = stats["lengths"]
        R = stats["rewards_total"]
        Rf = stats["rewards_final"]
        AN = stats["action_smoothness"]
        thresh = AN.mean() + 3 * AN.std()
        row_labels.append(name)
        rows.append([
            len(L),
            f"{stats['success'].mean()*100:.1f}%",
            f"{L.mean():.1f}",
            f"{L.std():.1f}",
            int(L.min()),
            int(L.max()),
            int((L < 10).sum()),
            stats["missing_rewards"],
            f"{R.mean():.0f}",
            f"{Rf.mean():.3f}",
            f"{(Rf==0).mean()*100:.1f}%",
            f"{AN.mean():.4f}",
            int((AN > thresh).sum()),
        ])

    fig, ax = plt.subplots(figsize=(20, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=metrics,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # color header
    for j in range(len(metrics)):
        tbl[(0, j)].set_facecolor("#4C72B0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # color row labels
    for i in range(len(row_labels)):
        tbl[(i + 1, -1)].set_facecolor("#f0f0f0")
        tbl[(i + 1, -1)].set_text_props(fontweight="bold")
    # highlight anomalies: short trajs, missing, jerk outliers
    short_col = metrics.index("short(<10)")
    missing_col = metrics.index("missing")
    jerk_col = metrics.index("jerk outliers")
    for i, row in enumerate(rows):
        if int(row[short_col]) > 0:
            tbl[(i + 1, short_col)].set_facecolor("#ffcccc")
        if int(row[missing_col]) > 0:
            tbl[(i + 1, missing_col)].set_facecolor("#ffcccc")
        if int(row[jerk_col]) > 0:
            tbl[(i + 1, jerk_col)].set_facecolor("#fff3cc")

    fig.suptitle("Trajectory Quality Summary — All Datasets", fontsize=13, fontweight="bold", y=1.01)
    out_path = BASE_DIR / "quality_summary.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── anomaly report ──────────────────────────────────────────────────────────

def print_anomaly_report(all_stats):
    print("\n" + "=" * 60)
    print("ANOMALY REPORT")
    print("=" * 60)
    for name, stats in all_stats.items():
        L = stats["lengths"]
        AN = stats["action_smoothness"]
        Rf = stats["rewards_final"]
        thresh = AN.mean() + 3 * AN.std()
        issues = []
        if (L < 10).sum():
            issues.append(f"{(L<10).sum()} ultra-short trajs (len<10)")
        if stats["missing_rewards"]:
            issues.append(f"{stats['missing_rewards']} demos missing rewards: {stats['corrupt']}")
        if (AN > thresh).sum():
            issues.append(f"{(AN>thresh).sum()} jerk outliers (|Δa|>{thresh:.3f})")
        if (Rf == 0).mean() > 0.3:
            issues.append(f"{(Rf==0).mean()*100:.1f}% demos end with reward=0")
        if issues:
            print(f"\n{name}:")
            for iss in issues:
                print(f"  ⚠  {iss}")
        else:
            print(f"\n{name}: OK")
    print()


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_stats = {}
    for (name, fn), color in zip(DATASETS.items(), COLORS):
        print(f"Processing {name}...")
        filepath = BASE_DIR / fn
        stats = plot_dataset_quality(name, filepath, color)
        all_stats[name] = stats

    plot_summary_table(all_stats)
    print_anomaly_report(all_stats)
    print("Done.")
