"""
Visualize the distribution of initial conditions across all trajectories
in each HDF5 dataset. For each dataset, we plot:
  - Task object XY positions (scatter + marginal histograms)
  - Object Z positions (histogram)
  - Object yaw angle extracted from quaternion (histogram)
  - Cabinet joint position, if present
  - Robot arm joint positions (first 7 DOF)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from kuafu.paths import external_artifact_root
import os


# ─── helpers ──────────────────────────────────────────────────────────────────

def quat_to_yaw(q):
    """quaternion [qw, qx, qy, qz] → yaw (rad)"""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))


def load_initial_conditions(path):
    """Return a dict {field_path: np.ndarray (N, D)} for every dataset under initial_state."""
    data = {}
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        # collect field names from first demo
        sample_demo = f[f"data/{demos[0]}"]
        fields = []
        sample_demo["initial_state"].visititems(
            lambda name, obj: fields.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        for field in fields:
            arrays = []
            for demo_key in demos:
                arr = f[f"data/{demo_key}/initial_state/{field}"][:]  # (1, D)
                arrays.append(arr[0])  # remove env dim → (D,)
            data[field] = np.stack(arrays, axis=0)  # (N, D)
    return data, len(demos)


# ─── dataset configs ──────────────────────────────────────────────────────────

DATASETS = {
    "table_drawer": {
        "file": "fr3_finray_table_drawer_damped_noise.hdf5",
        "task_objects": ["articulation/cabinet"],
        "has_cabinet": True,
    },
    "pen_cup": {
        "file": "fr3_finray_pen_cup_damped_noise.hdf5",
        "task_objects": ["rigid_object/marker", "rigid_object/pen_holder"],
        "has_cabinet": False,
    },
    "table_drawer_stack": {
        "file": "fr3_finray_table_drawer_stack_damped_noise.hdf5",
        "task_objects": ["articulation/cabinet", "rigid_object/strawberry"],
        "has_cabinet": True,
    },
    "table_right_can": {
        "file": "fr3_finray_table_right_can_spice_damped_noise.hdf5",
        "task_objects": ["rigid_object/can"],
        "has_cabinet": False,
    },
    "table_stack2": {
        "file": "fr3_finray_table_stack2_damped_higher_noise.hdf5",
        "task_objects": ["rigid_object/cube_1", "rigid_object/cube_2"],
        "has_cabinet": False,
    },
    "table_stack3": {
        "file": "fr3_finray_table_stack3_damped_higher_noise.hdf5",
        "task_objects": ["rigid_object/cube_1", "rigid_object/cube_2", "rigid_object/cube_3"],
        "has_cabinet": False,
    },
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
BASE_DIR = external_artifact_root() / "Trajectories"


# ─── per-dataset plot ─────────────────────────────────────────────────────────

def plot_dataset(name, cfg):
    filepath = BASE_DIR / cfg["file"]
    ic, n_demos = load_initial_conditions(filepath)

    task_objs = cfg["task_objects"]
    has_cabinet = cfg["has_cabinet"]
    n_obj = len(task_objs)

    # robot arm joints (first 7 of 15)
    robot_joints = ic["articulation/robot/joint_position"][:, :7]  # (N, 7)
    joint_labels = [f"J{i+1}" for i in range(7)]

    # figure layout
    # row 0: XY scatter (one per task object)
    # row 1: Z + yaw histograms (one per task object, 2 plots each)
    # row 2 (optional): cabinet joint position
    # row 3: robot joint positions

    n_rows = 3 + int(has_cabinet)
    fig = plt.figure(figsize=(max(14, 4 * n_obj + 2), 4 * n_rows))
    fig.suptitle(f"Initial Condition Distribution — {name}  (N={n_demos})", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(n_rows, max(n_obj * 2, 7), figure=fig, hspace=0.45, wspace=0.4)

    # ── row 0: XY scatter ────────────────────────────────────────────────────
    for j, obj_key in enumerate(task_objs):
        pose_key = obj_key + "/root_pose"
        if pose_key not in ic:
            continue
        pose = ic[pose_key]          # (N, 7)  [x, y, z, qw, qx, qy, qz]
        x, y = pose[:, 0], pose[:, 1]
        color = COLORS[j % len(COLORS)]
        obj_name = obj_key.split("/")[-1]

        span = max(2, 2 * n_obj)
        ax = fig.add_subplot(gs[0, j * 2: j * 2 + 2])
        ax.scatter(x, y, s=6, alpha=0.35, color=color, rasterized=True)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{obj_name}  XY position")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── row 1: Z + yaw per object ────────────────────────────────────────────
    for j, obj_key in enumerate(task_objs):
        pose_key = obj_key + "/root_pose"
        if pose_key not in ic:
            continue
        pose = ic[pose_key]
        z = pose[:, 2]
        yaw = quat_to_yaw(pose[:, 3:])   # columns 3-6 are [qw, qx, qy, qz]
        yaw_deg = np.degrees(yaw)
        color = COLORS[j % len(COLORS)]
        obj_name = obj_key.split("/")[-1]

        ax_z = fig.add_subplot(gs[1, j * 2])
        ax_z.hist(z, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax_z.set_xlabel("Z (m)")
        ax_z.set_ylabel("count")
        ax_z.set_title(f"{obj_name}  Z height")
        ax_z.grid(True, axis="y", linewidth=0.4, alpha=0.5)

        ax_yaw = fig.add_subplot(gs[1, j * 2 + 1])
        ax_yaw.hist(yaw_deg, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax_yaw.set_xlabel("Yaw (deg)")
        ax_yaw.set_ylabel("count")
        ax_yaw.set_title(f"{obj_name}  Yaw")
        ax_yaw.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    # ── row 2 (optional): cabinet joint ─────────────────────────────────────
    row = 2
    if has_cabinet:
        cab_jpos = ic.get("articulation/cabinet/joint_position")
        if cab_jpos is not None:
            ax_cab = fig.add_subplot(gs[row, :4])
            ax_cab.hist(cab_jpos[:, 0], bins=40, color="#2ca02c", alpha=0.75, edgecolor="white", linewidth=0.3)
            ax_cab.set_xlabel("Joint position (rad/m)")
            ax_cab.set_ylabel("count")
            ax_cab.set_title("Cabinet joint position")
            ax_cab.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        row += 1

    # ── last row: robot joint positions ─────────────────────────────────────
    n_cols = gs.get_geometry()[1]
    cols_per_joint = max(1, n_cols // 7)
    for ji in range(7):
        ax_j = fig.add_subplot(gs[row, ji * cols_per_joint: (ji + 1) * cols_per_joint])
        ax_j.hist(robot_joints[:, ji], bins=40, color="#6baed6", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax_j.set_xlabel("rad")
        ax_j.set_ylabel("count")
        ax_j.set_title(f"Robot {joint_labels[ji]}")
        ax_j.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    out_path = BASE_DIR / f"init_cond_{name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── summary overview ─────────────────────────────────────────────────────────

def plot_summary_overview():
    """One figure: XY scatter of every task object across all datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Task Object Initial XY Positions — All Datasets", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (name, cfg) in enumerate(DATASETS.items()):
        filepath = BASE_DIR / cfg["file"]
        ic, n_demos = load_initial_conditions(filepath)
        ax = axes[idx]
        ax.set_title(f"{name}  (N={n_demos})", fontsize=9)

        for j, obj_key in enumerate(cfg["task_objects"]):
            pose_key = obj_key + "/root_pose"
            if pose_key not in ic:
                continue
            pose = ic[pose_key]
            x, y = pose[:, 0], pose[:, 1]
            obj_name = obj_key.split("/")[-1]
            ax.scatter(x, y, s=5, alpha=0.3, color=COLORS[j % len(COLORS)], label=obj_name, rasterized=True)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(fontsize=7, markerscale=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    out_path = BASE_DIR / "init_cond_overview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_summary_overview()
    for name, cfg in DATASETS.items():
        print(f"Processing {name}...")
        plot_dataset(name, cfg)
    print("Done.")
