from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from kuafu.paths import external_artifact_root


_ASSET_DIR = Path(__file__).resolve().parent
_DEFAULT_BODY_ASSET = (_ASSET_DIR / "panda_drawer_body" / "panda_drawer_body.usdc").resolve()
_DEFAULT_DRAWER_ASSET = (_ASSET_DIR / "panda_drawer_drawer" / "panda_drawer_drawer.usdc").resolve()
_DEFAULT_OUTPUT_USD = (_ASSET_DIR / "panda_drawer.usda").resolve()
_DEFAULT_PREVIEW = external_artifact_root() / "asset_previews" / "panda_drawer_preview.png"


@dataclass(slots=True)
class PandaDrawerBuilderCfg:
    """Geometry and mechanics used to compose the two-part Panda drawer USD."""

    body_asset_path: str = str(_DEFAULT_BODY_ASSET)
    drawer_asset_path: str = str(_DEFAULT_DRAWER_ASSET)
    output_usd_path: str = str(_DEFAULT_OUTPUT_USD)
    preview_path: str = str(_DEFAULT_PREVIEW)

    drawer_joint_name: str = "upper_drawer_joint"
    drawer_joint_travel: float = 0.15
    max_joint_velocity: float = 10.0

    body_mass: float = 2.0
    drawer_mass: float = 0.6
    handle_mass: float = 0.05
    base_mass: float = 0.01

    # The body source already opens toward local -Y.  The drawer source is rotated
    # 180 deg around Z so its handle/front also faces local -Y, then translated into
    # the body opening.
    drawer_visual_offset: tuple[float, float, float] = (0.0, 0.081, 0.12)
    handle_center: tuple[float, float, float] = (0.0, -0.017, 0.12)

    # Primitive collision dimensions.  These replace raw mesh collision while keeping
    # the cabinet cavity and drawer tray open so objects can sit inside the drawer.
    frame_shell_thickness: float = 0.012
    drawer_wall_thickness: float = 0.01
    drawer_floor_thickness: float = 0.01
    # The handle is a D-shaped protrusion: a rounded front half that transitions
    # into a rectangular rear tongue.  The box begins at the cylinder center plane.
    handle_grip_center: tuple[float, float, float] = (0.0, -0.007, 0.0)
    handle_grip_length: float = 0.1582
    handle_grip_radius: float = 0.0085
    handle_mount_center: tuple[float, float, float] = (0.0, 0.00425, 0.0)
    handle_mount_size: tuple[float, float, float] = (0.1582, 0.0225, 0.017)


def _as_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _apply_rigid_apis(prim, *, mass: float):
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_api.CreateEnableGyroscopicForcesAttr(True)
    physx_api.CreateSolverPositionIterationCountAttr(16)
    physx_api.CreateSolverVelocityIterationCountAttr(4)


def _define_collision_cube(stage, path: str, *, center: tuple[float, float, float], size: tuple[float, float, float]):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3f(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _define_collision_cylinder(
    stage,
    path: str,
    *,
    center: tuple[float, float, float],
    radius: float,
    height: float,
    axis: str = "X",
):
    from pxr import Gf, UsdGeom, UsdPhysics

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    cylinder.CreateAxisAttr(axis)
    cylinder.AddTranslateOp().Set(Gf.Vec3f(*center))
    cylinder.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return cylinder


def frame_collision_boxes(cfg: PandaDrawerBuilderCfg) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Simplified cabinet shell collision boxes with the drawer cavity open."""

    cx, cy, cz = 0.0, 0.078826, 0.111325
    sx, sy, sz = 0.33, 0.174288, 0.222
    t = cfg.frame_shell_thickness

    # Keep the central drawer cavity open while covering the visible body shell.
    # Retract the front-added frame collision 1 cm from the visual front face.
    front_collision_y_min = cy - sy / 2.0 + 0.01
    side_y_min = front_collision_y_min
    side_y_max = cy + sy / 2.0
    side_y = (side_y_min + side_y_max) / 2.0
    side_sy = side_y_max - side_y_min
    side_z_min = 0.000325
    side_z_max = cz + sz / 2.0
    side_z = (side_z_min + side_z_max) / 2.0
    side_sz = side_z_max - side_z_min

    roof_x_min, roof_x_max = -0.156, 0.156
    roof_y_min, roof_y_max = 0.0, 0.1585
    roof_z_min, roof_z_max = 0.151, 0.1595
    roof_center = (
        (roof_x_min + roof_x_max) / 2.0,
        (roof_y_min + roof_y_max) / 2.0,
        (roof_z_min + roof_z_max) / 2.0,
    )
    roof_size = (
        roof_x_max - roof_x_min,
        roof_y_max - roof_y_min,
        roof_z_max - roof_z_min,
    )

    front_y_min, front_y_max = front_collision_y_min, 0.018
    front_y = (front_y_min + front_y_max) / 2.0
    front_sy = front_y_max - front_y_min
    lower_front_z_min, lower_front_z_max = side_z_min, 0.08709
    lower_front_z = (lower_front_z_min + lower_front_z_max) / 2.0
    lower_front_sz = lower_front_z_max - lower_front_z_min
    roof_front_y_min, roof_front_y_max = front_collision_y_min, front_collision_y_min + 0.006818
    roof_front_y = (roof_front_y_min + roof_front_y_max) / 2.0
    roof_front_sy = roof_front_y_max - roof_front_y_min

    return (
        ("left_side_rail_collision", (cx - sx / 2.0 + t / 2.0, side_y, side_z), (t, side_sy, side_sz)),
        ("right_side_rail_collision", (cx + sx / 2.0 - t / 2.0, side_y, side_z), (t, side_sy, side_sz)),
        ("bottom_collision", (cx, cy, cz - sz / 2.0 + t / 2.0), (sx, sy, t)),
        ("back_wall_collision", (cx, cy + sy / 2.0 - t / 2.0, cz), (sx, t, sz)),
        ("roof_collision", roof_center, roof_size),
        ("front_lower_panel_collision", (cx, front_y, lower_front_z), (sx, front_sy, lower_front_sz)),
        ("front_roof_lip_collision", (cx, roof_front_y, roof_center[2]), (roof_size[0], roof_front_sy, roof_size[2])),
    )


def drawer_tray_collision_boxes(cfg: PandaDrawerBuilderCfg) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Open-top drawer tray collision boxes: floor, side walls, back, and front lip."""

    cx, cy, cz = 0.0, 0.08107, 0.12009
    sx, sy, sz = 0.305, 0.165, 0.066
    wall_t = cfg.drawer_wall_thickness
    floor_t = cfg.drawer_floor_thickness
    wall_z = sz - floor_t
    wall_cz = cz - sz / 2.0 + floor_t + wall_z / 2.0
    return (
        ("floor_collision", (cx, cy, cz - sz / 2.0 + floor_t / 2.0), (sx, sy, floor_t)),
        ("left_wall_collision", (cx - sx / 2.0 + wall_t / 2.0, cy, wall_cz), (wall_t, sy, wall_z)),
        ("right_wall_collision", (cx + sx / 2.0 - wall_t / 2.0, cy, wall_cz), (wall_t, sy, wall_z)),
        ("back_wall_collision", (cx, cy + sy / 2.0 - wall_t / 2.0, wall_cz), (sx, wall_t, wall_z)),
        ("front_lip_collision", (cx, cy - sy / 2.0 + wall_t / 2.0, wall_cz), (sx, wall_t, wall_z)),
    )


def build_panda_drawer_usd(cfg: PandaDrawerBuilderCfg) -> str:
    """Compose the articulated Panda drawer USD from body and drawer source assets."""

    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    body_asset = _as_path(cfg.body_asset_path)
    drawer_asset = _as_path(cfg.drawer_asset_path)
    output_path = _as_path(cfg.output_usd_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = output_path.parent
    body_asset_ref = os.path.relpath(str(body_asset), str(output_dir))
    drawer_asset_ref = os.path.relpath(str(drawer_asset), str(output_dir))

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/PandaDrawer")
    stage.SetDefaultPrim(root.GetPrim())

    base = UsdGeom.Xform.Define(stage, "/PandaDrawer/base")
    _apply_rigid_apis(base.GetPrim(), mass=cfg.base_mass)

    frame = UsdGeom.Xform.Define(stage, "/PandaDrawer/frame")
    _apply_rigid_apis(frame.GetPrim(), mass=cfg.body_mass)

    frame_visuals = UsdGeom.Scope.Define(stage, "/PandaDrawer/frame/visuals")
    body_visual = UsdGeom.Xform.Define(stage, f"{frame_visuals.GetPath()}/body_visual")
    body_source = UsdGeom.Xform.Define(stage, f"{body_visual.GetPath()}/source")
    body_source.GetPrim().GetReferences().AddReference(body_asset_ref)

    frame_collisions = UsdGeom.Scope.Define(stage, "/PandaDrawer/frame/collisions")
    for name, center, size in frame_collision_boxes(cfg):
        _define_collision_cube(stage, f"{frame_collisions.GetPath()}/{name}", center=center, size=size)

    drawer = UsdGeom.Xform.Define(stage, "/PandaDrawer/upper_drawer")
    _apply_rigid_apis(drawer.GetPrim(), mass=cfg.drawer_mass)

    drawer_visuals = UsdGeom.Scope.Define(stage, "/PandaDrawer/upper_drawer/visuals")
    drawer_visual = UsdGeom.Xform.Define(stage, f"{drawer_visuals.GetPath()}/drawer_visual")
    drawer_visual.AddTranslateOp().Set(Gf.Vec3f(*cfg.drawer_visual_offset))
    drawer_source = UsdGeom.Xform.Define(stage, f"{drawer_visual.GetPath()}/source")
    drawer_source.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 180.0))
    drawer_source.GetPrim().GetReferences().AddReference(drawer_asset_ref)

    drawer_collisions = UsdGeom.Scope.Define(stage, "/PandaDrawer/upper_drawer/collisions")
    for name, center, size in drawer_tray_collision_boxes(cfg):
        _define_collision_cube(stage, f"{drawer_collisions.GetPath()}/{name}", center=center, size=size)

    handle = UsdGeom.Xform.Define(stage, "/PandaDrawer/upper_drawer_handle")
    handle.AddTranslateOp().Set(Gf.Vec3f(*cfg.handle_center))
    _apply_rigid_apis(handle.GetPrim(), mass=cfg.handle_mass)
    handle_collisions = UsdGeom.Scope.Define(stage, "/PandaDrawer/upper_drawer_handle/collisions")
    _define_collision_cylinder(
        stage,
        f"{handle_collisions.GetPath()}/handle_grip_collision",
        center=cfg.handle_grip_center,
        radius=cfg.handle_grip_radius,
        height=cfg.handle_grip_length,
        axis="X",
    )
    _define_collision_cube(
        stage,
        f"{handle_collisions.GetPath()}/handle_mount_collision",
        center=cfg.handle_mount_center,
        size=cfg.handle_mount_size,
    )

    # Copy the old drawer's articulation contract: root_joint marks the single articulation root,
    # /frame/upper_drawer_joint is the controllable prismatic joint, and the handle is a rigid link.
    base_to_frame = UsdPhysics.FixedJoint.Define(stage, "/PandaDrawer/base/base_to_frame")
    base_to_frame.CreateBody0Rel().SetTargets([Sdf.Path("/PandaDrawer/base")])
    base_to_frame.CreateBody1Rel().SetTargets([Sdf.Path("/PandaDrawer/frame")])
    base_to_frame.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    base_to_frame.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    root_joint = UsdPhysics.FixedJoint.Define(stage, "/PandaDrawer/root_joint")
    root_joint.CreateBody0Rel().SetTargets([])
    root_joint.CreateBody1Rel().SetTargets([Sdf.Path("/PandaDrawer/base")])
    UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(root_joint.GetPrim())

    drawer_joint = UsdPhysics.PrismaticJoint.Define(stage, f"/PandaDrawer/frame/{cfg.drawer_joint_name}")
    drawer_joint.CreateAxisAttr("Y")
    drawer_joint.CreateBody0Rel().SetTargets([Sdf.Path("/PandaDrawer/frame")])
    drawer_joint.CreateBody1Rel().SetTargets([Sdf.Path("/PandaDrawer/upper_drawer")])
    drawer_joint.CreateLocalRot0Attr(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    drawer_joint.CreateLocalRot1Attr(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    drawer_joint.CreateLowerLimitAttr(0.0)
    drawer_joint.CreateUpperLimitAttr(cfg.drawer_joint_travel)

    drawer_drive = UsdPhysics.DriveAPI.Apply(drawer_joint.GetPrim(), "linear")
    drawer_drive.CreateTypeAttr("force")
    drawer_drive.CreateTargetVelocityAttr(0.0)
    drawer_drive.CreateStiffnessAttr(0.0)
    drawer_drive.CreateDampingAttr(0.0)
    drawer_drive.CreateMaxForceAttr(0.1)
    PhysxSchema.PhysxJointAPI.Apply(drawer_joint.GetPrim()).CreateMaxJointVelocityAttr(cfg.max_joint_velocity)

    handle_joint = UsdPhysics.FixedJoint.Define(stage, "/PandaDrawer/upper_drawer/upper_drawer_handle_joint")
    handle_joint.CreateBody0Rel().SetTargets([Sdf.Path("/PandaDrawer/upper_drawer")])
    handle_joint.CreateBody1Rel().SetTargets([Sdf.Path("/PandaDrawer/upper_drawer_handle")])
    handle_joint.CreateLocalPos0Attr(Gf.Vec3f(*cfg.handle_center))
    handle_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    cameras = UsdGeom.Scope.Define(stage, "/PandaDrawer/Cameras")

    def add_camera(name: str, eye, target, up):
        camera = UsdGeom.Camera.Define(stage, f"{cameras.GetPath()}/{name}")
        look = Gf.Matrix4d(1.0)
        look.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
        camera.AddTransformOp().Set(look.GetInverse())
        camera.CreateFocalLengthAttr(45.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(16.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 5.0))

    add_camera("front_minus_y", (0.0, -0.55, 0.13), (0.0, 0.075, 0.12), (0.0, 0.0, 1.0))
    add_camera("right_side", (0.55, 0.075, 0.14), (0.0, 0.075, 0.12), (0.0, 0.0, 1.0))
    add_camera("top", (0.0, 0.075, 0.65), (0.0, 0.075, 0.12), (0.0, 1.0, 0.0))
    add_camera("three_quarter", (0.45, -0.48, 0.32), (0.0, 0.075, 0.12), (0.0, 0.0, 1.0))

    stage.Save()
    return str(output_path)


def _box_edges(center, size):
    import numpy as np

    center = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    corners = np.array(
        [
            [center[0] + sx * half[0], center[1] + sy * half[1], center[2] + sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=float,
    )
    edge_idx = (
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    )
    return [(corners[a], corners[b]) for a, b in edge_idx]


def render_preview(cfg: PandaDrawerBuilderCfg) -> str:
    """Render a small diagnostic image of the closed drawer and the pull direction."""

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from pxr import Usd, UsdGeom

    usd_path = _as_path(cfg.output_usd_path)
    preview_path = _as_path(cfg.preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD: {usd_path}")

    def collect_meshes(open_offset: float):
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        meshes = []
        all_points = []

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if not points:
                continue

            world = cache.GetLocalToWorldTransform(prim)
            pts = np.array([tuple(world.Transform(point)) for point in points], dtype=float)
            path = prim.GetPath().pathString
            moving = "/upper_drawer/" in path or "/upper_drawer_handle/" in path
            if moving:
                pts[:, 1] -= open_offset

            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            faces = []
            cursor = 0
            step = max(1, len(counts) // 900)
            for face_index, count in enumerate(counts):
                face_indices = indices[cursor : cursor + count]
                cursor += count
                if face_index % step != 0 or count < 3:
                    continue
                faces.append(pts[list(face_indices)])

            color = "#6c8497" if "/frame/" in path else "#d67f38"
            meshes.append((faces, color))
            all_points.append(pts)

        return meshes, np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))

    def add_collision_box(ax, center, size, color, *, open_offset=0.0):
        center = np.asarray(center, dtype=float).copy()
        center[1] -= open_offset
        for start, end in _box_edges(center, size):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=1.0,
                alpha=0.95,
            )

    def add_collision_cylinder_x(ax, center, radius, length, color, *, open_offset=0.0):
        center = np.asarray(center, dtype=float).copy()
        center[1] -= open_offset
        theta = np.linspace(0.0, 2.0 * np.pi, 48)
        x0 = center[0] - length / 2.0
        x1 = center[0] + length / 2.0
        for x in (x0, x1):
            ax.plot(
                np.full_like(theta, x),
                center[1] + radius * np.cos(theta),
                center[2] + radius * np.sin(theta),
                color=color,
                linewidth=1.0,
                alpha=0.95,
            )
        for t in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            y = center[1] + radius * np.cos(t)
            z = center[2] + radius * np.sin(t)
            ax.plot([x0, x1], [y, y], [z, z], color=color, linewidth=0.8, alpha=0.9)

    def set_equal_axes(ax, points):
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max((maxs - mins).max() / 2.0, 0.12)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=22.0, azim=-62.0)

    fig = plt.figure(figsize=(12.0, 5.2), dpi=160)
    fig.suptitle("Panda drawer composition: closed fit and prismatic pull direction", fontsize=12)

    for panel_index, (title, open_offset) in enumerate((("closed", 0.0), ("opened 12 cm along local -Y", 0.12)), start=1):
        ax = fig.add_subplot(1, 2, panel_index, projection="3d")
        meshes, points = collect_meshes(open_offset)
        for faces, color in meshes:
            if not faces:
                continue
            collection = Poly3DCollection(faces, facecolor=color, edgecolor="#202020", linewidths=0.05, alpha=0.42)
            ax.add_collection3d(collection)

        for _, center, size in frame_collision_boxes(cfg):
            add_collision_box(ax, center, size, "#252525")
        for _, center, size in drawer_tray_collision_boxes(cfg):
            add_collision_box(ax, center, size, "#b85d1e", open_offset=open_offset)
        add_collision_cylinder_x(
            ax,
            tuple(cfg.handle_center[i] + cfg.handle_grip_center[i] for i in range(3)),
            cfg.handle_grip_radius,
            cfg.handle_grip_length,
            "#b21f2d",
            open_offset=open_offset,
        )
        add_collision_box(
            ax,
            tuple(cfg.handle_center[i] + cfg.handle_mount_center[i] for i in range(3)),
            cfg.handle_mount_size,
            "#b21f2d",
            open_offset=open_offset,
        )
        ax.quiver(
            cfg.handle_center[0],
            cfg.handle_center[1],
            cfg.handle_center[2],
            0.0,
            -0.08,
            0.0,
            color="#111111",
            arrow_length_ratio=0.25,
            linewidth=1.5,
        )
        ax.set_title(title, fontsize=10)
        set_equal_axes(ax, points)

    fig.tight_layout()
    fig.savefig(str(preview_path))
    plt.close(fig)
    return str(preview_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the composed Panda drawer articulation USD.")
    parser.add_argument("--body-asset", default=str(_DEFAULT_BODY_ASSET))
    parser.add_argument("--drawer-asset", default=str(_DEFAULT_DRAWER_ASSET))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT_USD))
    parser.add_argument("--preview", default=str(_DEFAULT_PREVIEW))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    cfg = PandaDrawerBuilderCfg(
        body_asset_path=args.body_asset,
        drawer_asset_path=args.drawer_asset,
        output_usd_path=args.output,
        preview_path=args.preview,
    )
    usd_path = build_panda_drawer_usd(cfg)
    print(f"Wrote {usd_path}")
    if not args.no_preview:
        preview_path = render_preview(cfg)
        print(f"Wrote {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
