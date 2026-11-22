from __future__ import annotations

import math
from collections.abc import Iterable

ColliderSpec = tuple[
    str,
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
]

# Fitted in CupHolder-local coordinates from assets/objects/cup_hanger/cup_hanger.usdc.
# The visual mesh stays in place; these primitives replace its collision only.
CUP_HANGER_CYLINDER_COLLIDERS = (
    # name, center xyz, axis xyz, radius, height
    ("base", (0.0000, 0.0000, 0.0075), (0.0, 0.0, 1.0), 0.0800, 0.0150),
    ("stem", (0.0002, 0.0000, 0.1873), (0.0, 0.0, 1.0), 0.0140, 0.3460),
    (
        "peg_y_pos_lower",
        (0.0010, 0.0466, 0.1557),
        (0.0010, 0.8786, 0.4776),
        0.0055,
        0.1000,
    ),
    (
        "peg_y_neg_lower",
        (0.0013, -0.0467, 0.1560),
        (0.0023, -0.8882, 0.4595),
        0.0055,
        0.1000,
    ),
    (
        "peg_x_pos_middle",
        (0.0507, -0.0013, 0.2531),
        (0.8786, -0.0032, 0.4775),
        0.0055,
        0.1000,
    ),
    (
        "peg_x_neg_middle",
        (-0.0490, 0.0016, 0.2526),
        (-0.8750, 0.0073, 0.4841),
        0.0055,
        0.1000,
    ),
    (
        "peg_y_pos_top",
        (0.0008, 0.0501, 0.3382),
        (-0.0010, 0.9149, 0.4037),
        0.0055,
        0.0950,
    ),
    (
        "peg_y_neg_top",
        (0.0011, -0.0494, 0.3382),
        (0.0041, -0.9172, 0.3984),
        0.0055,
        0.0950,
    ),
)

# The Panda middle-branch task needs the upper pair to be visual-only: their
# physical colliders block the robot path to the reachable middle branch.
CUP_HANGER_NO_TOP_BRANCH_COLLIDERS = tuple(
    collider
    for collider in CUP_HANGER_CYLINDER_COLLIDERS
    if collider[0] not in {"peg_y_pos_top", "peg_y_neg_top"}
)



def _ring_wall_colliders(
    *,
    prefix: str,
    count: int,
    center_xy: tuple[float, float],
    wall_radius: float,
    segment_radius: float,
    z: float,
    height: float,
) -> tuple[ColliderSpec, ...]:
    return tuple(
        (
            f"{prefix}_{index:02d}",
            (
                center_xy[0] + wall_radius * math.cos(2.0 * math.pi * index / count),
                center_xy[1] + wall_radius * math.sin(2.0 * math.pi * index / count),
                z,
            ),
            (0.0, 0.0, 1.0),
            segment_radius,
            height,
        )
        for index in range(count)
    )


def _segment_collider(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> ColliderSpec:
    axis = tuple(end[index] - start[index] for index in range(3))
    height = math.sqrt(sum(component * component for component in axis))
    return (
        name,
        tuple((start[index] + end[index]) * 0.5 for index in range(3)),
        tuple(component / height for component in axis),
        radius,
        height,
    )


SIMPLE_HANGER_CYLINDER_COLLIDERS = (
    # Fitted from assets/objects/simple_hanger/assembly1_white_plastic_cyl_collision_static.usda.
    # The authored USD collision meshes are disabled at spawn time; these primitive
    # colliders match the visual base, pole, and single branch.
    ("base", (0.0000, 0.0000, 0.0075), (0.0, 0.0, 1.0), 0.0800, 0.0150),
    ("pole", (0.0000, 0.0000, 0.1000), (0.0, 0.0, 1.0), 0.0110, 0.1700),
    _segment_collider(
        "branch",
        (0.0000, -0.00921955, 0.1683561),
        (0.0000, -0.11276295, 0.2060421),
        0.0060,
    ),
)


def _mug_handle_colliders() -> tuple[ColliderSpec, ...]:
    radius = 0.0065
    points = (
        (0.0450, 0.0000, 0.0730),
        (0.0580, 0.0000, 0.0800),
        (0.0730, 0.0000, 0.0750),
        (0.0820, 0.0000, 0.0630),
        (0.0830, 0.0000, 0.0470),
        (0.0770, 0.0000, 0.0330),
        (0.0610, 0.0000, 0.0220),
        (0.0450, 0.0000, 0.0220),
    )
    return tuple(
        _segment_collider(f"handle_{index:02d}", points[index], points[index + 1], radius)
        for index in range(len(points) - 1)
    )


MUG_CYLINDER_COLLIDERS = (
    # The wall is a ring of vertical cylinders so the mug interior remains hollow.
    *_ring_wall_colliders(
        prefix="body_wall",
        count=32,
        center_xy=(0.0000, 0.0000),
        wall_radius=0.0450,
        segment_radius=0.0050,
        z=0.0460,
        height=0.0800,
    ),
    ("bottom", (0.0000, 0.0000, 0.0060), (0.0, 0.0, 1.0), 0.0410, 0.0100),
    *_mug_handle_colliders(),
)


def replace_mesh_collision_with_cylinders(
    prim,
    *,
    stage,
    colliders: Iterable[ColliderSpec],
    collision_props=None,
    material_path: str | None = None,
    collision_root_name: str = "cylinder_collisions",
    disable_existing_collision_enabled: bool = False,
) -> None:
    """Disable mesh collision and add fitted primitive cylinder colliders."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim import schemas
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    root_path = prim.GetPath().pathString

    def _remove_api_if_applied(child_prim, api_schema) -> None:
        try:
            has_api = child_prim.HasAPI(api_schema)
        except Exception:
            return
        if has_api:
            child_prim.RemoveAPI(api_schema)

    for child_prim in Usd.PrimRange(prim):
        if disable_existing_collision_enabled:
            collision_enabled_attr = child_prim.GetAttribute("physics:collisionEnabled")
            if collision_enabled_attr and collision_enabled_attr.IsValid():
                try:
                    collision_enabled_attr.Set(False)
                except Exception:
                    pass
        for api_schema in (
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.CollisionAPI,
            PhysxSchema.PhysxCollisionAPI,
        ):
            _remove_api_if_applied(child_prim, api_schema)

    collision_root = f"{root_path}/{collision_root_name}"
    UsdGeom.Xform.Define(stage, collision_root)

    for name, center, axis, radius, height in colliders:
        collider_path = f"{collision_root}/{name}"
        cylinder = UsdGeom.Cylinder.Define(stage, collider_path)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        cylinder.CreateExtentAttr(
            [
                Gf.Vec3f(-radius, -radius, -0.5 * height),
                Gf.Vec3f(radius, radius, 0.5 * height),
            ]
        )
        cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

        axis_vec = Gf.Vec3d(*axis).GetNormalized()
        rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), axis_vec)
        rotation_quat = rotation.GetQuat()

        xformable = UsdGeom.Xformable(cylinder.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
        xformable.AddOrientOp().Set(
            Gf.Quatf(
                rotation_quat.GetReal(),
                Gf.Vec3f(*rotation_quat.GetImaginary()),
            )
        )

        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        if collision_props is not None:
            schemas.define_collision_properties(collider_path, collision_props, stage)
        if material_path is not None:
            sim_utils.bind_physics_material(collider_path, material_path, stage)


def replace_mesh_collision_with_cup_hanger_cylinders(
    prim,
    *,
    stage,
    collision_props=None,
    material_path: str | None = None,
    colliders: Iterable[ColliderSpec] = CUP_HANGER_CYLINDER_COLLIDERS,
) -> None:
    """Disable cup hanger mesh collision and add fitted primitive cylinder colliders."""
    replace_mesh_collision_with_cylinders(
        prim,
        stage=stage,
        colliders=colliders,
        collision_props=collision_props,
        material_path=material_path,
    )


def replace_mesh_collision_with_simple_hanger_cylinders(
    prim,
    *,
    stage,
    collision_props=None,
    material_path: str | None = None,
    colliders: Iterable[ColliderSpec] = SIMPLE_HANGER_CYLINDER_COLLIDERS,
) -> None:
    """Disable imported simple-hanger USD collision and add primitive colliders."""
    replace_mesh_collision_with_cylinders(
        prim,
        stage=stage,
        colliders=colliders,
        collision_props=collision_props,
        material_path=material_path,
        disable_existing_collision_enabled=True,
    )


def replace_mesh_collision_with_mug_cylinders(
    prim,
    *,
    stage,
    collision_props=None,
    material_path: str | None = None,
    colliders: Iterable[ColliderSpec] = MUG_CYLINDER_COLLIDERS,
) -> None:
    """Disable mug mesh collision and add fitted primitive cylinder colliders."""
    replace_mesh_collision_with_cylinders(
        prim,
        stage=stage,
        colliders=colliders,
        collision_props=collision_props,
        material_path=material_path,
    )
