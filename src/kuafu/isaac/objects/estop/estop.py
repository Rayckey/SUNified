from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from kuafu.paths import OBJECT_ROOT
from typing import Sequence

import torch

from isaaclab.assets import Articulation


_ASSET_DIR = OBJECT_ROOT / "estop"
_DEFAULT_BASE_ASSET = (_ASSET_DIR.parent / "stop_base" / "base.usdc").resolve()
_DEFAULT_CAP_ASSET = (_ASSET_DIR.parent / "stop_cap" / "cap.usdc").resolve()
_DEFAULT_OUTPUT_USD = (_ASSET_DIR / "estop.usda").resolve()


@dataclass(slots=True)
class EStopBuilderCfg:
    """Geometry and mechanics used to compose the e-stop articulation USD."""

    base_asset_path: str = str(_DEFAULT_BASE_ASSET)
    cap_asset_path: str = str(_DEFAULT_CAP_ASSET)
    output_usd_path: str = str(_DEFAULT_OUTPUT_USD)
    base_to_carrier_offset: tuple[float, float, float] = (0.0, 0.0, 0.022)
    carrier_to_cap_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    press_travel: float = 0.01              # 1.0 cm total press stroke (matches estop.usda)
    twist_unlock_angle_rad: float = math.radians(100.0)  # 100° physical joint limit around -Z
    carrier_half_extents: tuple[float, float, float] = (0.008, 0.008, 0.01)
    carrier_mass: float = 0.02
    base_mass: float = 0.35
    cap_mass: float = 0.08
    # Primitive collision dimensions
    # Base: box 0.078 × 0.072 × 0.073 m; origin at the bottom of the geometry
    base_col_size: tuple[float, float, float] = (0.078, 0.072, 0.073)
    # Cap: cylinder with diameter 0.040 m (radius 0.020).
    # NOTE: the cap mesh origin is at its BOTTOM (bbox z: 0 → 0.040 m).
    cap_col_radius: float = 0.020
    cap_col_half_height: float = 0.020  # half of the 0.040 m cap mesh height
    # Extra upward offset for the cap above the base top surface (negative = recessed).
    cap_z_offset: float = -0.005
    # High-friction rubber-like material on the cap so the gripper does not slip.
    # frictionCombineMode="max" makes these values dominate regardless of the gripper material.
    cap_static_friction: float = 2.5
    cap_dynamic_friction: float = 2.5
    cap_restitution: float = 0.0


@dataclass(slots=True)
class EStopSemanticCfg:
    """High-level behavior defaults layered on top of the raw joint DOFs."""

    press_joint_name: str = "estop_press_joint"
    twist_joint_name: str = "estop_twist_joint"
    released_position: float = 0.0
    latched_position: float = 0.008        # 0.8 cm — latch trigger depth (must fit inside press_travel)
    unlock_twist_angle_rad: float = math.radians(60.0)  # 60° detection threshold
    unlock_tolerance_rad: float = 0.05
    # Optional effort-based unlock path for cases where contact happens but
    # measured twist angle remains small. Uses |twist_joint_effort| >= threshold.
    use_torque_unlock: bool = False
    unlock_twist_torque_threshold: float = torch.pi * 0.5  # ~1.57 N·m, tuned experimentally with the real button and sim feedback
    # EMA smoothing factor for |effort| (0 < alpha <= 1; lower = more smoothing).
    torque_ema_alpha: float = 0.1
    # Require this many consecutive smoothed-above-threshold steps before unlocking.
    torque_unlock_persistence_steps: int = 5
    # press_pos must be >= -release_tolerance to count as fully released
    release_tolerance: float = 0.002


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


def _apply_collision_apis(prim):
    from pxr import UsdGeom, UsdPhysics

    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    elif prim.IsA(UsdGeom.Cube):
        UsdPhysics.CollisionAPI.Apply(prim)


def _configure_referenced_body(prim, *, mass: float):
    # Apply rigid-body physics APIs only.
    # Collision is handled by explicit primitive shapes added after this call;
    # do NOT walk into the referenced mesh hierarchy and apply convexHull.
    _apply_rigid_apis(prim, mass=mass)


def build_estop_usd(cfg: EStopBuilderCfg) -> str:
    """Compose the articulated e-stop USD from the base and cap source assets."""

    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    sem = EStopSemanticCfg()

    base_asset = _as_path(cfg.base_asset_path)
    cap_asset = _as_path(cfg.cap_asset_path)
    output_path = _as_path(cfg.output_usd_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import os
    output_dir = output_path.parent
    base_asset_ref = os.path.relpath(str(base_asset), str(output_dir))
    cap_asset_ref = os.path.relpath(str(cap_asset), str(output_dir))

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/EStop")
    stage.SetDefaultPrim(root.GetPrim())
    root_prim = root.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(root_prim)

    base_link = UsdGeom.Xform.Define(stage, "/EStop/base_link")
    base_link_prim = base_link.GetPrim()
    base_link_prim.GetReferences().AddReference(base_asset_ref)
    _configure_referenced_body(base_link_prim, mass=cfg.base_mass)

    # Box collision for the base.
    # Base origin is at the bottom of the geometry, so translate the box up by half its height
    # so the collision shape is centred on the mesh.
    bx, by, bz = cfg.base_col_size
    base_col = UsdGeom.Cube.Define(stage, "/EStop/base_link/base_collision")
    base_col.CreateSizeAttr(1.0)
    base_col.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, bz / 2.0))
    base_col.AddScaleOp().Set(Gf.Vec3f(bx, by, bz))
    base_col.GetVisibilityAttr().Set("invisible")
    UsdPhysics.CollisionAPI.Apply(base_col.GetPrim())

    carrier_link = UsdGeom.Xform.Define(stage, "/EStop/carrier_link")
    carrier_link_prim = carrier_link.GetPrim()
    _apply_rigid_apis(carrier_link_prim, mass=cfg.carrier_mass)
    carrier_geom = UsdGeom.Cube.Define(stage, "/EStop/carrier_link/carrier_visual")
    carrier_geom.CreateSizeAttr(1.0)
    carrier_geom.AddScaleOp().Set(Gf.Vec3f(*(2.0 * value for value in cfg.carrier_half_extents)))
    carrier_geom.GetDisplayColorAttr().Set([(0.15, 0.15, 0.15)])
    carrier_geom.GetVisibilityAttr().Set("invisible")
    _apply_collision_apis(carrier_geom.GetPrim())

    cap_link = UsdGeom.Xform.Define(stage, "/EStop/cap_link")
    cap_link_prim = cap_link.GetPrim()
    cap_link_prim.GetReferences().AddReference(cap_asset_ref)
    _configure_referenced_body(cap_link_prim, mass=cfg.cap_mass)

    # Cylinder collision for the cap.
    # The cap mesh origin is at its BOTTOM (bbox z: 0 → 0.040 m), so translate the
    # cylinder up by cap_col_half_height to centre the collision shape on the mesh.
    cap_col = UsdGeom.Cylinder.Define(stage, "/EStop/cap_link/cap_collision")
    cap_col.CreateRadiusAttr(cfg.cap_col_radius)
    cap_col.CreateHeightAttr(cfg.cap_col_half_height * 2.0)
    cap_col.CreateAxisAttr("Z")
    cap_col.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, cfg.cap_col_half_height))
    cap_col.GetVisibilityAttr().Set("invisible")
    UsdPhysics.CollisionAPI.Apply(cap_col.GetPrim())

    # High-friction physics material on the cap so the gripper fingers don't slip.
    # Defined under /EStop/Looks/ and bound to the cap collision with purpose "physics".
    UsdGeom.Scope.Define(stage, "/EStop/Looks")
    cap_material = UsdShade.Material.Define(stage, "/EStop/Looks/CapPhysicsMaterial")
    cap_material_prim = cap_material.GetPrim()
    cap_phys_mat = UsdPhysics.MaterialAPI.Apply(cap_material_prim)
    cap_phys_mat.CreateStaticFrictionAttr(cfg.cap_static_friction)
    cap_phys_mat.CreateDynamicFrictionAttr(cfg.cap_dynamic_friction)
    cap_phys_mat.CreateRestitutionAttr(cfg.cap_restitution)
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(cap_material_prim)
    # "max" makes the higher of the two contacting materials win, so the cap stays sticky
    # regardless of whatever (default-friction) material is on the gripper fingers.
    physx_mat.CreateFrictionCombineModeAttr("max")
    physx_mat.CreateRestitutionCombineModeAttr("min")
    cap_col_prim = cap_col.GetPrim()
    cap_col_prim.ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(cap_col_prim).Bind(
        cap_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    joints = UsdGeom.Scope.Define(stage, "/EStop/Joints")

    # carrier_z: place the joint pivot at the top of the base collision box.
    # The cap mesh origin is at its bottom, so the cap mesh sits flush on the base top.
    # cap_z_offset can add a small gap if needed.
    carrier_z = cfg.base_col_size[2] + cfg.cap_z_offset
    base_to_carrier = (cfg.base_to_carrier_offset[0], cfg.base_to_carrier_offset[1], carrier_z)

    press_joint = UsdPhysics.PrismaticJoint.Define(stage, f"{joints.GetPath()}/{sem.press_joint_name}")
    press_joint.CreateAxisAttr("Z")
    press_joint.CreateBody0Rel().SetTargets([Sdf.Path("/EStop/base_link")])
    press_joint.CreateBody1Rel().SetTargets([Sdf.Path("/EStop/carrier_link")])
    press_joint.CreateLocalPos0Attr(Gf.Vec3f(*base_to_carrier))
    press_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    press_joint.CreateLowerLimitAttr(-cfg.press_travel)
    press_joint.CreateUpperLimitAttr(0.0)

    press_drive = UsdPhysics.DriveAPI.Apply(press_joint.GetPrim(), "linear")
    press_drive.CreateTypeAttr("force")
    press_drive.CreateTargetPositionAttr(0.0)
    press_drive.CreateTargetVelocityAttr(0.0)
    press_drive.CreateStiffnessAttr(800.0)
    press_drive.CreateDampingAttr(80.0)
    press_drive.CreateMaxForceAttr(200.0)
    PhysxSchema.PhysxJointAPI.Apply(press_joint.GetPrim()).CreateMaxJointVelocityAttr(0.5)

    twist_joint = UsdPhysics.RevoluteJoint.Define(stage, f"{joints.GetPath()}/{sem.twist_joint_name}")
    twist_joint.CreateAxisAttr("Z")
    twist_joint.CreateBody0Rel().SetTargets([Sdf.Path("/EStop/carrier_link")])
    twist_joint.CreateBody1Rel().SetTargets([Sdf.Path("/EStop/cap_link")])
    twist_joint.CreateLocalPos0Attr(Gf.Vec3f(*cfg.carrier_to_cap_offset))
    twist_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    twist_joint.CreateLowerLimitAttr(-cfg.twist_unlock_angle_rad * 180.0 / math.pi)
    twist_joint.CreateUpperLimitAttr(0.0)

    twist_drive = UsdPhysics.DriveAPI.Apply(twist_joint.GetPrim(), "angular")
    twist_drive.CreateTypeAttr("force")
    twist_drive.CreateTargetPositionAttr(0.0)
    twist_drive.CreateTargetVelocityAttr(0.0)
    twist_drive.CreateStiffnessAttr(0.5)
    twist_drive.CreateDampingAttr(0.8)
    twist_drive.CreateMaxForceAttr(1.0)
    PhysxSchema.PhysxJointAPI.Apply(twist_joint.GetPrim()).CreateMaxJointVelocityAttr(4.0)

    # Fixed joint: anchors base_link to the world frame.
    # This is required because the ArticulationRootAPI lives on the root Xform (which
    # has no RigidBodyAPI), so Isaac Lab cannot auto-create a fixed joint via
    # fix_root_link=True.  Defining it explicitly here is the correct USD approach.
    world_fix = UsdPhysics.FixedJoint.Define(stage, f"{joints.GetPath()}/world_fix")
    world_fix.CreateBody0Rel().SetTargets([])                             # empty = world frame
    world_fix.CreateBody1Rel().SetTargets([Sdf.Path("/EStop/base_link")])

    stage.Save()
    return str(output_path)


def _is_valid_estop_usd(path_like: str | Path) -> bool:
    """Return True when the cached USD exists and still contains the articulation root."""

    from pxr import Usd, UsdPhysics

    path = _as_path(path_like)
    if not path.exists() or path.stat().st_size == 0:
        return False

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return False

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        return False

    return default_prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def ensure_estop_usd(cfg: EStopBuilderCfg | None = None, *, rebuild: bool = False) -> str:
    """Build the composed USD on demand and return its path.

    Pass ``rebuild=True`` (or change any ``EStopBuilderCfg`` field) to force
    regeneration of the cached USD file.
    """

    cfg = cfg or EStopBuilderCfg()
    output_path = _as_path(cfg.output_usd_path)
    if rebuild or not _is_valid_estop_usd(output_path):
        return build_estop_usd(cfg)
    return str(output_path)


class EStopArticulation(Articulation):
    """Thin articulation wrapper with press/twist/release semantics."""

    def __init__(self, cfg, *, builder_cfg: EStopBuilderCfg | None = None, semantic_cfg: EStopSemanticCfg | None = None):
        self.builder_cfg = builder_cfg or EStopBuilderCfg()
        self.semantic_cfg = semantic_cfg or EStopSemanticCfg(
            latched_position=self.builder_cfg.press_travel,
            # unlock_twist_angle_rad stays at the semantic default threshold;
            # builder's twist_unlock_angle_rad is the physical joint limit, not the threshold.
        )
        ensure_estop_usd(self.builder_cfg, rebuild=True)
        # Lazily initialised after joint names become available (first update_state call)
        self._is_closed: torch.Tensor | None = None
        self._press_joint_idx: int | None = None
        self._twist_joint_idx: int | None = None
        # Smoothed |twist_effort| and consecutive-above-threshold counter for
        # the torque-based unlock path.
        self._twist_torque_ema: torch.Tensor | None = None
        self._twist_torque_above_count: torch.Tensor | None = None
        self._pressed_enough_prev: torch.Tensor | None = None
        self._unlocked_prev: torch.Tensor | None = None
        super().__init__(cfg)

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> tuple[torch.Tensor | None, int]:
        if env_ids is None:
            return None, self.num_instances
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        return env_ids_tensor, int(env_ids_tensor.numel())

    def _joint_id(self, joint_name: str) -> int:
        return self.joint_names.index(joint_name)

    def _ensure_state_initialized(self) -> None:
        """Resolve joint indices and allocate the closed-state tensor on first call."""
        if self._press_joint_idx is not None:
            return
        self._press_joint_idx = self.joint_names.index(self.semantic_cfg.press_joint_name)
        self._twist_joint_idx = self.joint_names.index(self.semantic_cfg.twist_joint_name)
        self._is_closed = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
        self._twist_torque_ema = torch.zeros(
            self.num_instances, dtype=torch.float32, device=self.device
        )
        self._twist_torque_above_count = torch.zeros(
            self.num_instances, dtype=torch.int32, device=self.device
        )
        self._pressed_enough_prev = torch.zeros(
            self.num_instances, dtype=torch.bool, device=self.device
        )
        self._unlocked_prev = torch.zeros(
            self.num_instances, dtype=torch.bool, device=self.device
        )

    def _twist_effort(self) -> torch.Tensor | None:
        """Return twist-joint effort if the backend exposes it, else None."""
        for attr_name in ("joint_effort", "applied_torque", "joint_torque"):
            effort_all = getattr(self.data, attr_name, None)
            if effort_all is None:
                continue
            try:
                return effort_all[:, self._twist_joint_idx]
            except Exception:
                continue
        return None

    def update_state(self, interaction_allowed: torch.Tensor | None = None) -> None:
        """Advance the latched-closed/open state machine from current joint positions.

        Call once per physics step (before querying is_closed / is_open).
        Transition rules:
          open  → closed  : press_pos <= -75 % of latched_position
          closed → open   : twist_pos <= -unlock threshold AND press_pos >= -release_tolerance

        Args:
            interaction_allowed: Optional boolean mask. When provided, open/close
                latch transitions are accepted only on the threshold-crossing step
                for envs where this mask is true.
        """
        self._ensure_state_initialized()
        assert self._is_closed is not None
        assert self._pressed_enough_prev is not None
        assert self._unlocked_prev is not None
        sem = self.semantic_cfg
        press_pos = self.data.joint_pos[:, self._press_joint_idx]
        twist_pos = self.data.joint_pos[:, self._twist_joint_idx]

        pressed_enough = press_pos <= -(0.75 * sem.latched_position)
        if interaction_allowed is None:
            close_trigger = pressed_enough
        else:
            allowed = interaction_allowed.to(device=self.device, dtype=torch.bool)
            close_trigger = pressed_enough & ~self._pressed_enough_prev & allowed
        self._is_closed |= close_trigger

        unlocked_by_angle = twist_pos <= -(sem.unlock_twist_angle_rad - sem.unlock_tolerance_rad)
        unlocked = unlocked_by_angle
        if sem.use_torque_unlock:
            twist_effort = self._twist_effort()
            if twist_effort is not None:
                assert self._twist_torque_ema is not None
                assert self._twist_torque_above_count is not None
                abs_effort = torch.abs(twist_effort).to(self._twist_torque_ema.dtype)
                alpha = float(sem.torque_ema_alpha)
                self._twist_torque_ema.mul_(1.0 - alpha).add_(abs_effort, alpha=alpha)
                above = self._twist_torque_ema >= sem.unlock_twist_torque_threshold
                self._twist_torque_above_count = torch.where(
                    above,
                    self._twist_torque_above_count + 1,
                    torch.zeros_like(self._twist_torque_above_count),
                )
                unlocked_by_torque = self._twist_torque_above_count >= int(
                    sem.torque_unlock_persistence_steps
                )
                unlocked = unlocked | unlocked_by_torque

        raw_unlocked = unlocked
        if interaction_allowed is not None:
            allowed = interaction_allowed.to(device=self.device, dtype=torch.bool)
            unlocked = unlocked & ~self._unlocked_prev & allowed

        # Once unlocked, command the press joint back to released position so the
        # button is allowed to spring up instead of being held latched down.
        unlocking_ids = (self._is_closed & unlocked).nonzero(as_tuple=False).squeeze(-1)
        if unlocking_ids.numel() > 0:
            release_target = torch.full(
                (unlocking_ids.numel(), 1), sem.released_position, device=self.device
            )
            self.set_joint_position_target(
                release_target, joint_ids=[self._press_joint_idx], env_ids=unlocking_ids
            )

        self._is_closed &= ~unlocked

        # Hold at latched depth only while still closed and not yet unlocked.
        closed_hold_mask = self._is_closed & ~unlocked
        if closed_hold_mask.any():
            closed_ids = closed_hold_mask.nonzero(as_tuple=False).squeeze(-1)
            target = torch.full(
                (closed_ids.numel(), 1), -sem.latched_position, device=self.device
            )
            self.set_joint_position_target(
                target, joint_ids=[self._press_joint_idx], env_ids=closed_ids
            )

        self._pressed_enough_prev.copy_(pressed_enough)
        self._unlocked_prev.copy_(unlocked_by_angle)

    def reset_state(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the closed/open latch for selected environments (pass None for all)."""
        if self._is_closed is None:
            return
        if env_ids is None:
            self._is_closed.zero_()
            if self._twist_torque_ema is not None:
                self._twist_torque_ema.zero_()
            if self._twist_torque_above_count is not None:
                self._twist_torque_above_count.zero_()
            if self._pressed_enough_prev is not None:
                self._pressed_enough_prev.zero_()
            if self._unlocked_prev is not None:
                self._unlocked_prev.zero_()
        else:
            self._is_closed[env_ids] = False
            if self._twist_torque_ema is not None:
                self._twist_torque_ema[env_ids] = 0.0
            if self._twist_torque_above_count is not None:
                self._twist_torque_above_count[env_ids] = 0
            if self._pressed_enough_prev is not None:
                self._pressed_enough_prev[env_ids] = False
            if self._unlocked_prev is not None:
                self._unlocked_prev[env_ids] = False

    def set_closed(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Force selected environments into the latched-closed state.

        Writes the press joint state to ``-latched_position`` so the button is
        physically pressed in, marks ``is_closed`` True, and sets the press
        drive target to hold it down. Use this at env launch / reset to start
        episodes with the e-stop already latched, without waiting for
        ``update_state`` to detect the press on the next physics step.
        """
        self._ensure_state_initialized()
        assert self._is_closed is not None
        sem = self.semantic_cfg
        resolved_env_ids, env_count = self._resolve_env_ids(env_ids)

        # Mark latched
        if resolved_env_ids is None:
            self._is_closed.fill_(True)
        else:
            self._is_closed[resolved_env_ids] = True

        # Write press joint state and target at the latched depth
        press_pos = torch.full((env_count, 1), -sem.latched_position, device=self.device)
        zero_vel = torch.zeros_like(press_pos)
        self.write_joint_state_to_sim(
            press_pos, zero_vel, joint_ids=[self._press_joint_idx], env_ids=resolved_env_ids
        )
        self.set_joint_position_target(
            press_pos, joint_ids=[self._press_joint_idx], env_ids=resolved_env_ids
        )

    @property
    def is_closed(self) -> torch.Tensor:
        """Bool tensor (num_envs,): True when the e-stop is latched closed."""
        self._ensure_state_initialized()
        assert self._is_closed is not None
        return self._is_closed

    @property
    def is_open(self) -> torch.Tensor:
        """Bool tensor (num_envs,): True when the e-stop is open (not latched)."""
        return ~self.is_closed

    def _set_single_joint_target(
        self, joint_name: str, value: float, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        resolved_env_ids, env_count = self._resolve_env_ids(env_ids)
        target = torch.full((env_count, 1), float(value), device=self.device)
        self.set_joint_position_target(target, joint_ids=[self._joint_id(joint_name)], env_ids=resolved_env_ids)

    def press(self, env_ids: Sequence[int] | torch.Tensor | None = None, depth: float | None = None) -> None:
        target_depth = self.semantic_cfg.latched_position if depth is None else float(depth)
        self._set_single_joint_target(self.semantic_cfg.press_joint_name, -abs(target_depth), env_ids)

    def unlock(self, env_ids: Sequence[int] | torch.Tensor | None = None, angle: float | None = None) -> None:
        target_angle = -abs(self.semantic_cfg.unlock_twist_angle_rad if angle is None else float(angle))
        self._set_single_joint_target(self.semantic_cfg.twist_joint_name, target_angle, env_ids)

    def release(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        resolved_env_ids, _ = self._resolve_env_ids(env_ids)
        press_joint_id = self._joint_id(self.semantic_cfg.press_joint_name)
        twist_joint_id = self._joint_id(self.semantic_cfg.twist_joint_name)
        if resolved_env_ids is None:
            press_pos = self.data.joint_pos[:, press_joint_id]
            twist_pos = self.data.joint_pos[:, twist_joint_id]
        else:
            press_pos = self.data.joint_pos[resolved_env_ids, press_joint_id]
            twist_pos = self.data.joint_pos[resolved_env_ids, twist_joint_id]
        unlocked_mask = twist_pos <= -(
            self.semantic_cfg.unlock_twist_angle_rad - self.semantic_cfg.unlock_tolerance_rad
        )
        press_targets = torch.where(
            unlocked_mask,
            torch.full_like(press_pos, self.semantic_cfg.released_position),
            press_pos,
        )
        twist_targets = torch.where(unlocked_mask, torch.zeros_like(twist_pos), twist_pos)
        self.set_joint_position_target(
            press_targets.unsqueeze(-1),
            joint_ids=[press_joint_id],
            env_ids=resolved_env_ids,
        )
        self.set_joint_position_target(
            twist_targets.unsqueeze(-1),
            joint_ids=[twist_joint_id],
            env_ids=resolved_env_ids,
        )
        return unlocked_mask

    def pull_up(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        return self.release(env_ids=env_ids)

    def set_pose_offsets(
        self,
        *,
        base_to_carrier_offset: tuple[float, float, float] | None = None,
        carrier_to_cap_offset: tuple[float, float, float] | None = None,
        rebuild: bool = True,
    ) -> str:
        if base_to_carrier_offset is not None:
            x, y, z = base_to_carrier_offset
            self.builder_cfg.base_to_carrier_offset = (float(x), float(y), float(z))
        if carrier_to_cap_offset is not None:
            x, y, z = carrier_to_cap_offset
            self.builder_cfg.carrier_to_cap_offset = (float(x), float(y), float(z))
        return ensure_estop_usd(self.builder_cfg, rebuild=rebuild)
