"""The model emits data in this schema, never executable Python or shell commands."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0, le=1000)]
Coordinate = Annotated[float, Field(ge=-1000, le=1000)]
Unit = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Vec3(StrictModel):
    x: Coordinate
    y: Coordinate
    z: Coordinate

    def array(self) -> list[float]:
        return [self.x, self.y, self.z]


class Dimensions(StrictModel):
    x: Positive
    y: Positive
    z: Positive

    def array(self) -> list[float]:
        return [self.x, self.y, self.z]


class Color(StrictModel):
    r: Unit
    g: Unit
    b: Unit

    def array(self) -> list[float]:
        return [self.r, self.g, self.b]


class Box(StrictModel):
    kind: Literal["box"]
    size: Dimensions


class RoundedBox(StrictModel):
    """Solid box with a spherical fillet at all edges; centered at local origin."""

    kind: Literal["rounded_box"]
    size: Dimensions
    radius: Positive

    @model_validator(mode="after")
    def usable_core(self):
        if 2 * self.radius >= min(self.size.array()):
            raise ValueError("rounded_box radius must be smaller than half its smallest dimension")
        return self


class Ellipsoid(StrictModel):
    kind: Literal["ellipsoid"]
    radii: Dimensions


class Cylinder(StrictModel):
    kind: Literal["cylinder"]
    radius: Positive
    height: Positive


class Frustum(StrictModel):
    kind: Literal["frustum"]
    radius_bottom: Positive
    radius_top: Annotated[float, Field(ge=0, le=1000)]
    height: Positive


class Torus(StrictModel):
    kind: Literal["torus"]
    major_radius: Positive
    minor_radius: Positive

    @model_validator(mode="after")
    def ring_has_hole(self):
        if self.minor_radius >= self.major_radius:
            raise ValueError("torus minor_radius must be smaller than major_radius")
        return self


class ProfilePoint(StrictModel):
    radius: Annotated[float, Field(ge=0, le=1000)]
    z: Coordinate


class Lathe(StrictModel):
    """Revolve a closed radial cross section; supports hollow cups and bottles."""

    kind: Literal["lathe"]
    profile: list[ProfilePoint] = Field(min_length=3, max_length=64)


class Sweep(StrictModel):
    """A round tube along a polyline, with spherical joints and end caps."""

    kind: Literal["sweep"]
    radius: Positive
    points: list[Vec3] = Field(min_length=2, max_length=32)


class Hull(StrictModel):
    kind: Literal["convex_hull"]
    points: list[Vec3] = Field(min_length=4, max_length=128)


class RoundedContainer(StrictModel):
    """A hollow rounded rectangle with a rounded rim, closed floor and slight wall taper."""

    kind: Literal["rounded_container"]
    size: Dimensions
    corner_radius: Positive
    wall_thickness: Positive
    base_thickness: Positive
    bottom_inset: Annotated[float, Field(ge=0, le=1000)]
    edge_radius: Positive

    @model_validator(mode="after")
    def usable_cavity(self):
        half_width = min(self.size.x, self.size.y) / 2
        if self.corner_radius >= half_width:
            raise ValueError("corner_radius must be smaller than half the width and depth")
        if self.wall_thickness + self.bottom_inset + self.edge_radius >= self.corner_radius:
            raise ValueError("corner_radius must exceed wall_thickness + bottom_inset + edge_radius")
        if self.edge_radius > self.base_thickness:
            raise ValueError("edge_radius must not exceed base_thickness")
        if self.base_thickness + self.edge_radius + self.wall_thickness >= self.size.z:
            raise ValueError("container needs a non-zero interior depth above its base fillet")
        return self


class Part(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    # All part transforms describe the assembled, zero-joint-position pose.
    link: str = "base"
    geometry: (
        Box | RoundedBox | Ellipsoid | Cylinder | Frustum | Torus | Lathe | Sweep | Hull | RoundedContainer
    )
    position: Vec3
    rotation_deg: Vec3
    color: Color
    roughness: Unit
    metallic: Unit
    evidence: str = Field(max_length=2000)


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]


class SemanticAxis(StrictModel):
    """A unit direction expressed in the interaction frame, not the camera/world."""

    name: Identifier
    direction: Vec3
    description: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def unit_direction(self):
        if abs(sum(v * v for v in self.direction.array()) - 1) > 1e-5:
            raise ValueError("Semantic axis direction must be a unit vector")
        if self.name in {"joint_positive", "open", "close"}:
            raise ValueError("Joint motion axes are derived by the exporter, not authored by the model")
        return self


class InteractionFrame(StrictModel):
    name: Identifier
    body: Identifier = "base"
    role: Literal["grasp", "center", "hinge", "contact", "insertion", "support", "tool", "other"]
    description: str = Field(min_length=1, max_length=1000)
    # SAME assembled rest coordinates as the geometry, before export normalization.
    position: Vec3
    rotation_deg: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))
    size: Dimensions | None = None  # Interaction region dimensions in this frame's axes.
    part: str | None = None  # Optional evidence/ownership reference, NOT an extra transform.
    axes: list[SemanticAxis] = Field(default_factory=list, max_length=12)
    joint: Identifier | None = None
    opening_direction: Literal["positive", "negative"] | None = None
    evidence: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def usable_frame(self):
        if not self.name.endswith("_frame"):
            raise ValueError("Interaction frame names must end in _frame for environment sensor lookup")
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("Semantic axis names must be unique within a frame")
        if (self.joint is None) != (self.opening_direction is None):
            raise ValueError("A motion frame needs both joint and opening_direction")
        return self


class ObjectSemantics(StrictModel):
    name: Identifier  # Default scene/detectable_objects key, e.g. cabinet.
    description: str = Field(min_length=1, max_length=2000)
    axes: list[SemanticAxis] = Field(default_factory=list, max_length=12)
    frames: list[InteractionFrame] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_names(self):
        names = [frame.name for frame in self.frames]
        if len(set(names)) != len(names) or self.name in names:
            raise ValueError("Object and interaction frame names must be unique")
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("Object semantic axis names must be unique")
        return self


class Link(StrictModel):
    name: Identifier
    mass_fraction: Positive = 1.0


class Joint(StrictModel):
    name: Identifier
    parent: Identifier
    child: Identifier
    kind: Literal["prismatic", "revolute", "fixed"]
    axis: Literal["X", "Y", "Z", "-X", "-Y", "-Z"] = "-Y"
    # Anchor in the SAME assembled coordinate system as the parts, before scaling.
    origin: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))
    # Meters for prismatic, degrees for revolute. Zero is the modeled pose.
    lower: Coordinate = 0
    upper: Coordinate = 0
    damping: Annotated[float, Field(ge=0, le=10000)] = 3.0
    max_effort: Positive = 87.0

    @model_validator(mode="after")
    def usable_limits(self):
        if self.parent == self.child:
            raise ValueError("A joint must connect distinct links")
        if self.kind == "fixed":
            if self.lower != 0 or self.upper != 0:
                raise ValueError("Fixed joints must have zero limits")
        elif not self.lower <= 0 <= self.upper or self.lower >= self.upper:
            raise ValueError("Moving joint limits must enclose zero and have positive travel")
        return self


class ArticulationSpec(StrictModel):
    root: Identifier
    fixed_base: bool = True
    links: list[Link] = Field(min_length=2, max_length=32)
    joints: list[Joint] = Field(min_length=1, max_length=31)

    @model_validator(mode="after")
    def connected_tree(self):
        names = [link.name for link in self.links]
        joint_names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names) or len(set(joint_names)) != len(joint_names):
            raise ValueError("Link and joint names must be unique")
        if set(names + joint_names) & {"Looks", "root_joint"} or set(names) & set(joint_names):
            raise ValueError("Link/joint names conflict with another prim or a reserved name")
        if self.root not in names or len(self.joints) != len(names) - 1:
            raise ValueError("Articulation must be a tree with one joint per non-root link")
        parents = {}
        for joint in self.joints:
            if joint.parent not in names or joint.child not in names or joint.child == self.root:
                raise ValueError("Joint references an unknown link or drives the root")
            if joint.child in parents:
                raise ValueError("A link cannot have multiple parent joints")
            parents[joint.child] = joint.parent
        for name in names:
            seen = set()
            while name != self.root:
                if name in seen or name not in parents:
                    raise ValueError("Articulation contains a cycle or disconnected link")
                seen.add(name)
                name = parents[name]
        if not any(joint.kind != "fixed" for joint in self.joints):
            raise ValueError("An articulation needs at least one moving joint")
        return self


class SceneSpec(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=4000)
    # All lengths in the scene are meters, even when scale is only an estimate.
    scale_basis: Literal["estimated", "user_measurement"]
    assumptions: list[str] = Field(max_length=32)
    parts: list[Part] = Field(min_length=1, max_length=48)
    articulation: ArticulationSpec | None = None
    # Legacy recipes remain buildable; new agent candidates must supply semantics.
    semantics: ObjectSemantics | None = None

    @model_validator(mode="after")
    def unique_names(self):
        names = [part.name for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("part names must be unique")
        links = {part.link for part in self.parts}
        if self.articulation:
            if links != {link.name for link in self.articulation.links}:
                raise ValueError(
                    "Every articulated link needs geometry; every part must belong to a defined link"
                )
        elif links != {"base"}:
            raise ValueError("Rigid scene parts must use the default base link")
        if self.semantics:
            parts = {part.name: part for part in self.parts}
            joints = {joint.name: joint for joint in self.articulation.joints} if self.articulation else {}
            parents = {joint.child: joint for joint in joints.values()}
            for frame in self.semantics.frames:
                if frame.body not in links:
                    raise ValueError(f"Interaction frame {frame.name} references an unknown body")
                if frame.part is not None and (
                    frame.part not in parts or parts[frame.part].link != frame.body
                ):
                    raise ValueError(f"Interaction frame {frame.name} part must belong to its body")
                if frame.joint is None:
                    if frame.role == "hinge":
                        raise ValueError("A hinge frame needs its revolute joint")
                    continue
                joint = joints.get(frame.joint)
                if joint is None or joint.kind == "fixed":
                    raise ValueError(f"Interaction frame {frame.name} needs a moving joint")
                body = frame.body
                while body != joint.child and body in parents and parents[body].kind == "fixed":
                    body = parents[body].parent
                if body != joint.child:
                    raise ValueError("A joint frame must follow its child body or a fixed descendant")
                if (frame.opening_direction == "positive" and joint.upper <= 0) or (
                    frame.opening_direction == "negative" and joint.lower >= 0
                ):
                    raise ValueError("Opening direction must have travel from the zero rest pose")
                if frame.role == "hinge" and (
                    joint.kind != "revolute"
                    or any(abs(a - b) > 1e-6 for a, b in zip(frame.position.array(), joint.origin.array()))
                ):
                    raise ValueError("A hinge frame must be at its revolute joint origin")
        return self


class Review(StrictModel):
    # This is a subjective visual score, not a geometric accuracy measurement.
    score: Unit
    acceptable: bool
    issues: list[str] = Field(max_length=24)
    suggested_changes: list[str] = Field(max_length=24)
    uncertainty: list[str] = Field(max_length=24)


class Scale(StrictModel):
    size_m: Positive
    axis: Literal["x", "y", "z", "longest"]


class Frame(StrictModel):
    path: str
    index: int = Field(ge=0)
    time_s: float = Field(ge=0)
    sharpness: float = Field(ge=0)
    sha256: str
    source_index: int = Field(default=0, ge=0, le=3)


class ReferenceVideo(StrictModel):
    source_video: str
    source_sha256: str
    fps: float = Field(gt=0)
    duration_s: float = Field(gt=0)
    frame_count: int = Field(gt=0)


class PreparedInput(StrictModel):
    version: Literal[1]
    source_video: str
    source_sha256: str
    description: str = Field(min_length=1, max_length=16000)
    fps: float = Field(gt=0)
    duration_s: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    crop: list[Unit] | None
    scale: Scale | None
    frames: list[Frame] = Field(min_length=1, max_length=32)
    warnings: list[str]
    reference_videos: list[ReferenceVideo] = Field(default_factory=list, max_length=3)
    motion: Literal["auto", "rigid", "articulated"] = "auto"


class RunConfig(StrictModel):
    model: str = "gpt-6-astra"
    iterations: int = Field(default=3, ge=1, le=8)
    target_score: Unit = 0.85
    max_output_tokens: int = Field(default=16000, ge=2000, le=64000)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    timeout_s: float = Field(default=300, gt=0, le=3600)


class PhysicsConfig(StrictModel):
    mass_kg: Positive | None = None
    mass_is_estimate: bool = False
    friction: Annotated[float, Field(ge=0, le=10)] = 0.5
    restitution: Unit = 0.0

    @model_validator(mode="after")
    def explicit_estimate(self):
        if self.mass_is_estimate and self.mass_kg is None:
            raise ValueError("An estimated mass requires an explicit mass_kg")
        return self
