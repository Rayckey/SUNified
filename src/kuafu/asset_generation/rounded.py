"""Rounded rectangular container: a single closed shell and 17 convex proxies."""

import numpy as np
import trimesh

from .schema import RoundedContainer


def rounded_rectangle(width: float, depth: float, radius: float, segments: int) -> np.ndarray:
    """CCW perimeter; include both ends of each corner arc so straight walls stay planar."""
    corners = [
        (width / 2 - radius, depth / 2 - radius, 0),
        (-width / 2 + radius, depth / 2 - radius, np.pi / 2),
        (-width / 2 + radius, -depth / 2 + radius, np.pi),
        (width / 2 - radius, -depth / 2 + radius, 3 * np.pi / 2),
    ]
    return np.array(
        [
            [cx + radius * np.cos(a), cy + radius * np.sin(a)]
            for cx, cy, start in corners
            for a in np.linspace(start, start + np.pi / 2, segments + 1)
        ]
    )


def _ring(g: RoundedContainer, offset: float, z: float, segments: int) -> np.ndarray:
    perimeter = rounded_rectangle(
        g.size.x - 2 * offset, g.size.y - 2 * offset, g.corner_radius - offset, segments
    )
    return np.column_stack([perimeter, np.full(len(perimeter), z)])


def _skin(rings: list[np.ndarray]) -> trimesh.Trimesh:
    n = len(rings[0])
    vertices = np.vstack([*rings, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    faces = []
    for layer in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = layer * n + i, layer * n + j, (layer + 1) * n + j, (layer + 1) * n + i
            faces.extend([[a, b, c], [a, c, d]])
    bottom_center, floor_center = len(vertices) - 2, len(vertices) - 1
    last = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[bottom_center, j, i], [floor_center, last + i, last + j]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def rounded_container(g: RoundedContainer) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    """Nominal dimensions describe the outer rim before its small edge rounding."""
    h, t, e = g.size.z, g.wall_thickness, g.edge_radius
    lip, floor_fillet = t / 2, min(t, e)
    profile = []

    def add(offset, z):
        profile.append((offset + g.bottom_inset * (1 - z / h), z))

    # Outer base quarter-circle, then the long outer wall.
    for a in np.linspace(0, np.pi / 2, 9):
        add(e * (1 - np.sin(a)), e * (1 - np.cos(a)))
    # Traverse the rounded rim from the outside to the inside.
    for a in np.linspace(0, np.pi, 17):
        add(lip * (1 - np.cos(a)), h - lip + lip * np.sin(a))
    # Inner wall and its transition onto the closed floor.
    for a in np.linspace(0, np.pi / 2, 9):
        add(t + floor_fillet * (1 - np.cos(a)), g.base_thickness + floor_fillet * (1 - np.sin(a)))
    visual = _skin([_ring(g, offset, z, 16) for offset, z in profile])

    # Low-poly compound shape. Never convexify the complete hollow object.
    bottom_outer = _ring(g, g.bottom_inset * (1 - g.base_thickness / h), g.base_thickness, 3)
    bottom_inner = _ring(g, g.bottom_inset * (1 - g.base_thickness / h) + t, g.base_thickness, 3)
    top_outer, top_inner = _ring(g, 0, h, 3), _ring(g, t, h, 3)
    base_bottom = _ring(g, g.bottom_inset + e, 0, 3)
    collisions = [trimesh.convex.convex_hull(np.vstack([base_bottom, bottom_outer]))]
    for i in range(len(top_outer)):
        j = (i + 1) % len(top_outer)
        vertices = np.vstack([ring[[i, j]] for ring in (bottom_outer, bottom_inner, top_outer, top_inner)])
        collisions.append(trimesh.convex.convex_hull(vertices))
    return visual, collisions
