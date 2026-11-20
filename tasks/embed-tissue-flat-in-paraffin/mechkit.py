"""mechkit: props for a yam_arena task file, written as plain-text USD: plain meshes, simple colliders, free bodies, and mechanisms on sprung or free joints.

A task file declares rigid primitives. `Part` subclasses `Box`, so that yam_arena's task-file checks accept a prop, and importing this module hooks
`yam_arena.scene` and `yam_arena.inspect_embodiment` as they load (nothing changes on disk), so that a `Part` spawns from USD text written at start-up.

    OBJECTS = [
        m.fixed("post", (0.3, 0.2, TOP), [("body", m.cylinder(0.025, 0.08, (0, 0, 0.04)), "steel")], [m.col_cylinder("c", 0.025, 0.08, (0, 0, 0.04))], LOOKS),
        m.free("can", (0.25, -0.1, TOP), [("body", m.cylinder(0.026, 0.12, (0, 0, 0.06)), "red")], [m.col_cylinder("c", 0.026, 0.12, (0, 0, 0.06))], LOOKS, mass=0.2),
        m.mechanism("tap", (0.3, 0.2, TOP), [m.knob("knob", (0, 0, 0.08), look="red")], LOOKS),
    ]

Rules these parts follow:
- Joint zero is the pose written in the file, whatever else is authored: model a mechanism at its start pose.
- A sprung joint rests at joint zero. Springs, damping and friction are set by the actuator from the `Link`, in SI units: N m/rad and N/m.
- Author nothing inside a collider, and no link touching a collider of another part: a light link on a joint blows up the solver. Whatever a link's pivot sits on goes in the mechanism's own `frame_parts`
  and `frame_colliders`, where self-collisions are off. Walls that stand clear of the links go in a `fixed` part of their own. A joint that leaves its limits prints a warning.
- Colliders stay simple and separate from what is seen: hidden, and they still collide.
- Check a moving part against the arms at their start pose, and with an arm left there.

Read a mechanism with `state(env, name).deg("knob")`, or `ctx.deg("tap", "knob")` in a demo. Register per-step logic with `on_step` and `on_reset`. Both run with the Isaac Lab environment.
Pure Python and NumPy at import; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from yam_arena.task_file import Box, ObjectState

ENV_PATH = "/World/envs/env_0"
_ROOT = "__ROOT__"  # stands for the prim's root in the text below, until a part is written

# -- meshes --------------------------------------------------------------------------------


@dataclass
class Mesh:
    v: np.ndarray  # [n, 3]
    f: np.ndarray  # [m, 3], counter-clockwise from outside

    def moved(self, t) -> "Mesh":
        return Mesh(self.v + np.asarray(t, float), self.f)

    def turned(self, deg: float, axis: str = "z") -> "Mesh":
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]], "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        return Mesh(self.v @ np.array(r).T, self.f)

    def aimed(self, outward: str) -> "Mesh":
        """Turn a mesh built along +z, so that +z points along `outward`: "+X", "-Y", ... ."""
        return {"+Z": lambda: self, "-Z": lambda: self.turned(180, "x"), "+X": lambda: self.turned(90, "y"), "-X": lambda: self.turned(-90, "y"),
                "+Y": lambda: self.turned(-90, "x"), "-Y": lambda: self.turned(90, "x")}[outward]()


def join(*meshes: Mesh) -> Mesh:
    vs, fs, n = [], [], 0
    for m in meshes:
        vs.append(m.v)
        fs.append(m.f + n)
        n += len(m.v)
    return Mesh(np.vstack(vs), np.vstack(fs))


def prism(outline, z0: float, z1: float) -> Mesh:
    """A closed solid from a counter-clockwise, star-shaped outline in xy, from z0 to z1."""
    outline = np.asarray(outline, float)
    n = len(outline)
    c = outline.mean(axis=0)
    v = np.vstack([np.c_[outline, np.full(n, z0)], np.c_[outline, np.full(n, z1)], [[c[0], c[1], z0], [c[0], c[1], z1]]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    return Mesh(v, np.array(f))


def box(size, center=(0.0, 0.0, 0.0)) -> Mesh:
    sx, sy, sz = size
    outline = np.array([[sx, -sy], [sx, sy], [-sx, sy], [-sx, -sy]], float) * 0.5
    return prism(outline, -sz / 2, sz / 2).moved(center)


def lathe(profile, segments: int = 32) -> Mesh:
    """Revolve a (radius, z) profile about z, from bottom to top; radius 0 closes the surface."""
    th = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    rings, verts = [], []
    for r, z in profile:
        if r == 0.0:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + segments)))
            verts += [(r * np.cos(t), r * np.sin(t), z) for t in th]
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) > 1:
            faces += [(a[0], b[(i + 1) % segments], b[i]) for i in range(segments)]
        elif len(b) == 1 and len(a) > 1:
            faces += [(a[i], a[(i + 1) % segments], b[0]) for i in range(segments)]
        elif len(a) > 1:
            for i in range(segments):
                j = (i + 1) % segments
                faces += [(a[i], a[j], b[j]), (a[i], b[j], b[i])]
    return Mesh(np.array(verts, float), np.array(faces))


def cylinder(radius: float, height: float, center=(0.0, 0.0, 0.0), axis: str = "z", segments: int = 32) -> Mesh:
    m = lathe([(0.0, -height / 2), (radius, -height / 2), (radius, height / 2), (0.0, height / 2)], segments)
    if axis == "x":
        m = m.turned(90.0, "y")
    elif axis == "y":
        m = m.turned(-90.0, "x")
    return m.moved(center)


def sphere(radius: float, center=(0.0, 0.0, 0.0), rings: int = 10, segments: int = 24) -> Mesh:
    a = np.linspace(-np.pi / 2, np.pi / 2, rings + 2)
    prof = [(0.0 if k in (0, rings + 1) else float(radius * np.cos(t)), float(radius * np.sin(t))) for k, t in enumerate(a)]
    return lathe(prof, segments).moved(center)


def tube(points, radius: float, sides: int = 12) -> Mesh:
    """A closed tube along a polyline."""
    p = np.asarray(points, float)
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = np.cross(d[0], [0.0, 0.0, 1.0] if abs(d[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    th = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    rings = []
    for k in range(len(p)):
        u = u - u.dot(d[k]) * d[k]
        u /= np.linalg.norm(u)
        w = np.cross(d[k], u)
        rings.append(p[k] + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), w)))
    v = np.vstack(rings + [p[0][None], p[-1][None]])
    f = []
    for k in range(len(p) - 1):
        for i in range(sides):
            j = (i + 1) % sides
            a, b, c, e = k * sides + i, k * sides + j, (k + 1) * sides + j, (k + 1) * sides + i
            f += [(a, b, c), (a, c, e)]
    first, last = len(p) * sides, len(p) * sides + 1
    for i in range(sides):
        j = (i + 1) % sides
        f += [(first, j, i), (last, (len(p) - 1) * sides + i, (len(p) - 1) * sides + j)]
    return Mesh(v, np.array(f))


def _corner_normals(v: np.ndarray, f: np.ndarray, crease_deg: float = 35.0) -> np.ndarray:
    """One normal per triangle corner: the mean over the faces round its vertex that lie within `crease_deg` of its own face."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    _, ids = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    around: dict[int, list[int]] = {}
    for i, tri in enumerate(f):
        for c in tri:
            around.setdefault(int(ids[c]), []).append(i)
    cos_crease = np.cos(np.radians(crease_deg))
    out = np.zeros((len(f), 3, 3))
    for i, tri in enumerate(f):
        for k, c in enumerate(tri):
            faces = around[int(ids[c])]
            n = fn[faces][fn[faces] @ fn[i] > cos_crease].sum(axis=0)
            out[i, k] = n / max(np.linalg.norm(n), 1e-12)
    return out


# -- materials -----------------------------------------------------------------------------


def rgb(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB (0..1) to the linear values a USD material takes."""

    def one(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (one(r), one(g), one(b))


@dataclass(frozen=True)
class Look:
    """A plain surface. `color` and `emissive` are linear: use `rgb`. `glass` is the renderer's thin-walled glass, with this colour."""

    color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    metallic: float = 0.0
    roughness: float = 0.6
    opacity: float = 1.0
    emissive: tuple[float, float, float] | None = None
    glass: tuple[float, float, float] | None = None


def _material(name: str, look: Look) -> str:
    base = f"/{_ROOT}/Looks/{name}"
    if look.glass is not None:
        c = look.glass
        return f"""
        def Material "{name}"
        {{
            token outputs:mdl:surface.connect = <{base}/Shader.outputs:out>
            token outputs:mdl:displacement.connect = <{base}/Shader.outputs:out>
            token outputs:mdl:volume.connect = <{base}/Shader.outputs:out>
            def Shader "Shader"
            {{
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniGlass.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniGlass"
                color3f inputs:glass_color = ({c[0]}, {c[1]}, {c[2]})
                float inputs:glass_ior = 1.1
                float inputs:frosting_roughness = 0
                bool inputs:thin_walled = 1
                token outputs:out
            }}
        }}"""
    c = look.color
    extra = f"\n                color3f inputs:emissiveColor = ({look.emissive[0]}, {look.emissive[1]}, {look.emissive[2]})" if look.emissive else ""
    if look.opacity < 1.0:
        extra += f"\n                float inputs:opacity = {look.opacity}"
    return f"""
        def Material "{name}"
        {{
            token outputs:surface.connect = <{base}/surface.outputs:surface>
            def Shader "surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({c[0]}, {c[1]}, {c[2]})
                float inputs:metallic = {look.metallic}
                float inputs:roughness = {look.roughness}{extra}
                token outputs:surface
            }}
        }}"""


# -- prims ---------------------------------------------------------------------------------


def _fmt(a: np.ndarray, digits: int = 5) -> str:
    return ", ".join("(" + ", ".join(f"{x:.{digits}f}" for x in row) + ")" for row in a)


def _mesh(name: str, mesh: Mesh, look: str, hidden: bool = False) -> str:
    v, f = mesh.v, mesh.f
    vis = '\n            token visibility = "invisible"' if hidden else ""
    return f"""
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{{vis}
            rel material:binding = </{_ROOT}/Looks/{look}>
            uniform bool doubleSided = 0
            int[] faceVertexCounts = [{", ".join("3" for _ in range(len(f)))}]
            int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
            point3f[] points = [{_fmt(v)}]
            normal3f[] normals = [{_fmt(_corner_normals(v, f).reshape(-1, 3), 4)}] (
                interpolation = "faceVarying"
            )
            uniform token subdivisionScheme = "none"
        }}"""


def _visual(parts) -> str:
    """`parts`: (prim name, Mesh, look name) or (prim name, Mesh, look name, hidden at the start)."""
    body = "".join(_mesh(p[0], p[1], p[2], len(p) > 3 and p[3]) for p in parts)
    return f"""
    def Xform "visual"
    {{{body}
    }}""" if parts else ""


_COL_HEAD = f"""(
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{_ROOT}/PhysicsMaterial>
        token visibility = "invisible"
        float physxCollision:contactOffset = 0.004
        float physxCollision:restOffset = 0"""


def col_box(name: str, size, center=(0.0, 0.0, 0.0)) -> str:
    return f"""
    def Cube "{name}" {_COL_HEAD}
        double size = 1
        double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})
        float3 xformOp:scale = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}"""


def col_cylinder(name: str, radius: float, height: float, center=(0.0, 0.0, 0.0)) -> str:
    """A cylinder collider about z."""
    return f"""
    def Cylinder "{name}" {_COL_HEAD}
        uniform token axis = "Z"
        double height = {height:.6f}
        double radius = {radius:.6f}
        double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""


def col_hull(name: str, mesh: Mesh) -> str:
    """The convex hull of a mesh, as a collider: the way to a collider that is not axis-aligned."""
    v, f = mesh.v, mesh.f
    head = _COL_HEAD.replace('"MaterialBindingAPI"', '"PhysicsMeshCollisionAPI", "MaterialBindingAPI"')
    return f"""
    def Mesh "{name}" {head}
        uniform token physics:approximation = "convexHull"
        int[] faceVertexCounts = [{", ".join("3" for _ in range(len(f)))}]
        int[] faceVertexIndices = [{", ".join(str(int(i)) for i in f.reshape(-1))}]
        point3f[] points = [{_fmt(v)}]
        uniform token subdivisionScheme = "none"
    }}"""


def _file(root: str, head_api: str, attrs: str, body: str, looks: dict[str, Look], friction: tuple[float, float]) -> str:
    api = f""" (
    prepend apiSchemas = [{head_api}]
)""" if head_api else ""
    mats = "".join(_material(n, k) for n, k in looks.items())
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{api}
{{{attrs}
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {friction[0]}
        float physics:dynamicFriction = {friction[1]}
        float physics:restitution = 0
    }}

    def Scope "Looks"
    {{{mats}
    }}
{body}
}}
"""
    return text.replace(_ROOT, root)


# -- mechanisms ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    """One moving part of a mechanism, on one joint. Meshes and colliders are in the part's frame, posed as at joint zero.

    `kind` is "hinge" (limits in degrees) or "slide" (limits in metres). `axis` is "X", "Y" or "Z" of the part's frame; `at` is a point on it.
    `spring` pulls back to joint zero (N m/rad or N/m): 0 leaves the joint free, held only by `friction` (N m or N) and `damping`. `parent` is another link's name; None hangs it on the frame.
    Links have no gravity unless `gravity`, in which case the centre of mass is `com`, or the mean of the mesh's points."""

    name: str
    kind: str
    axis: str
    at: tuple
    lower: float
    upper: float
    parts: tuple = ()
    colliders: tuple = ()
    mass: float = 0.05
    spring: float = 0.0
    damping: float = 0.2
    friction: float = 0.02
    effort: float = 50.0
    armature: float = 0.001
    gravity: bool = False
    com: tuple | None = None
    parent: str | None = None
    inertia: tuple | None = None  # principal inertia (kg m^2) along the part's x, y, z; None keeps the round guess below. Give it for slender parts turned about their own axis (README, 16)

    @property
    def joint(self) -> str:
        return f"{self.name}_joint"

    def gains(self) -> tuple:
        return (self.spring, self.damping, self.effort, self.friction, self.armature)


def knob(name: str, at, outward: str = "+Z", look: str = "knob", mark: str | None = None, hub_r: float = 0.016, hub_h: float = 0.01, wing=(0.06, 0.01, 0.025), lower: float = -170.0, upper: float = 170.0,
         friction: float = 0.03, damping: float = 0.05, spring: float = 0.0, mass: float = 0.05) -> Link:
    """A wing knob: a hub and a wing to pinch, standing out of `at` along `outward`, its wing's length along y at joint zero for "+Z". A gripper pinches the wing across its thickness and turns.
    `mark`, a look name, adds a light stripe along one half of the wing's top, to read the angle by."""
    hub = cylinder(hub_r, hub_h, (0, 0, hub_h / 2))
    bar = box((wing[1], wing[0], wing[2]), (0, 0, hub_h + wing[2] / 2))
    parts =[("hub", hub, look), ("wing", bar, look)]
    if mark:
        parts.append(("mark", box((wing[1] * 0.7, wing[0] * 0.45, 0.001), (0, wing[0] * 0.25, hub_h + wing[2] + 0.0005)), mark))
    a = np.asarray(at, float)
    placed = tuple((n, m.aimed(outward).moved(a), k) for n, m, k in parts)
    return Link(name, "hinge", outward[1], tuple(a), lower, upper, placed, (col_hull(f"{name}_hub_c", hub.aimed(outward).moved(a)), col_hull(f"{name}_wing_c", bar.aimed(outward).moved(a))),
                mass=mass, spring=spring, damping=damping, friction=friction)


def lever(name: str, pivot, along, axis: str = "Y", look: str = "lever", length: float = 0.12, radius: float = 0.008, ball_r: float = 0.016, lower: float = -30.0, upper: float = 30.0,
          spring: float = 0.0, damping: float = 0.05, friction: float = 0.01, mass: float = 0.08, gravity: bool = False) -> Link:
    """A bar from `pivot` along the direction `along` at joint zero, with a ball on its end, hinged about `axis` through `pivot`."""
    p = np.asarray(pivot, float)
    d = np.asarray(along, float)
    d = d / np.linalg.norm(d)
    bar, ball = tube([p, p + 0.5 * length * d, p + length * d], radius), sphere(ball_r, p + length * d)
    return Link(name, "hinge", axis, tuple(p), lower, upper, (("bar", bar, look), ("ball", ball, look)), (col_hull(f"{name}_bar_c", bar), col_hull(f"{name}_ball_c", ball)),
                mass=mass, spring=spring, damping=damping, friction=friction, gravity=gravity)


def button(name: str, at, outward: str = "+Z", look: str = "button", radius: float = 0.02, height: float = 0.015, travel: float = 0.008, spring: float = 300.0, damping: float = 2.0, mass: float = 0.02) -> Link:
    """A sprung button standing out of `at` along `outward`, which goes in by `travel`."""
    cap = cylinder(radius, height, (0, 0, height / 2)).aimed(outward).moved(at)
    lower, upper = (-travel, 0.0) if outward[0] == "+" else (0.0, travel)
    return Link(name, "slide", outward[1], tuple(float(v) for v in at), lower, upper, (("cap", cap, look),), (col_hull(f"{name}_c", cap),), mass=mass, spring=spring, damping=damping, friction=0.0)


def slider(name: str, axis: str, lower: float, upper: float, parts, colliders, at=(0.0, 0.0, 0.0), **kw) -> Link:
    """A part that slides along `axis`, from `lower` to `upper` metres. Free unless given a `spring`."""
    return Link(name, "slide", axis, tuple(float(v) for v in at), lower, upper, tuple(parts), tuple(colliders), **kw)


def hinge(name: str, axis: str, at, lower: float, upper: float, parts, colliders, **kw) -> Link:
    """A part that turns about `axis` through `at`, from `lower` to `upper` degrees: a door, a flap, a wheel. Free unless given a `spring`."""
    return Link(name, "hinge", axis, tuple(float(v) for v in at), lower, upper, tuple(parts), tuple(colliders), **kw)


def _link_usda(link: Link) -> str:
    pts = np.vstack([p[1].v for p in link.parts]) if link.parts else np.asarray([link.at], float)
    at = np.asarray(link.at, float)
    com = np.asarray(link.com, float) if link.com is not None else (pts.mean(axis=0) if link.gravity else at)
    reach = max(float(np.linalg.norm(pts - com, axis=1).max()), 0.01)
    inertia = max(link.mass * (0.5 * reach) ** 2, 1e-6)
    diag = (inertia,) * 3 if link.inertia is None else tuple(link.inertia)
    gravity = "" if link.gravity else "\n        bool physxRigidBody:disableGravity = 1"
    body = _visual(link.parts) + "".join(link.colliders)
    return f"""
    def Xform "{link.name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {link.mass:.5f}
        point3f physics:centerOfMass = ({com[0]:.5f}, {com[1]:.5f}, {com[2]:.5f})
        float3 physics:diagonalInertia = ({diag[0]:.3e}, {diag[1]:.3e}, {diag[2]:.3e})
        float physxRigidBody:maxDepenetrationVelocity = 1{gravity}
{body}
    }}"""


def _joint_usda(link: Link) -> str:
    kind, drive = ("PhysicsRevoluteJoint", "angular") if link.kind == "hinge" else ("PhysicsPrismaticJoint", "linear")
    x, y, z = link.at
    parent = "frame" if link.parent is None else link.parent
    limits = "" if not (math.isfinite(link.lower) and math.isfinite(link.upper)) else f"""
        float physics:lowerLimit = {link.lower:.6f}
        float physics:upperLimit = {link.upper:.6f}"""  # an infinite limit leaves the joint free: a wheel, a roller
    return f"""
    def {kind} "{link.joint}" (
        prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]
    )
    {{
        uniform token drive:{drive}:physics:type = "force"
        float drive:{drive}:physics:stiffness = 0
        float drive:{drive}:physics:damping = 0
        float drive:{drive}:physics:maxForce = {link.effort}
        rel physics:body0 = </{_ROOT}/{parent}>
        rel physics:body1 = </{_ROOT}/{link.name}>
        uniform token physics:axis = "{link.axis}"
        point3f physics:localPos0 = ({x:.6f}, {y:.6f}, {z:.6f})
        quatf physics:localRot0 = (1, 0, 0, 0)
        point3f physics:localPos1 = ({x:.6f}, {y:.6f}, {z:.6f})
        quatf physics:localRot1 = (1, 0, 0, 0)
{limits}
    }}"""


# -- parts of a task file ------------------------------------------------------------------


@dataclass(frozen=True)
class Part(Box):
    """A prop spawned from USD text. `pos` places the file's origin; `size` is nominal. Make one with `fixed`, `free` or `mechanism`."""

    role: str = "fixed"
    build: Callable[[], str] | None = None
    drives: tuple = ()
    start_joints: tuple = ()
    limits: tuple = ()  # (joint name, lower, upper), in radians or metres
    friction: tuple[float, float] = (1.2, 1.0)
    torsional_patch: float | None = None
    solver_iters: int = 32
    yaw_deg: float = 0.0
    start_quat: tuple | None = None  # xyzw: the starting rotation, if not only a yaw

    def _check(self) -> None:
        if self.role not in ("fixed", "free", "jointed"):
            raise ValueError(f"{self.name}: unknown role {self.role!r}")
        if self.role == "free" and self.mass is None:
            raise ValueError(f"{self.name}: a free part needs a mass")


def _root_name(name: str) -> str:
    return "P_" + name


def fixed(name: str, pos, parts, colliders=(), looks=None, yaw_deg: float = 0.0, friction=(1.0, 0.8)) -> Part:
    """A prop that does not move. `parts`: (prim name, Mesh, look name[, hidden]). `colliders`: from `col_box`, `col_cylinder` or `col_hull`."""
    return Part(name, pos=tuple(pos), size=(0.1, 0.1, 0.1), role="fixed", yaw_deg=yaw_deg,
                build=lambda: _file(_root_name(name), "", "", _visual(parts) + "".join(colliders), looks or {}, friction))


def free(name: str, pos, parts, colliders, looks, mass: float, friction=(0.9, 0.8), torsional_patch: float | None = 0.015, solver_iters: int = 32, yaw_deg: float = 0.0, start_quat: tuple | None = None,
         com=None) -> Part:
    """One rigid body, with `mass` in kg. Its origin is what `state` reports. `start_quat`, xyzw, starts it in any rotation, such as lying on its side.
    `com`, in the part's frame, sets the centre of mass; otherwise PhysX takes it from the colliders."""
    attrs = "" if com is None else f"""
    point3f physics:centerOfMass = ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})"""
    return Part(name, pos=tuple(pos), size=(0.1, 0.1, 0.1), mass=mass, role="free", friction=friction, torsional_patch=torsional_patch, solver_iters=solver_iters, yaw_deg=yaw_deg, start_quat=start_quat,
                build=lambda: _file(_root_name(name), '"PhysicsRigidBodyAPI", "PhysicsMassAPI"', attrs, _visual(parts) + "".join(colliders), looks, friction))


def mechanism(name: str, pos, links, looks, frame_parts=(), frame_colliders=(), yaw_deg: float = 0.0, friction=(1.2, 1.0), solver_iters: int = 32, start=(), free_mass: float | None = None, start_quat: tuple | None = None,
              frame_inertia: tuple = (0.01, 0.01, 0.01)) -> Part:
    """An articulation: a frame, with `frame_parts` and `frame_colliders`, and each `Link` on its joint. `start` gives (link name, joint value) to differ from joint zero at a reset.
    The frame is fixed in place, unless `free_mass` gives it a mass in kg: it then stands free on what is under it, and needs colliders. `start_quat`, xyzw, starts a free one in any rotation.
    `frame_inertia` is the frame's principal inertia along the part's x, y, z (kg m^2); the default suits a fixed frame, not a free slender one (README, 16)."""
    links = tuple(links)
    world_joint = "" if free_mass is not None else f"""
    def PhysicsFixedJoint "world_joint"
    {{
        rel physics:body1 = </{_ROOT}/frame>
    }}
"""
    frame_mass = 1.0 if free_mass is None else float(free_mass)

    def build() -> str:
        body = f"""
    def Xform "frame" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = {frame_mass}
        float3 physics:diagonalInertia = ({frame_inertia[0]:g}, {frame_inertia[1]:g}, {frame_inertia[2]:g})
{_visual(frame_parts)}{"".join(frame_colliders)}
    }}
{"".join(_link_usda(k) for k in links)}
{world_joint}{"".join(_joint_usda(k) for k in links)}"""
        attrs = f"""
    bool physxArticulation:enabledSelfCollisions = 0
    int physxArticulation:solverPositionIterationCount = {solver_iters}
    int physxArticulation:solverVelocityIterationCount = 1"""
        return _file(_root_name(name), '"PhysicsArticulationRootAPI", "PhysxArticulationAPI"', attrs, body, looks, friction)

    limits = tuple((k.joint, *(math.radians(v) if k.kind == "hinge" else v for v in (k.lower, k.upper))) for k in links if math.isfinite(k.lower) and math.isfinite(k.upper))
    return Part(name, pos=tuple(pos), size=(0.1, 0.1, 0.1), role="jointed", build=build, friction=friction, yaw_deg=yaw_deg, start_quat=start_quat, limits=limits,
                drives=tuple((k.joint, k.gains()) for k in links), start_joints=tuple((f"{n}_joint", v) for n, v in start))


@dataclass(frozen=True)
class PartState(ObjectState):
    """A part's origin pose and velocities, and its joint positions in radians or metres."""

    joints: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        return quat_matrix(self.quat_xyzw)

    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.rotation()[2, 2], -1.0, 1.0))))

    def joint(self, link: str) -> float:
        """A link's joint position: radians or metres."""
        return self.joints[f"{link}_joint"]

    def deg(self, link: str) -> float:
        return math.degrees(self.joint(link))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """World points in this part's frame."""
        return (np.atleast_2d(points) - self.pos) @ self.rotation()

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return np.atleast_2d(points) @ self.rotation().T + self.pos


def quat_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


_SPECS: dict[str, Part] = {}


def _usd_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), f"mechkit_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    return path


def _arena_object(spec: Part):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    usd = os.path.join(_usd_dir(), f"{spec.name}.usda")
    with open(usd, "w") as f:
        f.write(spec.build())
    h = math.radians(spec.yaw_deg) / 2
    pose = Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(float(v) for v in spec.start_quat) if spec.start_quat else (0.0, 0.0, math.sin(h), math.cos(h)))
    if spec.role == "fixed":
        return Object(name=spec.name, object_type=ObjectType.BASE, usd_path=usd, initial_pose=pose)
    if spec.role == "jointed":
        drives = dict(spec.drives)

        class Jointed(Object):  # Arena's own articulation object has neither actuators nor initial joint positions
            def _generate_articulation_cfg(self):
                actuators = {n: ImplicitActuatorCfg(joint_names_expr=[n], stiffness=g[0], damping=g[1], joint_effort_limit=g[2], friction=g[3], armature=g[4]) for n, g in drives.items()}
                start = {n: 0.0 for n in drives} | dict(spec.start_joints)
                cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self._get_spawn_cfg(activate_contact_sensors=False),
                                      init_state=ArticulationCfg.InitialStateCfg(joint_pos=start), actuators=actuators)
                return self._add_initial_pose_to_cfg(cfg)

        return Jointed(name=spec.name, object_type=ObjectType.ARTICULATION, usd_path=usd, initial_pose=pose)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
    collision: dict[str, Any] = {"contact_offset": 0.005, "rest_offset": 0.0}
    if spec.torsional_patch:
        collision.update(torsional_patch_radius=spec.torsional_patch, min_torsional_patch_radius=0.8 * spec.torsional_patch)
    addon = {"physics_material": material, "collision_props": sim_utils.CollisionPropertiesCfg(**collision),
             "rigid_props": sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=spec.solver_iters),
             "mass_props": sim_utils.MassPropertiesCfg(mass=spec.mass)}
    obj = Object(name=spec.name, object_type=ObjectType.RIGID, usd_path=usd, spawn_cfg_addon=addon)
    obj.set_initial_pose(pose)
    return obj


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(environment, spec: Part) -> PartState:
    h = math.radians(spec.yaw_deg) / 2
    if spec.role == "fixed":
        pos = np.asarray(spec.pos, float)
        return PartState(pos=pos, quat_xyzw=np.array([0.0, 0.0, math.sin(h), math.cos(h)]), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pos, high=pos)
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    if spec.role == "jointed" and "frame" in asset.body_names and asset.body_names[0] != "frame":
        # A free mechanism's root link is PhysX's choice, and may be a link: report its frame, which is what the task authored.
        i = asset.body_names.index("frame")
        pose = _np(asset.data.body_link_pose_w)[0, i]
        vel = _np(asset.data.body_link_vel_w)[0, i]
    joints = {}
    if spec.role == "jointed":
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
    pos = pose[:3] - origin
    return PartState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos, high=pos, joints=joints)


def state(environment, name: str) -> PartState:
    """A part's state now."""
    return _state(environment, _SPECS[name])


def fingertips(environment) -> np.ndarray:
    """Points on both arms' fingers (station frame): tip and 3 cm up it, at the middle and both edges of each finger, opened as the fingers are now. Shape (24, 3): the left arm's 12, then the right's."""
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        i = robot.find_bodies("gripper")[0][0]
        pose = _np(robot.data.body_link_pose_w)[0, i]
        R = quat_matrix(pose[3:7])
        opening = abs(float(_np(robot.data.joint_pos)[0, robot.find_joints("joint7")[0][0]]))
        for sx in (-1.0, 1.0):
            for dy in (-0.018, 0.0, 0.018):
                for dz in (0.147, 0.117):
                    out.append(pose[:3] - origin + R @ np.array([sx * (opening + 0.0045), dy, dz]))
    return np.array(out)


# -- changing a mechanism while it runs -------------------------------------------------------

_TOUCHED: set[tuple[str, str]] = set()
_VISIBLE: dict[str, bool] = {}


def _tensor(asset, value: float):
    import torch

    return torch.tensor([[float(value)]], device=asset.device)


def _joint(environment, part: str, link: str):
    asset = environment.unwrapped.scene[part]
    return asset, asset.find_joints(f"{link}_joint")[0][0]


def set_target(environment, part: str, link: str, value: float) -> None:
    """Move a joint's spring to rest at `value`, in radians or metres: a motor, a solenoid, a task-driven part."""
    asset, j = _joint(environment, part, link)
    _TOUCHED.add((part, link))
    asset.set_joint_position_target(_tensor(asset, value), joint_ids=[j])


def lock(environment, part: str, link: str, stiffness: float = 50000.0, effort: float = 2000.0) -> None:
    """Hold a joint where it is now, as a latch does, against any force a gripper can apply."""
    asset, j = _joint(environment, part, link)
    _TOUCHED.add((part, link))
    now = float(_np(asset.data.joint_pos)[0, j])
    asset.write_joint_effort_limit_to_sim(_tensor(asset, effort), joint_ids=[j])
    asset.write_joint_stiffness_to_sim(_tensor(asset, stiffness), joint_ids=[j])
    asset.set_joint_position_target(_tensor(asset, now), joint_ids=[j])


def unlock(environment, part: str, link: str) -> None:
    """Give a locked joint back its own spring and effort."""
    asset, j = _joint(environment, part, link)
    gains = dict(_SPECS[part].drives)[f"{link}_joint"]
    asset.write_joint_effort_limit_to_sim(_tensor(asset, gains[2]), joint_ids=[j])
    asset.write_joint_stiffness_to_sim(_tensor(asset, gains[0]), joint_ids=[j])
    asset.set_joint_position_target(_tensor(asset, 0.0), joint_ids=[j])


_PUSHED: set[tuple[str, int]] = set()


def push(environment, part: str, force, at=None, body: str | None = None) -> None:
    """Hold `force` (N, station frame) on a part's body at station point `at` (default: the body's centre of mass), at every
    physics step until changed: the drag of a braked tape roll, a cable's pull. `push(env, part, (0, 0, 0))` stops it; a reset stops all."""
    import torch

    base = environment.unwrapped
    asset = base.scene[part]
    origin = _np(base.scene.env_origins[0])
    if body is None and "frame" in getattr(asset, "body_names", ()):
        body = "frame"  # a mechanism's frame, wherever PhysX put its root
    i = 0 if body is None else int(asset.find_bodies(body)[0][0])
    f = torch.tensor([[[float(v) for v in force]]], device=asset.device)
    pos = None if at is None else torch.tensor([[[float(v) for v in np.asarray(at, float) + origin]]], device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), positions=pos, body_ids=[i], is_global=True)
    _PUSHED.add((part, i))


def _restore_touched(environment) -> None:
    """Give changed joints back their own gains and targets, and stop held forces, before a reset's settling steps can act on the old ones."""
    for part, link in list(_TOUCHED):
        unlock(environment, part, link)
    _TOUCHED.clear()
    for part, i in list(_PUSHED):
        asset = environment.unwrapped.scene[part]
        asset.permanent_wrench_composer.reset()
    _PUSHED.clear()


def set_visible(path: str, visible: bool) -> None:
    """Show or hide a prim under `ENV_PATH` (only when that changes), for anything seen that follows the task's state. A path that is not there raises."""
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    if _VISIBLE.get(path) == visible:
        return
    _VISIBLE[path] = visible
    prim = sim_utils.get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"no prim at {path}")
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


# -- hooking yam_arena -----------------------------------------------------------------------

_STEP: list = []
_RESET: list = []


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled."""
    _RESET.append(fn)
    return fn


_SEED: list = [None]


def run_seed() -> int | None:
    """The seed of the latest reset. `inspect-robots run` derives it from its `--seed`; demokit passes its `--seed` as it is, or None if not given."""
    return _SEED[0]


_WARNED: set[str] = set()


def _check_limits(environment) -> None:
    """A joint well outside its limits means that the solver has blown up, and every later number is noise: say so at once."""
    for spec in _SPECS.values():
        if spec.role != "jointed":
            continue
        joints = _state(environment, spec).joints
        for joint, low, high in spec.limits:
            margin = 0.2 * (high - low) + 0.01
            if not low - margin <= joints[joint] <= high + margin and joint not in _WARNED:
                _WARNED.add(joint)
                print(f"[mechkit] BLOWN UP: {spec.name} {joint} is at {joints[joint]:.3f}, limits {low:.3f} to {high:.3f}. Look for a link authored inside or touching a collider of another part.", flush=True)


def _patch_scene(scene) -> None:
    if getattr(scene, "_mechkit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._mechkit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_mechkit", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def task_step_hook(self, *a, **k):
        for fn in _STEP:
            fn(self._env)
        _check_limits(self._env)
        if os.environ.get("TASK_STEP_FILE"):  # the simulated time, for anything following the run from outside
            with open(os.environ["TASK_STEP_FILE"], "w") as f:
                f.write(f"{self._t:.3f}\n")
        return task_step(self, *a, **k)

    def reset_hook(self, *a, **k):
        _SEED[0] = k.get("seed")
        if self._env is not None:
            _restore_touched(self._env)
        obs = reset(self, *a, **k)
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = task_step_hook, reset_hook
    cls._mechkit = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena's scene and embodiment modules as they are imported: they need Isaac Sim, which starts after the task file is loaded."""

    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now for modules already imported, and on import for the rest. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
