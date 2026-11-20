"""Articulated, multi-colour parts for yam_arena task files: rigid parts built from boxes, cylinders, convex hulls
and drawn-only meshes, with hinges (revolute joints) and spring sliders (prismatic joints). Copy this file next to
your task file and import it there:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import parts  # noqa: E402

This is the parts kit of the accepted task put-the-tongs-away-in-the-cutlery-tray, extended with turned boxes,
convex hulls (:func:`hull`), drawn-only meshes (:func:`mesh`), shapes that collide without being drawn, and a
friction per shape.

A :class:`Part` goes in ``OBJECTS`` like a Box. It is a rigid body made of shapes, each with its own colour, given in
the part's frame (origin at ``pos``, turned by ``yaw``). ``mass=None`` fixes the part in place; a mass makes it loose.
Each :class:`Hinge` or :class:`Slider` in ``joints`` adds one moving link, with its own shapes and mass, jointed to
the part's body. Shapes are given where they are at joint position 0, so the pose you write down is the starting
pose. A joint with ``spring > 0`` is pulled back to 0.

success()/score() receive a :class:`PartState` for each part: ``pos``/``quat_xyzw`` are the part's body origin and
orientation (not a bounding-box centre), ``low``/``high`` the world box around all its shapes, ``joints`` each
joint's position (rad or m) and ``links`` each moving link's pose. ``on_step``/``on_reset`` register callbacks that
run in the simulator after every control step / reset with a :class:`Sim` handle (states, joint targets).

Importing this file is pure Python (NumPy and ``yam_arena.task_file``), so the task still loads offline. Isaac is
imported only once the embodiment builds its scene: an import hook then patches ``yam_arena.scene`` (spawning and
state reading) and the ``yam_arena`` embodiment's step and reset. Nothing in yam_arena is modified on disk.
"""

from __future__ import annotations

import importlib.abc
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import GREY, ObjectState

FRICTION = (1.2, 1.0)  # static, dynamic: yam_arena's grippy material
CONTACT_OFFSET, REST_OFFSET = 0.005, 0.0  # as yam_arena's free objects
SOLVER_ITERATIONS = (32, 4)  # articulation position, velocity iterations
MAX_DEPENETRATION = 1.0  # m/s: caps how hard PhysX pushes overlapping bodies apart
IDENTITY = (0.0, 0.0, 0.0, 1.0)


# -- geometry (NumPy) -------------------------------------------------------------------------------------------


def _rot(axis, angle: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(m) -> np.ndarray:
    """xyzw, w >= 0."""
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _x_to(axis) -> np.ndarray:
    """The rotation that turns +x onto ``axis`` (a USD joint's axis is its frame's x)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c = float(np.clip(a[0], -1.0, 1.0))
    v = np.cross([1.0, 0.0, 0.0], a)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else _rot((0, 0, 1), math.pi)
    return _rot(v, math.acos(c))


def turn(axis, angle: float) -> tuple[float, float, float, float]:
    """The quaternion (xyzw) of a turn by ``angle`` (rad) about ``axis``, for a shape's ``rot``."""
    return tuple(float(v) for v in _matrix_to_quat(_rot(axis, angle)))


# -- shapes, joints, parts (pure Python: what a task file declares) ----------------------------------------------


@dataclass(frozen=True)
class Shape:
    kind: str  # "box", "cylinder", "hull" (convex collider from points) or "mesh" (drawn only)
    size: tuple[float, float, float]  # box: edge lengths; cylinder: (radius, height, 0); hull, mesh: unused
    at: tuple[float, float, float]  # centre (hull, mesh: origin of its points) in the part frame, at joint position 0
    color: tuple[float, float, float] = GREY  # sRGB 0..1
    axis: str = "z"  # cylinder axis in the part frame: "x", "y" or "z"
    rot: tuple[float, float, float, float] = IDENTITY  # box, hull, mesh: turned by this quaternion (xyzw)
    points: tuple[tuple[float, float, float], ...] = ()  # hull, mesh: vertices in the shape's frame (m)
    faces: tuple[tuple[int, int, int], ...] = ()  # hull, mesh: triangles, counter-clockwise from outside
    normals: tuple[tuple[float, float, float], ...] = ()  # mesh: one per vertex (smooth shading); none: flat
    collide: bool = True  # False: drawn, nothing collides with it
    visible: bool = True  # False: collides, not drawn
    friction: tuple[float, float] | None = None  # static, dynamic; None: the part's

    @property
    def rotation(self) -> np.ndarray:
        return _quat_to_matrix(self.rot)

    @property
    def half(self) -> np.ndarray:
        """Half edge lengths of the box around the shape, along the part's axes, about ``at`` (a hull's or mesh's
        box is about the middle of its points, see :attr:`middle`)."""
        if self.kind == "box":
            return np.abs(self.rotation) @ (np.asarray(self.size, float) / 2)
        if self.kind == "cylinder":
            r, h = self.size[0], self.size[1]
            return np.array([h / 2 if self.axis == a else r for a in "xyz"])
        p = self.vertices
        return (p.max(axis=0) - p.min(axis=0)) / 2

    @property
    def vertices(self) -> np.ndarray:
        """A hull's or mesh's vertices in the part frame at joint position 0."""
        return np.asarray(self.points, float) @ self.rotation.T + np.asarray(self.at, float)

    @property
    def middle(self) -> np.ndarray:
        """The middle of the box around the shape, in the part frame."""
        if self.kind in ("box", "cylinder"):
            return np.asarray(self.at, float)
        p = self.vertices
        return (p.max(axis=0) + p.min(axis=0)) / 2


def box(size, at=(0.0, 0.0, 0.0), color=GREY, rot=IDENTITY, collide=True, visible=True, friction=None) -> Shape:
    return Shape("box", tuple(float(s) for s in size), tuple(float(a) for a in at), tuple(color),
                 rot=tuple(float(v) for v in rot), collide=collide, visible=visible, friction=friction)


def cylinder(radius, height, at=(0.0, 0.0, 0.0), color=GREY, axis="z", collide=True, visible=True,
             friction=None) -> Shape:
    if axis not in ("x", "y", "z"):
        raise ValueError(f"cylinder axis must be 'x', 'y' or 'z', got {axis!r}")
    return Shape("cylinder", (float(radius), float(height), 0.0), tuple(float(a) for a in at), tuple(color), axis,
                 collide=collide, visible=visible, friction=friction)


def _tuples(a, kind=float) -> tuple:
    return tuple(tuple(kind(v) for v in row) for row in np.asarray(a))


def hull(points, faces=(), at=(0.0, 0.0, 0.0), color=GREY, rot=IDENTITY, visible=False, friction=None) -> Shape:
    """A convex collider: the convex hull of ``points`` (at most 64). ``faces`` (triangles) are needed only to draw
    it; by default it is not drawn and a :func:`mesh` gives the look."""
    if not 4 <= len(points) <= 64:
        raise ValueError(f"a hull needs 4 to 64 points, got {len(points)}")
    return Shape("hull", (0.0, 0.0, 0.0), tuple(float(a) for a in at), tuple(color), rot=tuple(float(v) for v in rot),
                 points=_tuples(points), faces=_tuples(faces, int) if len(faces) else (), visible=visible,
                 friction=friction)


def mesh(points, faces, normals=(), at=(0.0, 0.0, 0.0), color=GREY, rot=IDENTITY) -> Shape:
    """A drawn-only triangle mesh (nothing collides with it): the look of a body whose colliders are hulls or boxes."""
    return Shape("mesh", (0.0, 0.0, 0.0), tuple(float(a) for a in at), tuple(color), rot=tuple(float(v) for v in rot),
                 points=_tuples(points), faces=_tuples(faces, int), normals=_tuples(normals) if len(normals) else (),
                 collide=False)


@dataclass(frozen=True)
class Hinge:
    """A revolute joint: the link (``shapes``, ``mass``) turns about ``axis`` (part frame) through ``pivot``.
    Position 0 is the pose the shapes are given in; ``lower``/``upper`` (rad) bound it. ``spring`` (N m/rad) pulls it
    back to 0 (or to a target set with :meth:`Sim.set_target`); ``damping`` (N m s/rad) resists turning."""

    name: str
    shapes: tuple[Shape, ...]
    pivot: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float
    mass: float = 0.1
    spring: float = 0.0
    damping: float = 0.01
    max_effort: float = 50.0  # N m the spring/drive may exert

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.pivot, float)

    def motion(self, q: float) -> tuple[np.ndarray, np.ndarray]:
        """(rotation, origin) of the link frame in the part frame at position q."""
        return _rot(self.axis, q), self.origin


@dataclass(frozen=True)
class Slider:
    """A prismatic joint: the link slides along ``axis`` (part frame) from 0 to ``travel`` (m). ``spring`` (N/m)
    pushes it back to 0 (or to a set target); ``damping`` (N s/m) resists motion."""

    name: str
    shapes: tuple[Shape, ...]
    axis: tuple[float, float, float]
    travel: float
    mass: float = 0.05
    spring: float = 0.0
    damping: float = 1.0
    max_effort: float = 200.0  # N

    @property
    def lower(self) -> float:
        return 0.0

    @property
    def upper(self) -> float:
        return float(self.travel)

    @property
    def origin(self) -> np.ndarray:
        return np.mean([s.at for s in self.shapes], axis=0)

    def motion(self, q: float) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(self.axis, float) / np.linalg.norm(self.axis)
        return np.eye(3), self.origin + q * a


Joint = Hinge | Slider


@dataclass(frozen=True)
class PartState(ObjectState):
    """A part after a control step (world frame). ``pos`` is the body origin, not the bounding-box centre."""

    joints: dict[str, float] = field(default_factory=dict)  # rad (hinge) or m (slider)
    links: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # joint name -> (pos, quat_xyzw)


@dataclass(frozen=True)
class Part(task_file.Box):
    """A rigid part made of ``shapes`` (part frame; origin at ``pos``, turned by ``yaw``), with optional ``joints``.
    ``mass=None``: fixed in place. ``size`` and ``color`` are derived (the starting box around every shape, the first
    shape's colour) so offline tools that read a Box still see something sensible."""

    shapes: tuple[Shape, ...] = ()
    joints: tuple[Joint, ...] = ()
    friction: tuple[float, float] = FRICTION

    def __post_init__(self):
        object.__setattr__(self, "shapes", tuple(self.shapes))
        object.__setattr__(self, "joints", tuple(self.joints))
        if self.shapes:
            object.__setattr__(self, "color", self.shapes[0].color)
        lo, hi = self._local_box()
        object.__setattr__(self, "size", tuple(float(v) for v in hi - lo))

    def _check(self) -> None:
        if not self.shapes:
            raise ValueError(f"{self.name}: a Part needs at least one shape")
        super()._check()
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names) or "base" in names or not all(n.isidentifier() for n in names):
            raise ValueError(f"{self.name}: joint names must be unique identifiers other than 'base', got {names}")
        for j in self.joints:
            if not j.shapes or not j.mass > 0 or not j.lower <= 0.0 <= j.upper or j.lower == j.upper:
                raise ValueError(f"{self.name}.{j.name}: needs shapes, a positive mass and limits around 0")
            if j.spring < 0 or j.damping < 0 or not np.linalg.norm(j.axis) > 0:
                raise ValueError(f"{self.name}.{j.name}: spring and damping must be >= 0 and the axis non-zero")
        for _, _, shapes in self.links:
            if not any(s.collide for s in shapes):
                raise ValueError(f"{self.name}: every body needs at least one shape that collides")
        for s in self.shapes + tuple(s for j in self.joints for s in j.shapes):
            if any(not 0.0 <= c <= 1.0 for c in s.color) or any(v <= 0 for v in s.half):
                raise ValueError(f"{self.name}: {s.kind} at {s.at} needs positive sizes and an sRGB colour in 0..1")
            if s.kind in ("hull", "mesh") and len(s.faces) and np.max(s.faces) >= len(s.points):
                raise ValueError(f"{self.name}: a {s.kind}'s faces index past its points")

    @property
    def rotation(self) -> np.ndarray:
        return _rot((0, 0, 1), self.yaw)

    @property
    def links(self) -> tuple[tuple[str, np.ndarray, tuple[Shape, ...]], ...]:
        """(name, origin in the part frame, shapes) for the body ("base") and each joint's link."""
        return (("base", np.zeros(3), self.shapes), *((j.name, j.origin, j.shapes) for j in self.joints))

    def _local_box(self):
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        pts = [s.middle + corners * s.half for _, _, shapes in self.links for s in shapes]
        pts = np.concatenate(pts) if pts else np.zeros((1, 3))
        return pts.min(axis=0), pts.max(axis=0)

    def world_shapes(self, link_poses: dict[str, tuple[np.ndarray, np.ndarray]], colliders_only: bool = True):
        """(link, shape, origin, rotation) of every shape of the links in ``link_poses`` (world pos, rotation):
        ``origin`` is where the shape's ``at`` is and ``rotation`` the shape frame's world rotation."""
        out = []
        for name, origin, shapes in self.links:
            if name not in link_poses:
                continue
            p, r = link_poses[name]
            for s in shapes:
                if colliders_only and not s.collide:
                    continue
                out.append((name, s, p + r @ (np.asarray(s.at) - origin), r @ s.rotation))
        return out

    def world_boxes(self, link_poses: dict[str, tuple[np.ndarray, np.ndarray]]):
        """(link, centre, rotation, size) of every box of the links in ``link_poses``, and the box around every other
        colliding shape, turned with its link."""
        out = []
        for name, origin, shapes in self.links:
            if name not in link_poses:
                continue
            p, r = link_poses[name]
            for s in shapes:
                if not s.collide:
                    continue
                if s.kind == "box":
                    out.append((name, p + r @ (np.asarray(s.at) - origin), r @ s.rotation, np.asarray(s.size, float)))
                else:
                    out.append((name, p + r @ (s.middle - origin), r, 2 * s.half))
        return out

    def link_poses(self, joints=None, pos=None, quat_xyzw=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Each link's world (pos, rotation) with the body at ``pos``/``quat_xyzw`` (default: where it starts) and
        the given joint positions (default 0), from the part's own kinematics."""
        q = {j.name: 0.0 for j in self.joints} | dict(joints or {})
        unknown = sorted(set(q) - {j.name for j in self.joints})
        if unknown:
            raise KeyError(f"{self.name} has no joints {unknown}")
        p = np.asarray(self.pos if pos is None else pos, float)
        r = self.rotation if quat_xyzw is None else _quat_to_matrix(quat_xyzw)
        poses = {"base": (p, r)}
        for j in self.joints:
            jr, jo = j.motion(q[j.name])
            poses[j.name] = (p + r @ jo, r @ jr)
        return poses

    def state_poses(self, state) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Each link's world (pos, rotation) from a :class:`PartState` (or a plain ObjectState of a fixed part)."""
        poses = {"base": (np.asarray(state.pos, float), _quat_to_matrix(state.quat_xyzw))}
        for name, (p, q) in getattr(state, "links", {}).items():
            poses[name] = (np.asarray(p, float), _quat_to_matrix(q))
        return poses

    def make_state(self, link_poses, joints, lin_vel=None, ang_vel=None) -> PartState:
        """A PartState from each link's world (pos, rotation matrix) and the joint positions."""
        boxes = self.world_boxes(link_poses)
        corners = np.concatenate([c + (np.abs(r) @ (s / 2)) * [[-1], [1]] for _, c, r, s in boxes])
        p, r = link_poses["base"]
        return PartState(
            pos=np.asarray(p, float),
            quat_xyzw=_matrix_to_quat(r),
            lin_vel=np.zeros(3) if lin_vel is None else np.asarray(lin_vel, float),
            ang_vel=np.zeros(3) if ang_vel is None else np.asarray(ang_vel, float),
            low=corners.min(axis=0),
            high=corners.max(axis=0),
            joints={k: float(v) for k, v in joints.items()},
            links={k: (np.asarray(v[0], float), _matrix_to_quat(v[1])) for k, v in link_poses.items() if k != "base"},
        )

    def state(self, joints=None, pos=None, quat_xyzw=None, lin_vel=None) -> PartState:
        """The state the simulator would report with the body at ``pos``/``quat_xyzw`` (default: where it starts) and
        the given joint positions (default 0), from the part's own kinematics. For offline checks and near misses."""
        q = {j.name: 0.0 for j in self.joints} | dict(joints or {})
        return self.make_state(self.link_poses(q, pos, quat_xyzw), q, lin_vel)


# -- callbacks ---------------------------------------------------------------------------------------------------

_ON_STEP: list[Callable] = []
_ON_RESET: list[Callable] = []


def on_step(fn: Callable) -> Callable:
    """Decorator: ``fn(sim)`` runs after every control step, before success()/score() read the states."""
    _ON_STEP.append(fn)
    return fn


def on_reset(fn: Callable) -> Callable:
    """Decorator: ``fn(sim)`` runs after every reset (after the embodiment's settle steps)."""
    _ON_RESET.append(fn)
    return fn


class Sim:
    """What a callback gets: the running embodiment, its states and the joints' drive targets."""

    def __init__(self, embodiment):
        self.embodiment = embodiment
        self.env = embodiment._env
        self.t = embodiment._t  # seconds since the episode began

    def states(self) -> dict[str, ObjectState]:
        from yam_arena.scene import object_states

        return object_states(self.env, self.embodiment.task.names)

    def set_target(self, part: str, joint: str, value: float) -> None:
        """Move the spring's rest point of ``part``'s ``joint`` (rad or m); it needs ``spring > 0`` to act."""
        _set_target(self.env.unwrapped, part, joint, value)

    def robot_links(self, names=None) -> dict[str, dict[str, np.ndarray]]:
        """Each arm's link poses as 4x4 matrices in the scene frame: {"left": {link: pose}, "right": {...}}."""
        base = self.env.unwrapped
        origin = _np(base.scene.env_origins[0])
        out = {}
        for side, key in (("left", "left_robot"), ("right", "right_robot")):
            asset = base.scene[key]
            poses = _np(asset.data.body_link_pose_w)[0]
            out[side] = {}
            for i, n in enumerate(asset.body_names):
                if names is not None and n not in names:
                    continue
                m = np.eye(4)
                m[:3, :3] = _quat_to_matrix(poses[i, 3:])
                m[:3, 3] = poses[i, :3] - origin
                out[side][n] = m
        return out


# -- runtime: Isaac Lab / Arena (imported only inside the simulator) ---------------------------------------------

_SPECS: dict[str, Part] = {}  # parts in the env being built or run, by name


def _usd_quat(m):
    from pxr import Gf

    x, y, z, w = _matrix_to_quat(m)
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _define_shape(stage, path: str, s: Shape, origin):
    """Write one shape's geometry at ``path`` (pxr only): a Cube, a Cylinder or a Mesh, placed in its link's frame."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    at = Gf.Vec3d(*(float(v) for v in np.asarray(s.at) - origin))
    turned = tuple(s.rot) != IDENTITY
    if s.kind == "box":
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(at)
        if turned:
            geom.AddOrientOp().Set(_usd_quat(s.rotation))
        geom.AddScaleOp().Set(Gf.Vec3f(*s.size))
    elif s.kind == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateRadiusAttr(s.size[0])
        geom.CreateHeightAttr(s.size[1])
        geom.CreateAxisAttr(s.axis.upper())
        geom.AddTranslateOp().Set(at)
    else:
        geom = UsdGeom.Mesh.Define(stage, path)
        points = np.asarray(s.points, np.float32)
        faces = np.asarray(s.faces, np.int32).reshape(-1, 3)
        geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        geom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
        geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if len(s.normals):
            geom.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(s.normals, np.float32)))
            geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        geom.AddTranslateOp().Set(at)
        if turned:
            geom.AddOrientOp().Set(_usd_quat(s.rotation))
    prim = geom.GetPrim()
    if s.collide:
        UsdPhysics.CollisionAPI.Apply(prim)
        if s.kind == "hull":
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    if not s.visible:
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    return prim


def _author(part: Part, root_path: str, translation, orientation):
    """Write the part to the stage: rigid bodies with their colliders and looks, and its joints."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material, create_prim, get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    stage = get_current_stage()
    root = create_prim(root_path, "Xform", translation=translation, orientation=orientation, stage=stage)
    grips: dict[tuple, str] = {}

    def grip(friction) -> str:
        friction = tuple(part.friction if friction is None else friction)
        if friction not in grips:
            grips[friction] = f"{root_path}/friction{len(grips)}"
            cfg = sim_utils.RigidBodyMaterialCfg(static_friction=friction[0], dynamic_friction=friction[1])
            spawn_physics_material(grips[friction], cfg)
        return grips[friction]

    looks: dict[tuple, str] = {}

    def look(color) -> str:
        if color not in looks:
            looks[color] = f"{root_path}/looks/c{len(looks)}"
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.4)
            cfg.func(looks[color], cfg)
        return looks[color]

    def body(path: str, mass: float | None):
        prim = stage.GetPrimAtPath(path)
        if mass is None:
            return
        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(MAX_DEPENETRATION)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))

    def colliders(path: str, origin, shapes):
        for i, s in enumerate(shapes):
            sp = f"{path}/s{i}"
            prim = _define_shape(stage, sp, s, origin)
            if s.visible:
                bind_visual_material(sp, look(s.color), stage=stage)
            if s.collide:
                col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                col.CreateContactOffsetAttr(CONTACT_OFFSET)
                col.CreateRestOffsetAttr(REST_OFFSET)
                bind_physics_material(sp, grip(s.friction), stage=stage)

    if not part.joints:  # one rigid body (loose) or a static collider (fixed) at the root
        body(root_path, part.mass)
        colliders(root_path, np.zeros(3), part.shapes)
        return root

    UsdPhysics.ArticulationRootAPI.Apply(root)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root)
    art.CreateEnabledSelfCollisionsAttr(True)  # links collide with each other, except jointed pairs
    art.CreateSolverPositionIterationCountAttr(SOLVER_ITERATIONS[0])
    art.CreateSolverVelocityIterationCountAttr(SOLVER_ITERATIONS[1])
    for name, origin, shapes in part.links:
        create_prim(f"{root_path}/{name}", "Xform", translation=tuple(float(v) for v in origin), stage=stage)
        j = next((j for j in part.joints if j.name == name), None)
        body(f"{root_path}/{name}", j.mass if j else (part.mass or 1.0))
        colliders(f"{root_path}/{name}", origin, shapes)
    UsdGeom.Scope.Define(stage, f"{root_path}/joints")
    if part.mass is None:  # fixed: the body is welded to the world where it was spawned
        weld = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/joints/world")
        pose = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(f"{root_path}/base")).RemoveScaleShear()
        weld.CreateBody1Rel().SetTargets([Sdf.Path(f"{root_path}/base")])
        weld.CreateLocalPos0Attr().Set(Gf.Vec3f(pose.ExtractTranslation()))
        weld.CreateLocalRot0Attr().Set(Gf.Quatf(pose.ExtractRotationQuat()))
        weld.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        weld.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    for j in part.joints:
        hinge = isinstance(j, Hinge)
        kind = UsdPhysics.RevoluteJoint if hinge else UsdPhysics.PrismaticJoint
        joint = kind.Define(stage, f"{root_path}/joints/{j.name}")
        frame = _usd_quat(_x_to(j.axis))
        joint.CreateAxisAttr("X")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{root_path}/base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{root_path}/{j.name}")])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*j.origin))
        joint.CreateLocalRot0Attr().Set(frame)
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(frame)
        unit = math.degrees(1.0) if hinge else 1.0  # USD revolute limits and drive gains are per degree
        joint.CreateLowerLimitAttr(j.lower * unit)
        joint.CreateUpperLimitAttr(j.upper * unit)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if hinge else "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(j.spring / unit)
        drive.CreateDampingAttr(j.damping / unit)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateMaxForceAttr(j.max_effort)
    return root


def _arena_object(part: Part):
    """The part as an Arena object: a static collider, a rigid object, or an articulation."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
    from isaaclab_arena.utils.pose import Pose

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        return _author(part, prim_path, translation, orientation)

    class PartObject(Object):
        def _generate_articulation_cfg(self):
            actuators = {
                j.name: ImplicitActuatorCfg(
                    joint_names_expr=[j.name], stiffness=j.spring, damping=j.damping, effort_limit_sim=j.max_effort
                )
                for j in part.joints
            }
            return self._add_initial_pose_to_cfg(
                ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            )

        def _build_reset_event(self):
            event = super()._build_reset_event()
            # Arena puts an articulation's root back on reset; the joints go back to 0 as well.
            if event is not None and self.object_type == ObjectType.ARTICULATION:
                event.func = reset_articulation_pose_and_joints
            return event

    if part.joints:
        kind = ObjectType.ARTICULATION
    else:
        kind = ObjectType.BASE if part.mass is None else ObjectType.RIGID
    yaw = part.yaw
    return PartObject(
        name=part.name,
        object_type=kind,
        spawner_cfg=SpawnerCfg(func=spawn),
        initial_pose=Pose(position_xyz=tuple(part.pos), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))),
    )


def _np(x) -> np.ndarray:
    x = x.torch if hasattr(x, "torch") else x
    return x.detach().cpu().numpy().astype(np.float64)


def _read(base, part: Part) -> PartState:
    """A part's state in env 0 (world frame relative to the env origin)."""
    if part.joints == () and part.mass is None:
        return part.state()
    asset = base.scene[part.name]
    origin = _np(base.scene.env_origins[0])
    lin, ang = _np(asset.data.root_link_lin_vel_w)[0], _np(asset.data.root_link_ang_vel_w)[0]
    if not part.joints:
        pose = _np(asset.data.root_link_pose_w)[0]
        return part.make_state({"base": (pose[:3] - origin, _quat_to_matrix(pose[3:]))}, {}, lin, ang)
    bodies = _np(asset.data.body_link_pose_w)[0]
    poses = {n: (bodies[i, :3] - origin, _quat_to_matrix(bodies[i, 3:])) for i, n in enumerate(asset.body_names)}
    q = _np(asset.data.joint_pos)[0]
    joints = {n: q[i] for i, n in enumerate(asset.joint_names)}
    return part.make_state({name: poses[name] for name, _, _ in part.links}, joints, lin, ang)


def _set_target(base, part: str, joint: str, value: float) -> None:
    import torch

    asset = base.scene[part]
    ids = asset.find_joints([joint], preserve_order=True)[0]
    target = torch.tensor([[float(value)]], dtype=torch.float32, device=base.device)
    asset.set_joint_position_target_index(target=target, joint_ids=ids)


def _reset_targets(base) -> None:
    for part in _SPECS.values():
        for j in part.joints:
            _set_target(base, part.name, j.name, 0.0)


# -- hooks: patch yam_arena when its modules load ---------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_parts_patched", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def patched_task_object(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return task_object(spec)

    def patched_object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(env, plain) if plain else {}
        out.update({n: _read(env.unwrapped, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = patched_task_object, patched_object_states
    scene._parts_patched = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_parts_patched", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def patched_task_step(self):
        for fn in _ON_STEP:
            fn(Sim(self))
        return task_step(self)

    def patched_reset(self, scene, *, seed=None):
        if self._env is not None:
            _reset_targets(self._base)
        obs = reset(self, scene, seed=seed)
        for fn in _ON_RESET:
            fn(Sim(self))
        return obs

    cls._task_step, cls.reset = patched_task_step, patched_reset
    cls._parts_patched = True


_PATCHES = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self.loader, self.patch = loader, patch

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    """Finds the yam_arena modules we patch with the other finders, then patches them once they have run."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, _PATCHES[fullname])
                return spec
        return None


def _install() -> None:
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(type(f).__name__ == "_Finder" and getattr(f, "parts_kit", False) for f in sys.meta_path):
        finder = _Finder()
        finder.parts_kit = True
        sys.meta_path.insert(0, finder)


_install()
