"""bodykit: rigid bodies built from several boxes and cylinders, for yam_arena task files.

yam_arena's task-file objects are single primitives. A :class:`Body` is one rigid body made of several shapes (a pan
with a floor, walls and a handle, say), some of which may be visual only (markings that nothing collides with), with
its own friction. Keep this file next to the task file and import it from there:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from bodykit import Body, BoxShape, CylShape

    PAN = Body.build("pan", pos=(0.27, -0.08, 0.75), mass=0.45, shapes=[CylShape((0, 0, 0.0025), 0.106, 0.005), ...])

Importing it hooks ``yam_arena.scene`` without changing yam_arena: such a body is spawned as one rigid body, and
``success()``/``score()`` receive a :class:`BodyState` for it whose ``pos`` and ``quat_xyzw`` are the body origin's
pose (yam_arena reports the bounding-box centre for primitives). It also hooks the embodiment's ``reset()`` and step,
so a task file can clear what it tracks over an episode (:func:`on_reset`) and look at the simulation after every
control step (:func:`on_step`), for instance at where the grippers are. Two bodies can be joined by a :class:`Hinge`
(a PhysX revolute joint), optionally one that breaks under a set force: a perforation, say, or by a :class:`Thread` (a
PhysX D6 joint the task file switches on and drives while the episode runs): a nut on a screw. Pure Python at import time: everything that needs
Isaac Sim is imported lazily, so the task file still loads without the simulator.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Body", "BodyState", "BoxShape", "CylShape", "SphereShape", "Thread", "allow_matrix_grips", "as_numpy",
           "install", "on_reset", "on_step", "plate", "rotation", "segment", "set_slide_target", "set_thread_on", "thread"]

Color = tuple[float, float, float]


@dataclass(frozen=True)
class BoxShape:
    """A box in the body frame: ``centre`` and ``size`` (m), turned ``yaw`` (rad) about the body's z axis, or, when
    ``quat_xyzw`` is given, rotated by that quaternion instead. ``color=None``: collision only, not rendered (any
    shape)."""

    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    color: Color = (0.6, 0.6, 0.6)
    yaw: float = 0.0
    collide: bool = True
    quat_xyzw: tuple[float, float, float, float] | None = None

    def quat(self) -> tuple[float, float, float, float]:
        if self.quat_xyzw is not None:
            return tuple(self.quat_xyzw)
        return (0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2))

    def half_extent(self) -> np.ndarray:
        return np.abs(rotation(self.quat())) @ (np.array(self.size) / 2)


def _quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def plate(p1, p2, width: float, thickness: float, normal, color: Color, collide: bool = True) -> BoxShape:
    """A flat plate whose middle line runs from ``p1`` to ``p2`` (body frame, m), ``width`` wide, ``thickness`` thick
    along ``normal`` (made perpendicular to the line): a wall panel of a cylinder or a cone, say."""
    a, b = np.asarray(p1, float), np.asarray(p2, float)
    d = b - a
    length = float(np.linalg.norm(d))
    d /= length
    n = np.asarray(normal, float) - np.dot(normal, d) * d
    n /= np.linalg.norm(n)
    t = np.cross(d, n)
    return BoxShape(tuple(((a + b) / 2).tolist()), (thickness, width, length), color, collide=collide,
                    quat_xyzw=_quat_from_matrix(np.column_stack([n, t, d])))


def segment(p1, p2, thickness: float, color: Color, collide: bool = True) -> BoxShape:
    """A thin square bar from ``p1`` to ``p2`` (body frame, m): a box whose long (z) axis runs along the segment."""
    a, b = np.asarray(p1, float), np.asarray(p2, float)
    d = b - a
    length = float(np.linalg.norm(d))
    d /= length
    axis = np.cross([0.0, 0.0, 1.0], d)
    s, c = float(np.linalg.norm(axis)), float(d[2])
    if s < 1e-9:
        q = (0.0, 0.0, 0.0, 1.0) if c > 0 else (1.0, 0.0, 0.0, 0.0)
    else:
        angle = math.atan2(s, c)
        axis /= s
        q = (*(axis * math.sin(angle / 2)).tolist(), math.cos(angle / 2))
    return BoxShape(tuple(((a + b) / 2).tolist()), (thickness, thickness, length), color, collide=collide, quat_xyzw=q)


@dataclass(frozen=True)
class CylShape:
    """A cylinder in the body frame, its axis along the body's ``axis`` ("X", "Y" or "Z")."""

    centre: tuple[float, float, float]
    radius: float
    height: float
    color: Color = (0.6, 0.6, 0.6)
    collide: bool = True
    axis: str = "Z"

    def half_extent(self) -> np.ndarray:
        half = np.full(3, self.radius)
        half["XYZ".index(self.axis)] = self.height / 2
        return half


@dataclass(frozen=True)
class SphereShape:
    """A sphere in the body frame."""

    centre: tuple[float, float, float]
    radius: float
    color: Color = (0.6, 0.6, 0.6)
    collide: bool = True

    def half_extent(self) -> np.ndarray:
        return np.full(3, self.radius)


@dataclass(frozen=True)
class Body(Box):
    """A loose rigid body made of ``shapes``. It subclasses Box only so yam_arena's task-file checks accept it:
    ``pos`` is the body origin (world, m) and ``size`` its bounding box. ``friction`` is (static, dynamic);
    ``damping`` is (linear, angular). A ``kinematic`` body stays where it is put (its mass is then unused). ``com``, if
    given, is the centre of mass in the body frame (m); otherwise PhysX puts it where the shapes would put it at one
    density, which is wrong for a body of light and heavy parts (a steel blade on a wooden handle)."""

    shapes: tuple = ()
    kinematic: bool = False  # True: fixed where it is put (a PhysX kinematic body), for fixed parts with several shapes
    friction: tuple[float, float] = (0.6, 0.5)
    damping: tuple[float, float] = (0.0, 0.05)
    contact_offset: float = 0.002
    solver_iterations: int = 16
    com: tuple[float, float, float] | None = None
    start_xyzw: tuple[float, float, float, float] | None = None  # the starting orientation, if not just ``yaw``
    ignore: tuple[str, ...] = ()  # names of bodies (earlier in OBJECTS) this one never collides with
    slick: bool = False  # True: contacts with this body take the lower of the two frictions (a smooth, slick surface)

    @staticmethod
    def build(name: str, pos, mass: float, shapes, yaw: float = 0.0, color: Color = (0.6, 0.6, 0.6), **kw) -> "Body":
        """A Body whose ``size`` is the extent of its shapes."""
        lo = np.min([np.array(s.centre) - s.half_extent() for s in shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in shapes], axis=0)
        return Body(name=name, pos=tuple(float(v) for v in pos), size=tuple((hi - lo).tolist()), mass=mass, yaw=yaw,
                    color=color, shapes=tuple(shapes), **kw)

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min([np.array(s.centre) - s.half_extent() for s in self.shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in self.shapes], axis=0)
        return lo, hi

    def _check(self) -> None:
        super()._check()
        if not self.shapes:
            raise ValueError(f"{self.name}: a Body needs at least one shape")
        if self.mass is None:
            raise ValueError(f"{self.name}: a Body must be loose (give it a mass)")


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass(frozen=True)
class BodyState(ObjectState):
    """``pos``/``quat_xyzw``: the body origin's pose (world, m); velocities; world-aligned bounds."""

    def rotation(self) -> np.ndarray:
        return rotation(self.quat_xyzw)

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def tilt(self) -> float:
        """Angle (rad) between the body's z axis and the vertical."""
        return float(math.acos(max(-1.0, min(1.0, self.rotation()[2, 2]))))


def state_from_pose(spec: Body, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> BodyState:
    lo, hi = spec.local_bounds()
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    pts = np.asarray(pos, float) + corners @ rotation(quat_xyzw).T
    return BodyState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat_xyzw, float),
                     lin_vel=np.asarray(lin_vel, float), ang_vel=np.asarray(ang_vel, float),
                     low=pts.min(0), high=pts.max(0))


# -- spawning (needs Isaac Sim; imported lazily) ------------------------------------------------------------------

_SPECS: dict[str, Body] = {}


@dataclass(frozen=True)
class Hinge:
    """A revolute joint between two Bodies, about a line through ``anchor`` (world, m, at the start) along ``axis``
    ("X", "Y" or "Z"; the bodies start unrotated). ``limits_deg`` bounds the turn of ``body1`` against ``body0``;
    ``break_force`` (N) and ``break_torque`` (N m), if given, make PhysX break the joint for good once exceeded.
    ``body0`` must come before ``body1`` in OBJECTS: the joint is authored when ``body1`` is spawned. ``body1`` may
    start rotated (its ``start_xyzw``) if its origin is on the hinge line at ``anchor``. ``spring`` (N m per degree)
    pulls ``body1`` toward ``target_deg``; ``damping`` (N m per degree/s) resists turning (a stiff or a damped hinge)."""

    name: str
    body0: str
    body1: str
    anchor: tuple[float, float, float]
    axis: str = "Z"
    limits_deg: tuple[float, float] | None = None
    break_force: float | None = None
    break_torque: float | None = None
    spring: float = 0.0
    damping: float = 0.0
    target_deg: float = 0.0


_HINGES: list[Hinge] = []
_HINGE_PATHS: dict[str, str] = {}
_HINGE_ATTRS: dict[str, tuple] = {}
_BREAKABLE_PATHS: list[str] = []  # authored joints that can break, re-armed at every reset


def hinge(*args, **kwargs) -> Hinge:
    """Register a :class:`Hinge` (same arguments) and return it."""
    h = Hinge(*args, **kwargs)
    _HINGES.append(h)
    return h


def _author_hinges(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for h in _HINGES:
        if h.body1 != spec.name:
            continue
        b0, b1 = _SPECS[h.body0], spec
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{h.name}")
        joint.CreateBody0Rel().SetTargets([f"{parent}/{h.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(h.anchor, float) - np.asarray(b0.pos, float)).tolist()))
        if b1.start_xyzw is not None:  # its origin is on the hinge line; the hinge's 0 is body1 unrotated
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        else:
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(h.anchor, float) - np.asarray(b1.pos, float)).tolist()))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr(h.axis)
        if h.limits_deg is not None:
            joint.CreateLowerLimitAttr(float(h.limits_deg[0]))
            joint.CreateUpperLimitAttr(float(h.limits_deg[1]))
        _HINGE_PATHS[h.name] = f"{prim_path}/{h.name}"
        _HINGE_ATTRS.pop(h.name, None)
        if h.spring or h.damping:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")  # USD angular drives take degrees
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(h.spring))
            drive.CreateDampingAttr(float(h.damping))
            drive.CreateTargetPositionAttr(float(h.target_deg))
            drive.CreateMaxForceAttr(1.0e6)
        if h.break_force is not None:
            joint.CreateBreakForceAttr(float(h.break_force))
            _BREAKABLE_PATHS.append(f"{prim_path}/{h.name}")
        if h.break_torque is not None:
            joint.CreateBreakTorqueAttr(float(h.break_torque))
            if f"{prim_path}/{h.name}" not in _BREAKABLE_PATHS:
                _BREAKABLE_PATHS.append(f"{prim_path}/{h.name}")


def set_hinge_drive(name: str, target_deg: float | None = None, spring: float | None = None,
                    damping: float | None = None) -> None:
    """Move a sprung :class:`Hinge`'s target (degrees) and/or change its spring (N m/deg) and damping (N m s/deg), e.g.
    to lock it where it is; every reset puts them back."""
    if name not in _HINGE_ATTRS:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI(get_current_stage().GetPrimAtPath(_HINGE_PATHS[name]), "angular")
        _HINGE_ATTRS[name] = (drive.GetTargetPositionAttr(), drive.GetStiffnessAttr(), drive.GetDampingAttr())
    for attr, value in zip(_HINGE_ATTRS[name], (target_deg, spring, damping)):
        if value is not None:
            attr.Set(float(value))


def _hinges_reset() -> None:
    for h in _HINGES:
        if h.name in _HINGE_PATHS and (h.spring or h.damping):
            set_hinge_drive(h.name, h.target_deg, h.spring, h.damping)


def _rearm_hinges() -> None:
    """PhysX keeps a broken joint broken; switching it off and on again in USD has it rebuilt, whole, from the bodies'
    reset poses."""
    if not _BREAKABLE_PATHS:
        return
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()
    for path in _BREAKABLE_PATHS:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        if joint:
            attr = joint.GetJointEnabledAttr() or joint.CreateJointEnabledAttr(True)
            attr.Set(False)
            attr.Set(True)


@dataclass(frozen=True)
class Thread:
    """A screw thread between a screw (``body0``) and a nut (``body1``): a PhysX D6 joint, authored switched off. The task
    file switches it on (:func:`set_thread_on`) once the thread has caught, and off again to let the nut go. While it
    is on, the nut's ``axis1`` (its frame) stays on the screw's axis, the line through ``anchor0`` along ``axis0`` (the
    screw's frame): no sliding sideways, no tilting. The nut turns freely about the axis, against ``turn_damping``
    (N m s/rad: thread friction, a torque in proportion to how fast the two turn against each other), and slides along it
    between ``slide_limits`` (m, ``anchor1``'s distance from ``anchor0`` along the axis), pulled by a spring drive
    (``slide_stiffness`` N/m, ``slide_damping`` N s/m, at most ``slide_max_force`` N) toward a target the task file moves
    (:func:`set_slide_target`) as the screw turns. While it is on, the screw and the nut do not collide. Both
    bodies start unrotated; ``body0`` must come before ``body1`` in OBJECTS. Every reset switches it off again.
    ``play`` (m) and ``tilt_deg`` give the nut room off the axis, sideways and tilted, as a loose fit does (a rod in a
    wider hole); 0 locks it on the axis."""

    name: str
    body0: str
    body1: str
    anchor0: tuple[float, float, float]
    axis0: tuple[float, float, float]
    anchor1: tuple[float, float, float]
    axis1: tuple[float, float, float]
    slide_limits: tuple[float, float]
    turn_damping: float
    slide_stiffness: float
    slide_damping: float
    slide_max_force: float
    play: float = 0.0
    tilt_deg: float = 0.0
    always_on: bool = False  # True: authored switched on and left on (a slider that is always there)


_THREADS: list[Thread] = []
_THREAD_PATHS: dict[str, str] = {}
_THREAD_ATTRS: dict[str, tuple] = {}


def thread(*args, **kwargs) -> Thread:
    """Register a :class:`Thread` (same arguments) and return it."""
    t = Thread(*args, **kwargs)
    _THREADS.append(t)
    return t


def _x_to(v) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the shortest rotation taking +x onto the direction v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(v[0])
    if c < -0.999999:
        return (0.0, 0.0, 0.0, 1.0)  # half a turn about z
    axis = np.cross([1.0, 0.0, 0.0], v)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    angle = math.atan2(s, c)
    axis = axis / s * math.sin(angle / 2)
    return (math.cos(angle / 2), float(axis[0]), float(axis[1]), float(axis[2]))


def _author_threads(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for t in _THREADS:
        if t.body1 != spec.name:
            continue
        path = f"{prim_path}/{t.name}"
        joint = UsdPhysics.Joint.Define(stage, path)  # a plain Joint is a D6: limits lock or bound each axis
        joint.CreateBody0Rel().SetTargets([f"{parent}/{t.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in t.anchor0]))
        joint.CreateLocalRot0Attr(Gf.Quatf(*_x_to(t.axis0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in t.anchor1]))
        joint.CreateLocalRot1Attr(Gf.Quatf(*_x_to(t.axis1)))
        joint.CreateJointEnabledAttr(bool(t.always_on))
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        prim = joint.GetPrim()
        for dof, room in (("transY", t.play), ("transZ", t.play), ("rotY", t.tilt_deg), ("rotZ", t.tilt_deg)):
            limit = UsdPhysics.LimitAPI.Apply(prim, dof)  # low > high: locked
            limit.CreateLowAttr(-float(room) if room > 0 else 1.0)
            limit.CreateHighAttr(float(room) if room > 0 else -1.0)
        limit = UsdPhysics.LimitAPI.Apply(prim, "transX")
        limit.CreateLowAttr(float(t.slide_limits[0]))
        limit.CreateHighAttr(float(t.slide_limits[1]))
        turn = UsdPhysics.DriveAPI.Apply(prim, "rotX")  # USD angular drives take degrees: torque per (deg/s)
        turn.CreateTypeAttr("force")
        turn.CreateStiffnessAttr(0.0)
        turn.CreateDampingAttr(float(t.turn_damping) * math.pi / 180.0)
        turn.CreateTargetVelocityAttr(0.0)
        turn.CreateMaxForceAttr(1.0e6)
        slide = UsdPhysics.DriveAPI.Apply(prim, "transX")
        slide.CreateTypeAttr("force")
        slide.CreateStiffnessAttr(float(t.slide_stiffness))
        slide.CreateDampingAttr(float(t.slide_damping))
        slide.CreateTargetPositionAttr(float(t.slide_limits[1]))
        slide.CreateMaxForceAttr(float(t.slide_max_force))
        _THREAD_PATHS[t.name] = path
        _THREAD_ATTRS.pop(t.name, None)


@dataclass(frozen=True)
class Ball:
    """A ball joint: ``body1`` hangs from ``body0`` at ``anchor`` (world, m, at the start; both bodies start unrotated),
    free to swing every way and to spin about ``axis`` (the twist axis, from ``body0`` toward ``body1``), against
    ``swing_damping`` and ``twist_damping`` (N m s/rad). A PhysX D6 joint with its three slides locked. ``body0`` must
    come before ``body1`` in OBJECTS."""

    name: str
    body0: str
    body1: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    swing_damping: float = 0.0
    twist_damping: float = 0.0


_BALLS: list[Ball] = []


def ball(*args, **kwargs) -> Ball:
    """Register a :class:`Ball` (same arguments) and return it."""
    b = Ball(*args, **kwargs)
    _BALLS.append(b)
    return b


def _author_balls(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for b in _BALLS:
        if b.body1 != spec.name:
            continue
        b0 = _SPECS[b.body0]
        joint = UsdPhysics.Joint.Define(stage, f"{prim_path}/{b.name}")
        joint.CreateBody0Rel().SetTargets([f"{parent}/{b.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(b.anchor, float) - np.asarray(b0.pos, float)).tolist()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(b.anchor, float) - np.asarray(spec.pos, float)).tolist()))
        q = Gf.Quatf(*_x_to(b.axis))
        joint.CreateLocalRot0Attr(q)
        joint.CreateLocalRot1Attr(q)
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        prim = joint.GetPrim()
        for dof in ("transX", "transY", "transZ"):
            limit = UsdPhysics.LimitAPI.Apply(prim, dof)  # low > high: locked
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
        for dof, damping in (("rotX", b.twist_damping), ("rotY", b.swing_damping), ("rotZ", b.swing_damping)):
            if damping:
                drive = UsdPhysics.DriveAPI.Apply(prim, dof)  # USD angular drives take degrees
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(float(damping) * math.pi / 180.0)
                drive.CreateTargetVelocityAttr(0.0)
                drive.CreateMaxForceAttr(1.0e6)


@dataclass(frozen=True)
class Glue:
    """A fixed joint holding ``body1`` to ``body0`` as they start (both unrotated), at ``anchor`` (world, m). The task
    file can switch it off (:func:`set_glue_on`), e.g. once a cut through it is finished; every reset switches it back
    on. ``body0`` must come before ``body1`` in OBJECTS."""

    name: str
    body0: str
    body1: str
    anchor: tuple[float, float, float]


_GLUES: list[Glue] = []
_GLUE_ATTRS: dict[str, object] = {}
_GLUE_PATHS: dict[str, str] = {}


def glue(*args, **kwargs) -> Glue:
    """Register a :class:`Glue` (same arguments) and return it."""
    g = Glue(*args, **kwargs)
    _GLUES.append(g)
    return g


def _author_glues(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for g in _GLUES:
        if g.body1 != spec.name:
            continue
        b0 = _SPECS[g.body0]
        joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/{g.name}")
        joint.CreateBody0Rel().SetTargets([f"{parent}/{g.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(g.anchor, float) - np.asarray(b0.pos, float)).tolist()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(g.anchor, float) - np.asarray(spec.pos, float)).tolist()))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateJointEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        _GLUE_PATHS[g.name] = f"{prim_path}/{g.name}"
        _GLUE_ATTRS.pop(g.name, None)


def set_glue_on(name: str, on: bool) -> None:
    """Switch a :class:`Glue` on (the two bodies held together) or off (apart)."""
    if name not in _GLUE_ATTRS:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        prim = get_current_stage().GetPrimAtPath(_GLUE_PATHS[name])
        _GLUE_ATTRS[name] = UsdPhysics.Joint(prim).GetJointEnabledAttr()
    _GLUE_ATTRS[name].Set(bool(on))


def _glues_on() -> None:
    for name in _GLUE_PATHS:  # off and on again: rebuilt whole from the bodies' reset poses
        set_glue_on(name, False)
        set_glue_on(name, True)


@dataclass(frozen=True)
class Slider:
    """A prismatic joint: ``body1`` slides against ``body0`` along ``axis`` ("X", "Y" or "Z") through ``anchor`` (world,
    m, at the start; both bodies start unrotated), between ``limits`` (m, from the start), pulled by a spring drive
    (``stiffness`` N/m, ``damping`` N s/m, at most ``max_force`` N) toward ``target`` (m). The task file can move the
    target and change the stiffness (:func:`set_slider_drive`), e.g. to lock it; every reset puts them back. ``body0``
    must come before ``body1`` in OBJECTS."""

    name: str
    body0: str
    body1: str
    anchor: tuple[float, float, float]
    axis: str = "X"
    limits: tuple[float, float] | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    target: float = 0.0
    max_force: float = 1.0e3


_SLIDERS: list[Slider] = []
_SLIDER_PATHS: dict[str, str] = {}
_SLIDER_ATTRS: dict[str, tuple] = {}


def slider(*args, **kwargs) -> Slider:
    """Register a :class:`Slider` (same arguments) and return it."""
    sl = Slider(*args, **kwargs)
    _SLIDERS.append(sl)
    return sl


def _author_sliders(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for sl in _SLIDERS:
        if sl.body1 != spec.name:
            continue
        b0 = _SPECS[sl.body0]
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/{sl.name}")
        joint.CreateBody0Rel().SetTargets([f"{parent}/{sl.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(sl.anchor, float) - np.asarray(b0.pos, float)).tolist()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(sl.anchor, float) - np.asarray(spec.pos, float)).tolist()))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr(sl.axis)
        if sl.limits is not None:
            joint.CreateLowerLimitAttr(float(sl.limits[0]))
            joint.CreateUpperLimitAttr(float(sl.limits[1]))
        joint.CreateExcludeFromArticulationAttr(True)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(sl.stiffness))
        drive.CreateDampingAttr(float(sl.damping))
        drive.CreateTargetPositionAttr(float(sl.target))
        drive.CreateMaxForceAttr(float(sl.max_force))
        _SLIDER_PATHS[sl.name] = f"{prim_path}/{sl.name}"
        _SLIDER_ATTRS.pop(sl.name, None)


def _slider_attrs(name: str) -> tuple:
    if name not in _SLIDER_ATTRS:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI(get_current_stage().GetPrimAtPath(_SLIDER_PATHS[name]), "linear")
        _SLIDER_ATTRS[name] = (drive.GetTargetPositionAttr(), drive.GetStiffnessAttr(), drive.GetDampingAttr())
    return _SLIDER_ATTRS[name]


def set_slider_drive(name: str, target: float | None = None, stiffness: float | None = None,
                     damping: float | None = None) -> None:
    """Move a :class:`Slider`'s drive target (m) and/or change its stiffness (N/m) and damping (N s/m)."""
    attrs = _slider_attrs(name)
    for attr, value in zip(attrs, (target, stiffness, damping)):
        if value is not None:
            attr.Set(float(value))


def _sliders_reset() -> None:
    for sl in _SLIDERS:
        if sl.name in _SLIDER_PATHS:
            set_slider_drive(sl.name, sl.target, sl.stiffness, sl.damping)


def _thread_attrs(name: str) -> tuple:
    if name not in _THREAD_ATTRS:
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics

        prim = get_current_stage().GetPrimAtPath(_THREAD_PATHS[name])
        _THREAD_ATTRS[name] = (UsdPhysics.Joint(prim).GetJointEnabledAttr(),
                               UsdPhysics.DriveAPI(prim, "transX").GetTargetPositionAttr(),
                               UsdPhysics.DriveAPI(prim, "rotX").GetDampingAttr())
    return _THREAD_ATTRS[name]


def set_turn_damping(name: str, damping: float) -> None:
    """Change a :class:`Thread`'s turn damping (N m s/rad): how hard it is to turn the nut on the screw."""
    _thread_attrs(name)[2].Set(float(damping) * math.pi / 180.0)


def set_thread_on(name: str, on: bool) -> None:
    """Switch a :class:`Thread` on (the nut is caught on the screw) or off (the nut is free)."""
    _thread_attrs(name)[0].Set(bool(on))


def set_slide_target(name: str, x: float) -> None:
    """Move the target of a :class:`Thread`'s slide drive: where along the screw the nut is pulled (m from anchor0)."""
    _thread_attrs(name)[1].Set(float(x))


def _threads_off() -> None:
    keep = {t.name for t in _THREADS if t.always_on}
    for name in _THREAD_PATHS:
        if name not in keep:
            set_thread_on(name, False)


def _spawn_body(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the body under ``prim_path``: a plain Xform holding one rigid-body Xform with the shapes under it."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1],
                                              friction_combine_mode="min" if spec.slick else "average")
    material.func(phys, material)
    looks: dict = {}

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            surface = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            surface.func(path, surface)
            looks[color] = path
        return looks[color]

    body_path = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, body_path)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    prim = xform.GetPrim()
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    if spec.kinematic:
        rigid.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(spec.mass))
    if spec.com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in spec.com]))
    if spec.ignore:  # a joint's own collisionEnabled=False is not applied to a joint switched on while running
        parent = prim_path.rsplit("/", 1)[0]
        pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        for other in spec.ignore:
            pairs.AddTarget(f"{parent}/{other}/body")
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    rb.CreateSolverPositionIterationCountAttr(int(spec.solver_iterations))
    rb.CreateLinearDampingAttr(float(spec.damping[0]))
    rb.CreateAngularDampingAttr(float(spec.damping[1]))

    for i, shape in enumerate(spec.shapes):
        if isinstance(shape, BoxShape):
            geom = UsdGeom.Cube.Define(stage, f"{body_path}/s{i}")
            geom.CreateSizeAttr(1.0)
            geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
            qx, qy, qz, qw = shape.quat()
            if (qx, qy, qz) != (0.0, 0.0, 0.0):
                geom.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
            geom.AddScaleOp().Set(Gf.Vec3f(*shape.size))
        elif isinstance(shape, SphereShape):
            geom = UsdGeom.Sphere.Define(stage, f"{body_path}/s{i}")
            geom.CreateRadiusAttr(shape.radius)
            geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
        else:
            geom = UsdGeom.Cylinder.Define(stage, f"{body_path}/s{i}")
            geom.CreateAxisAttr(shape.axis)
            geom.CreateRadiusAttr(shape.radius)
            geom.CreateHeightAttr(shape.height)
            geom.AddTranslateOp().Set(Gf.Vec3d(*shape.centre))
        path = str(geom.GetPrim().GetPath())
        if shape.color is None:  # a collision-only shape: not rendered (a clear part: what is inside shows)
            UsdGeom.Imageable(geom.GetPrim()).MakeInvisible()
        else:
            bind_visual_material(path, look(tuple(shape.color)), stage=stage)
        if shape.collide:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
            collision.CreateContactOffsetAttr(float(spec.contact_offset))
            collision.CreateRestOffsetAttr(0.0)
            bind_physics_material(path, phys, stage=stage)
    _author_hinges(stage, prim_path, spec)
    _author_threads(stage, prim_path, spec)
    _author_balls(stage, prim_path, spec)
    _author_glues(stage, prim_path, spec)
    _author_sliders(stage, prim_path, spec)
    return root.GetPrim()


def _arena_object(spec: Body):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class BodySpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_body)
        body_name: str = ""

    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID,
        spawner_cfg=BodySpawnerCfg(body_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos),
                          rotation_xyzw=tuple(spec.start_xyzw) if spec.start_xyzw is not None else
                          (0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))),
    )


def as_numpy(x) -> np.ndarray:
    """An Isaac Lab array (a torch tensor, or a proxy with ``.torch``) as float64 NumPy."""
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


_np = as_numpy


def _body_state(environment, spec: Body) -> BodyState:
    base = environment.unwrapped
    data = base.scene[spec.name].data
    origin = _np(base.scene.env_origins[0])
    pose = _np(data.root_link_pose_w)[0]
    vel = _np(data.root_com_vel_w)[0] if hasattr(data, "root_com_vel_w") else _np(data.root_vel_w)[0]
    return state_from_pose(spec, pose[:3] - origin, pose[3:7], vel[:3], vel[3:6])


# -- hooking yam_arena.scene -------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_bodykit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Body):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _body_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._bodykit_patched = True


_RESET: list = []
_STEP: list = []


def on_reset(fn):
    """Register ``fn()`` to run after every episode reset (for state a task file keeps over an episode)."""
    _RESET.append(fn)
    return fn


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored, with the Isaac Lab env."""
    _STEP.append(fn)
    return fn


def _patch_embodiment(module) -> None:
    cls = getattr(module, "YamArenaEmbodiment", None)
    if cls is None or getattr(cls, "_bodykit_patched", False):
        return
    base_reset, base_task_step = cls.reset, cls._task_step

    def reset(self, *args, **kwargs):
        _rearm_hinges()
        _threads_off()
        _sliders_reset()
        _hinges_reset()
        obs = base_reset(self, *args, **kwargs)
        _glues_on()
        for fn in _RESET:
            fn()
        return obs

    def _task_step(self, *args, **kwargs):
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *args, **kwargs)

    cls.reset, cls._task_step = reset, _task_step
    cls._bodykit_patched = True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Finder(importlib.abc.MetaPathFinder):
    """Patches each target module right after it is first imported (yam_arena.scene needs Isaac Sim, so it loads after
    this file)."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _TARGETS[fullname](module)

        spec.loader.exec_module = exec_module
        return spec


def allow_matrix_grips() -> None:
    """Let a demo Waypoint's ``yaw`` be a full 3x3 TCP rotation (world frame) instead of an angle, for grips other than
    straight down (yam_arena.demo builds the rotation with grip_rotation(yaw), which this wraps). Idempotent."""
    from yam_arena import demo

    if getattr(demo, "_bodykit_matrix_grips", False):
        return
    plain = demo.grip_rotation

    def grip_rotation(yaw=0.0):
        return np.asarray(yaw, float) if np.ndim(yaw) == 2 else plain(yaw)

    demo.grip_rotation = grip_rotation
    demo._bodykit_matrix_grips = True


def install() -> None:
    """Hook yam_arena.scene and the embodiment (now if already imported, otherwise when they are). Idempotent."""
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
