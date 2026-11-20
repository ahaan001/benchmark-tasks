"""cardkit: thin plates (playing cards, a pack, a felt cloth) and fingertip positions for a yam_arena task file.

yam_arena task files build boxes with rubber-like friction (1.2) and a 5 mm contact offset, which is fifteen times a
playing card's thickness. This module adds :class:`Plate`: a box with its own friction, a contact offset below its
thickness and more solver iterations, free (a rigid body) or fixed (a static collider), with an optional printed
panel on its upper face. Importing the module wraps, without modifying yam_arena:

- ``yam_arena.scene.task_object``: builds a Plate as one body with one box collider.
- ``yam_arena.scene.object_states``: remembers the environment, so :func:`fingertips` and :func:`step_dt` can read it.
- ``yam_arena.inspect_embodiment.YamArenaEmbodiment.reset``: after every reset it calls the functions in
  ``ON_RESET``, so the task file can clear its whole-trial latches.
- ``yam_arena.scripted.YamScriptedPolicy``: if the task module defines ``demo_controller``, ``yam_scripted`` runs it
  closed loop on each observation.

Adapted from ``jengakit.py`` of the accepted task ``jenga-push-and-stack`` (spawner, reset and scripted-policy hooks).
Pure Python at import; Isaac Sim is imported only when the scene is built.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box

ON_RESET: list = []  # callables run after each embodiment reset
ARMS = ("left", "right")
FINGER_LINKS = ("tip_left", "tip_right")
PAD_UP = 0.013  # m: the second point checked on each finger, this far up it from its tip

# Solver settings for free plates (checked by the standing test in the README).
POS_ITERS = 32
VEL_ITERS = 4
MAX_DEPEN_V = 0.05  # m/s: slow depenetration, so a pressed pack does not spring apart
SLEEP_THRESHOLD = 0.0005
STABILIZATION = 0.0002
REST_OFFSET = 0.0


@dataclass(frozen=True)
class Plate(Box):
    """A thin box. ``friction`` is (static, dynamic) against everything it touches (combine: average)."""

    friction: tuple[float, float] = (0.12, 0.10)
    contact_offset: float = 0.0001  # m
    back: tuple[float, float, float] | None = None  # colour of a printed panel on the upper face
    back_inset: float = 0.004  # m: the panel's border
    solid: bool = True  # False: drawn only, nothing collides with it
    iterations: tuple[int, int] = (POS_ITERS, VEL_ITERS)  # solver position and velocity iterations
    max_depenetration: float = MAX_DEPEN_V  # m/s


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- fingertips (numpy only: the URDF's own forward kinematics on the measured joints) ------------------------------

_TIP_LOCAL: dict = {}


def tip_points_local() -> dict[str, np.ndarray]:
    """Per finger link, two points in the link's frame: the middle of its tip end and a point PAD_UP up the finger."""
    if not _TIP_LOCAL:
        from yam_arena import analysis, ik
        from yam_arena.demo import READY

        q = {n: float(v) for n, v in zip(ik.ARM_JOINTS, READY)}
        q.update(joint7=0.0, joint8=0.0)
        poses = analysis.fk_link_poses_world(q, (0.0, 0.0, 0.0))
        approach = poses["gripper"][:3, 2]  # the gripper's approach axis in the world
        for link in FINGER_LINKS:
            r, t = poses[link][:3, :3], poses[link][:3, 3]
            verts = np.vstack([h.vertices for h in analysis.link_hulls()[link]]) @ r.T + t
            reach = verts @ approach
            end = verts[reach > reach.max() - 0.0005].mean(axis=0)
            _TIP_LOCAL[link] = np.array([r.T @ (end - t), r.T @ (end - PAD_UP * approach - t)])
    return _TIP_LOCAL


def finger_points(joints: dict[str, float], base_pos) -> np.ndarray:
    """(4, 3) world points of one arm: tip and pad of each finger, from its eight joint positions."""
    from yam_arena import analysis

    poses = analysis.fk_link_poses_world(joints, base_pos)
    return np.vstack([tip_points_local()[link] @ poses[link][:3, :3].T + poses[link][:3, 3] for link in FINGER_LINKS])


# -- Isaac Sim side ---------------------------------------------------------------------------------------------

_SPECS: dict[str, Plate] = {}
_ENV: dict = {"env": None, "tick": 0}


def _arr(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def tick() -> int:
    """Counts object-state reads: one per control step."""
    return _ENV["tick"]


def step_dt(default: float = 0.05) -> float:
    env = _ENV["env"]
    return float(env.unwrapped.step_dt) if env is not None else default


def poses(names) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each named object's centre and 3x3 rotation, read now (for the scripted demo; does not count as a step)."""
    env, read = _ENV["env"], _ENV.get("read")
    if env is None or read is None:
        return {}
    return {n: (np.asarray(s.pos, float), quat_matrix(s.quat_xyzw)) for n, s in read(env, list(names)).items()}


def fingertips() -> np.ndarray:
    """(8, 3) tip and pad points of both arms' fingers (station frame), or an empty array before the first step."""
    env = _ENV["env"]
    if env is None:
        return np.zeros((0, 3))
    from yam_arena import station

    out = []
    try:
        for side in ARMS:
            robot = env.unwrapped.scene[f"{side}_robot"]
            pos = _arr(robot.data.joint_pos)[0]
            out.append(finger_points({n: float(pos[i]) for i, n in enumerate(robot.joint_names)}, station.BASES[side]))
    except Exception as e:  # reported once by the task's debug line; the release check then cannot pass
        _ENV["finger_error"] = repr(e)
        return np.full((1, 3), np.nan)
    return np.vstack(out)


def _spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.plate_name]
    stage = get_current_stage()
    dbl = UsdGeom.XformOp.PrecisionDouble
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=dbl).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=dbl).Set(Gf.Quatd(w, x, y, z))
    prim = root.GetPrim()
    if spec.free:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(spec.max_depenetration)
        rb.CreateSolverPositionIterationCountAttr(spec.iterations[0])
        rb.CreateSolverVelocityIterationCountAttr(spec.iterations[1])
        rb.CreateSleepThresholdAttr(SLEEP_THRESHOLD)
        rb.CreateStabilizationThresholdAttr(STABILIZATION)

    def look(path: str, color) -> str:
        vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.6)
        vis.func(path, vis)
        return path

    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/geom")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*spec.size))
    p = cube.GetPrim()
    if spec.solid:
        material = f"{prim_path}/Looks/physics"
        pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1])
        pm.func(material, pm)
        UsdPhysics.CollisionAPI.Apply(p)
        col = PhysxSchema.PhysxCollisionAPI.Apply(p)
        col.CreateContactOffsetAttr(spec.contact_offset)
        col.CreateRestOffsetAttr(REST_OFFSET)
        bind_physics_material(str(p.GetPath()), material, stage=stage)
    bind_visual_material(str(p.GetPath()), look(f"{prim_path}/Looks/look", spec.color), stage=stage)
    if spec.back is not None:
        sx, sy, sz = spec.size
        panel = UsdGeom.Cube.Define(stage, f"{prim_path}/back")
        panel.CreateSizeAttr(1.0)
        panel.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, sz / 2))
        panel.AddScaleOp().Set(Gf.Vec3f(sx - 2 * spec.back_inset, sy - 2 * spec.back_inset, 4e-5))
        bind_visual_material(str(panel.GetPrim().GetPath()), look(f"{prim_path}/Looks/back", spec.back), stage=stage)
    return prim


def _arena_object(spec: Plate):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class _Cfg(SpawnerCfg):
        func: object = clone(_spawn)
        plate_name: str = ""

    h = spec.yaw / 2
    return Object(name=spec.name, object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
                  spawner_cfg=_Cfg(plate_name=spec.name),
                  initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, math.sin(h), math.cos(h))))


def _patch_scene(scene) -> None:
    if getattr(scene, "_cardkit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def _task_object(spec):
        if isinstance(spec, Plate):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return task_object(spec)

    def _object_states(env, names):
        _ENV.update(env=env, tick=_ENV["tick"] + 1, read=object_states)
        return object_states(env, names)

    scene.task_object, scene.object_states, scene._cardkit = _task_object, _object_states, True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_cardkit", False):
        return
    reset = cls.reset

    def _reset(self, scene, *, seed=None):
        obs = reset(self, scene, seed=seed)
        for fn in ON_RESET:
            fn()
        return obs

    cls.reset, cls._cardkit = _reset, True


def _patch_scripted(scripted) -> None:
    """``yam_scripted`` runs the task module's ``demo_controller`` (if any) closed loop on each observation."""
    cls = scripted.YamScriptedPolicy
    if getattr(cls, "_cardkit", False):
        return
    reset, act = cls.reset, cls.act

    def _controller(self):
        task = getattr(self, "_task", None)
        return getattr(task.module, "demo_controller", None) if task is not None else None

    def _reset(self, scene):
        make = _controller(self)
        if make is None:
            return reset(self, scene)
        self._ck_ctl = make(dict(left=self._bases["left"], right=self._bases["right"]), self._dt)
        return None

    def _act(self, observation):
        ctl = getattr(self, "_ck_ctl", None)
        if ctl is None or _controller(self) is None:
            return act(self, observation)
        from inspect_robots.types import Action, ActionChunk

        data, done = ctl.act(observation)
        meta = {"request_stop": True} if done else {}
        return ActionChunk(actions=[Action(data=data, meta=meta)], control_hz=1.0 / self._dt)

    cls.reset, cls.act, cls._cardkit = _reset, _act, True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment,
            "yam_arena.scripted": _patch_scripted}


class _Finder(importlib.abc.MetaPathFinder):
    """Patches each target module right after it is first imported."""

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
        exec_module = spec.loader.exec_module

        def _exec(module):
            exec_module(module)
            _TARGETS[fullname](module)

        spec.loader.exec_module = _exec
        return spec


for _name, _patch in _TARGETS.items():
    if _name in sys.modules:
        _patch(sys.modules[_name])
if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
