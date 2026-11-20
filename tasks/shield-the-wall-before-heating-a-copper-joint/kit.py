"""Hooks for a yam_arena task file: compound rigid bodies, an every-step callback, and the grippers' finger positions.

Adapted from mech.py of the accepted task never-point-the-nail-gun-at-a-coworker in this repository (compounds, mass
centres, the control-step hook and the gripper states are that module's; joints, welds and ghosts are left out). One
addition: a compound body is reported at its own origin (its base shape's centre) instead of its bounding box's
centre, which moves as the body turns.

A task file stays pure Python. It imports this module, which hooks yam_arena's scene builder and step when the
simulator starts:

    kit.compound("torch", [(size, offset, rgb), (size, offset, rgb, (rx, ry, rz)), ...])   # extra boxes, one rigid body
    kit.mass_centre("torch", (0, 0, 0.07))
    kit.on_tick(fn)        # fn(sim) after every control step, before scoring: sim.t (s since the reset), sim.states

score() and ``sim.states`` also hold "<arm>_finger_l" / "<arm>_finger_r" for arm in left, right: each finger's centre
line at the TCP's height, with the gripper's orientation (its +z is the approach axis; the fingertip is 13 mm further).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

import numpy as np

COMPOUNDS: dict[str, list] = {}
MASS_CENTRES: dict[str, tuple] = {}
_TICKS: list = []


def compound(name, parts):
    """Make a task object one rigid body with extra boxes: each part is (size, offset of its centre from the object's
    centre, rgb) or (size, offset, rgb, (rx, ry, rz) in degrees). The object's mass is the whole body's."""
    COMPOUNDS[name] = [(tuple(p[0]), tuple(p[1]), tuple(p[2]), tuple(p[3]) if len(p) > 3 else None) for p in parts]


def mass_centre(name, offset):
    """Put the body's centre of mass at ``offset`` from its origin (m, body axes)."""
    MASS_CENTRES[name] = tuple(float(v) for v in offset)


def on_tick(fn):
    _TICKS.append(fn)
    return fn


def _add_parts(prim, name) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage = prim.GetStage()
    for i, (size, offset, rgb, rpy) in enumerate(COMPOUNDS[name]):
        path = prim.GetPath().AppendChild(f"part{i}")
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in offset)))
        if rpy is not None:
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        material = UsdShade.Material.Define(stage, path.AppendChild("look"))
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c ** 2.2 for c in rgb)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def matrix_to_quat_xyzw(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    t = np.trace(r)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def _patch_scene(scene) -> None:
    from isaaclab.sim.utils import clone

    from yam_arena.task_file import ObjectState

    base_task_object = scene.task_object
    base_states = scene.object_states

    def task_object(spec):
        obj = base_task_object(spec)
        if spec.name in COMPOUNDS or spec.name in MASS_CENTRES:
            cfg = obj.object_cfg.spawn
            inner = getattr(cfg.func, "__wrapped__", cfg.func)

            @clone
            def spawn(prim_path, cfg_, *args, **kwargs):
                prim = inner(prim_path, cfg_, *args, **kwargs)
                if spec.name in COMPOUNDS:
                    _add_parts(prim, spec.name)
                if spec.name in MASS_CENTRES:
                    from pxr import Gf, UsdPhysics

                    UsdPhysics.MassAPI.Apply(prim).CreateCenterOfMassAttr(Gf.Vec3f(*MASS_CENTRES[spec.name]))
                return prim

            cfg.func = spawn
        return obj

    def object_states(env, names):
        import torch
        from isaaclab.utils.math import quat_apply

        from yam_arena.constants import TCP_OFFSET

        out = base_states(env, names)
        base = env.unwrapped
        origin = base.scene.env_origins[0]
        for name in names:
            if name in COMPOUNDS and name in out:
                root = (base.arena_world.get_pose_w(name)[0, :3] - origin).detach().cpu().numpy().astype(np.float64)
                s = out[name]
                out[name] = ObjectState(root, s.quat_xyzw, s.lin_vel, s.ang_vel, s.low, s.high)
        for arm in ("left", "right"):
            robot = base.scene[f"{arm}_robot"]
            i = robot.find_bodies("gripper")[0][0]
            pose = robot.data.body_link_pose_w.torch[0:1, i]
            offset = torch.tensor([TCP_OFFSET], dtype=pose.dtype, device=pose.device)
            tcp = pose[:, :3] + quat_apply(pose[:, 3:7], offset) - origin
            # The gripper's orientation from Isaac Lab's own rotation of the basis vectors, so it is right whichever
            # order body_link_pose_w stores its quaternion in.
            basis = torch.eye(3, dtype=pose.dtype, device=pose.device)
            cols = quat_apply(pose[:, 3:7].expand(3, 4), basis).detach().cpu().numpy().astype(np.float64)
            quat = matrix_to_quat_xyzw(cols.T)
            z = np.zeros(3)
            fids = [robot.find_bodies(n)[0][0] for n in ("tip_left", "tip_right")]
            fp = robot.data.body_link_pose_w.torch[0, fids, :3]
            approach = quat_apply(pose[:, 3:7], torch.tensor([[0.0, 0.0, 1.0]], dtype=pose.dtype, device=pose.device))[0]
            mid = fp.mean(dim=0)
            for side, j in (("l", 0), ("r", 1)):
                lat = fp[j] - mid
                lat = lat - torch.dot(lat, approach) * approach
                p = (tcp[0] + lat).detach().cpu().numpy().astype(np.float64)
                out[f"{arm}_finger_{side}"] = ObjectState(p, quat, z, z, p - 0.01, p + 0.01)
        return out

    scene.task_object = task_object
    scene.object_states = object_states


class Sim:
    def __init__(self, emb, states):
        self.t = emb._t
        self.states = states


def _tick(emb) -> None:
    if not _TICKS:
        return
    from yam_arena.scene import object_states

    sim = Sim(emb, object_states(emb._env, emb.task.names))
    for fn in _TICKS:
        fn(sim)


def _patch_embodiment(mod) -> None:
    cls = mod.YamArenaEmbodiment
    if getattr(cls, "_kit_patched", False):
        return
    base_step = cls._task_step

    def _task_step(self):
        _tick(self)
        return base_step(self)

    cls._task_step = _task_step
    cls._kit_patched = True


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in ("yam_arena.scene", "yam_arena.inspect_embodiment"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        base_exec = spec.loader.exec_module

        def exec_module(module):
            base_exec(module)
            (_patch_scene if name == "yam_arena.scene" else _patch_embodiment)(module)

        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
for _name, _fn in (("yam_arena.scene", _patch_scene), ("yam_arena.inspect_embodiment", _patch_embodiment)):
    _mod = sys.modules.get(_name)
    if _mod is not None and not getattr(_mod, "_kit_done", False):
        _fn(_mod)
        _mod._kit_done = True
