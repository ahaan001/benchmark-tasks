"""A kinematic stand-in for the simulator, to shake out the demo and the scorer's wiring without simulator time.

Not physics. The arms follow their commands with a lag. The can follows a hand whose fingers have closed round its body and, let go,
comes down onto whatever is under it, standing if it was upright and lying if it was level. The clip follows a hand whose fingers have
closed on its sides. Brought up behind the tab's rear end in line with the tab, it goes onto the tab: from then on it can only slide
along the tab, no further than home, and once home it stays on the can. Out of line it stops at the tab's end. Closed fingertips that
come down on the tab's rear end press it, as far as a clip on the tab lets it go. Nothing here has weight, friction or a spring.
It runs the task's real demo_script, Trial and rule, so it finds wrong calls, unreachable poses, loops that never end, end states that
do not score, and how many control steps a run takes. With --clear it also measures the arms' collision hulls against the scene.

    PYTHONPATH=<yam-isaac-lab-arena> python3 dryrun.py [modes] [--clear] [--quiet]
"""
import importlib.util
import math
import os
import pathlib
import sys
import types

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "1")  # many small products: threads only get in each other's way

import numpy as np  # noqa: E402

TASK_DIR = str(pathlib.Path(__file__).resolve().parent)
QUIET = "--quiet" in sys.argv
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import demokit as d  # noqa: E402
import mechkit as m  # noqa: E402
import spray_geometry as g  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN, READY  # noqa: E402

spec = importlib.util.spec_from_file_location("spray_task", f"{TASK_DIR}/task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
TOP = task.TOP
UP = np.array([0.0, 0.0, TOP])
DT = 1.0 / d.CONTROL_HZ
LAG, FINGER_LAG = 0.35, 0.5
CAMERA = (0.050, 0.050, 0.040)  # a box round a wrist camera and its mount, in the gripper's frame (a D405 is 42 x 42 x 23 mm)
ENTER = (0.002, 8.0)  # the clip goes onto the tab if it meets the tab's end within this of the tab's line (m) and this of square (degrees)
REACH_BACK = 0.020  # the tab's end is in the sleeve's mouth once the clip is this far short of home
HOME_ROT, HOME_POS = g.clip_home()


class State:
    """Stands in for a part's state."""

    def __init__(self, pos, rot, joints=None):
        self.pos, self.rot, self.joints = np.asarray(pos, float), np.asarray(rot, float), joints or {}

    def rotation(self):
        return self.rot

    def deg(self, link):
        return math.degrees(self.joints[f"{link}_joint"])


def tilt_of(rot) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


class World:
    def __init__(self):
        self.q = {s: np.array(READY, float) for s in ("left", "right")}
        self.f = {s: FINGER_OPEN for s in ("left", "right")}
        self.body = {"can": [task.CAN_POS0.copy(), task.CAN_ROT0.copy()], "clip": [task.CLIP_POS0.copy(), task.CLIP_ROT0.copy()]}
        self.held = {}  # name -> (side, position in the hand's frame, rotation in the hand's frame)
        self.on_tab = task.START == "seated"
        self.back = 0.0  # how far short of home the clip is on the tab
        self.tab, self.pushed, self.notes = 0.0, False, []

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)
            if not QUIET:
                print(f"   [world] {text}", flush=True)

    # -- what the task reads ----------------------------------------------------------------------------------------------
    def tcp(self, side):
        return ik.tcp_pose(self.q[side], np.asarray(station.BASES[side], float))

    def state(self, name):
        pos, rot = self.body[name]
        return State(pos, rot, {"tab_joint": math.radians(self.tab), "strap_joint": 0.0})

    def fingertips(self):
        out = []
        for side in ("left", "right"):
            pos, rot = self.tcp(side)
            origin = pos - rot @ np.array([0.0, 0.0, 0.134])
            opening = abs(self.f[side])
            for sx in (-1.0, 1.0):
                for dy in (-0.018, 0.0, 0.018):
                    for dz in (0.147, 0.117):
                        out.append(origin + rot @ np.array([sx * (opening + 0.0045), dy, dz]))
        return np.array(out)

    # -- one control step -------------------------------------------------------------------------------------------------
    def step(self, ctx):
        for side, arm in (("left", ctx.left), ("right", ctx.right)):
            self.q[side] = self.q[side] + LAG * (arm.cmd[:6] - self.q[side])
            self.f[side] = self.f[side] + FINGER_LAG * (arm.cmd[6] * FINGER_OPEN - self.f[side])
        for name, half in (("can", g.CAN_R), ("clip", g.CLIP_W / 2)):  # fingers stop on what they hold
            if name in self.held:
                self.f[self.held[name][0]] = min(self.f[self.held[name][0]], -half)
        self.grasps()
        self.carry()
        self.clip_on_tab()
        self.press()

    def grasps(self):
        for name, half in (("can", g.CAN_R), ("clip", g.CLIP_W / 2)):
            pos, rot = self.body[name]
            if name in self.held:
                side = self.held[name][0]
                if abs(self.f[side]) > half + 0.003:
                    del self.held[name]
                    self.let_go(name)
                continue
            for side in ("left", "right"):
                hand_pos, hand = self.tcp(side)
                rel = rot.T @ (hand_pos - pos)  # the TCP in the part's frame
                if name == "can":
                    near = float(np.hypot(rel[0], rel[1])) < 0.008 and 0.03 < rel[2] < g.CAN_WALL_H and abs(float(hand[:, 0] @ rot[:, 2])) < 0.3
                else:
                    near = float(np.linalg.norm(rel - g.GRIP_AT)) < 0.006 and abs(float(hand[:, 0] @ rot[:, 1])) > 0.95
                if near and abs(self.f[side]) <= half + 0.0005:
                    self.held[name] = (side, hand.T @ (pos - hand_pos), hand.T @ rot)
                    self.note(f"{side} hand holds the {name}: TCP at {np.round(rel * 1000, 1)} mm in its frame")

    def support(self, xy) -> float:
        """The height (world) of what is under a point."""
        p = np.array([[xy[0], xy[1], 0.01]])
        if g.in_bin(p)[0]:
            return TOP + g.BIN_WALL
        mx, my = g.MAT_XY
        if abs(xy[0] - mx) <= g.MAT[0] / 2 and abs(xy[1] - my) <= g.MAT[1] / 2:
            return TOP + g.MAT[2]
        return TOP

    def let_go(self, name):
        pos, rot = self.body[name]
        if name == "clip":
            if self.on_tab:
                return  # it stays where it is on the tab
            stand, up = g.clip_standing(math.degrees(math.atan2(rot[1, 0], rot[0, 0])))
            z = self.support(pos[:2]) + up
            self.body[name] = [np.array([pos[0], pos[1], z]), stand]
            self.note(f"the clip is let go {1000 * (pos[2] - z):.0f} mm up and comes down")
            return
        hold = pos + rot @ task.HOLD
        if tilt_of(rot) < 45.0:
            z = self.support(pos[:2])
            self.body[name] = [np.array([pos[0], pos[1], z]), g.rot_z(math.degrees(math.atan2(rot[1, 0], rot[0, 0])))]
            where = "standing"
        else:
            z = self.support(hold[:2]) + g.CAN_R
            self.body[name] = [pos + np.array([0.0, 0.0, z - hold[2]]), rot]
            where = "lying"
        inside = bool(np.all(g.in_bin(task.CAN_POINTS @ self.body[name][1].T + self.body[name][0] - UP)))
        self.note(f"the can is let go {1000 * (hold[2] - z):.0f} mm up and comes down {where}, {'in the bin' if inside else 'outside the bin'}")

    def carry(self):
        for name, (side, p, rr) in self.held.items():
            hand_pos, hand = self.tcp(side)
            self.body[name] = [hand_pos + hand @ p, hand @ rr]

    def clip_on_tab(self):
        can_pos, can_rot = self.body["can"]
        clip_pos, clip_rot = self.body["clip"]
        held = "clip" in self.held
        if held:
            offset, tilt = g.clip_on_can(can_pos, can_rot, clip_pos, clip_rot)
            back = -float(offset[0])
            off = float(np.hypot(offset[1], offset[2]))
            if not self.on_tab and back < REACH_BACK:
                if off <= ENTER[0] and tilt <= ENTER[1] and back > REACH_BACK - 0.010:
                    self.on_tab = True
                    self.note(f"the clip goes onto the tab's end, {off * 1000:.1f} mm and {tilt:.1f} degrees out of line")
                elif off < 0.012 and back > -0.005:
                    self.note(f"the clip meets the tab's end {off * 1000:.1f} mm and {tilt:.1f} degrees out of line, and stops there")
                    back = REACH_BACK
                    self.back = back
                    self.body["clip"] = [can_pos + can_rot @ (HOME_POS + HOME_ROT @ np.array([-back, offset[1], offset[2]])), clip_rot]
                    return
            if self.on_tab:
                if back > REACH_BACK + 0.004:
                    self.on_tab = False
                    self.note("the clip is drawn back off the tab")
                    return
                self.back = max(back, 0.0)
        if self.on_tab:
            if not held and self.back > g.CATCH_BACKLASH + 0.0002 and tilt_of(can_rot) > 45.0:
                self.on_tab = False
                self.note("the clip, short of its catch, comes off as the can is turned")
                self.let_go("clip")
                return
            if self.back <= g.CATCH_BACKLASH:
                self.note("the clip's catch is past the ridge: it is home")
            self.body["clip"] = [can_pos + can_rot @ (HOME_POS - self.back * HOME_ROT[:, 0]), can_rot @ HOME_ROT]

    def press(self):
        can_pos, can_rot = self.body["can"]
        free = g.tab_free_deg(self.back) if self.on_tab else g.TAB_TRAVEL_DEG
        want = g.TAB_TRAVEL_DEG if self.pushed else 0.0
        rear = can_pos + can_rot @ g.TAB_REAR
        for side in ("left", "right"):
            if abs(self.f[side]) > 0.004:
                continue  # only closed fingers press
            pos, rot = self.tcp(side)
            tips = pos + rot[:, 2] * 0.013
            if np.linalg.norm((tips - rear)[:2]) < 0.012 and tips[2] < rear[2] + 0.004:
                want = max(want, math.degrees((rear[2] + 0.002 - tips[2]) / g.TAB_REACH))
        self.tab = float(min(max(want, 0.0), free))
        if self.tab >= g.TAB_FIRED_DEG:
            self.note("the trigger tab goes down past half its travel")


class Ctx(d.Ctx):
    def observed(self, side):
        return np.r_[WORLD.q[side], WORLD.f[side] / FINGER_OPEN]


# -- clearances (--clear): what the kinematic world cannot feel ----------------------------------------------------------------------
def _labelled(method):
    def run(self, *a, **k):
        if "label" in k:
            self.label = k["label"]
        return (yield from method(self, *a, **k))
    return run


for _name in ("move", "servo", "turn", "arc", "open", "close"):
    setattr(d.Arm, _name, _labelled(getattr(d.Arm, _name)))

_SHELLS: dict = {}


def _shell(size, step=0.012) -> np.ndarray:
    """Points over a box's surface, in its frame."""
    key = tuple(round(float(v), 6) for v in size)
    if key not in _SHELLS:
        axes = [np.linspace(-v / 2, v / 2, max(2, int(math.ceil(v / step)) + 1)) for v in key]
        pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        _SHELLS[key] = pts[np.any(np.abs(np.abs(pts) - np.asarray(key) / 2) < 1e-9, axis=1)]
    return _SHELLS[key]


class Clearance:
    """The least distance between each arm's collision hulls and each thing in the scene, by the primitive the arm was in, and between
    the two arms. Fingers on what they hold read about 0; a round thing is taken as a box 0.85 of its width."""

    def __init__(self):
        from yam_arena import analysis
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

        self.an, self.joints, self.least, self.arms = analysis, (ARM_JOINTS, FINGER_JOINTS), {}, {}

    def hulls(self, world, side):
        """The arm's collision hulls, and a box where its wrist camera is (the simulator's arm has no collider there; a real one has a camera)."""
        from yam_arena.constants import WRIST_CAM_POS

        q = {n: float(v) for n, v in zip(self.joints[0], world.q[side])} | {n: float(world.f[side]) for n in self.joints[1]}
        poses = self.an.fk_link_poses_world(q, station.BASES[side])
        rot, pos = poses["gripper"][:3, :3], poses["gripper"][:3, 3]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(CAMERA) / 2 + WRIST_CAM_POS
        normals = np.vstack([np.eye(3), -np.eye(3)])
        planes = np.c_[normals @ rot.T, np.r_[np.array(CAMERA) / 2 + WRIST_CAM_POS, np.array(CAMERA) / 2 - WRIST_CAM_POS] + (normals @ rot.T) @ pos]
        return self.an.world_hulls(poses) + [("camera", self.an.Hull(vertices=corners @ rot.T + pos, planes=planes))]

    def things(self, world) -> list:
        """(name, centre, rotation, size) in the world."""
        eye = np.eye(3)
        out = [("mat", np.array([*g.MAT_XY, g.MAT[2] / 2]) + UP, eye, g.MAT), ("table", np.array([0.3, 0.0, -0.05]) + UP, eye, (2.0, 2.0, 0.1))]
        out += [(f"bin{k}", np.array(c) + [*g.BIN_XY, 0.0] + UP, eye, sz) for k, (c, sz) in enumerate(g.bin_boxes())]
        out.append(("stove", np.array([*g.STOVE_XY, g.BIN_WALL + g.STOVE[2] / 2]) + UP, eye, g.STOVE))
        out.append(("cup", np.array([*g.CUP_XY, g.BIN_WALL + g.CUP_H / 2]) + UP, eye, (1.7 * g.CUP_R, 1.7 * g.CUP_R, g.CUP_H)))
        pos, rot = world.body["can"]
        wide = 0.85 * 2 * g.CAN_R
        out.append(("can body", pos + rot @ np.array([0.0, 0.0, g.COLLAR_Z[1] / 2]), rot, (wide, wide, g.COLLAR_Z[1])))
        out.append(("can head", pos + rot @ np.array([sum(g.BLOCK_R) / 2, 0.0, sum(g.BLOCK_Z) / 2]), rot, (g.BLOCK_R[1] - g.BLOCK_R[0], g.BLOCK_W, g.BLOCK_Z[1] - g.BLOCK_Z[0])))
        out.append(("can loop", pos + rot @ g.side(27.75, 184.0), rot, (0.0355, g.LOOP_W, 0.036)))
        out.append(("can tab", pos + rot @ g.side(26.0, 207.0), rot, (0.024, g.TAB_W, 0.010)))
        pos, rot = world.body["clip"]
        out.append(("clip", pos + rot @ np.array([0.002, 0.0, -0.004]), rot, (0.024, g.CLIP_W, 0.018)))
        return out

    def sample(self, world, ctx):
        an = self.an
        sets = {side: self.hulls(world, side) for side in ("left", "right")}
        labels = {side: getattr(ctx.arm(side), "label", "start") for side in sets}
        for name, centre, rot, size in self.things(world):
            size = np.asarray(size, float)
            reach = float(np.linalg.norm(size)) / 2 + 0.04
            shell = None
            for side, hulls in sets.items():
                best, link = 0.04, None
                for part, h in hulls:
                    if name == "table" and not part.startswith(("tip", "gripper")):
                        continue
                    if np.min(np.linalg.norm(h.vertices - centre, axis=1)) > reach:
                        continue
                    dist = float(an.box_signed_distance(h.vertices, np.asarray(centre), np.asarray(rot), size).min())
                    if dist < 0.04 and max(size) < 0.5:  # near: the thing's own surface against the hull's faces too
                        if shell is None:
                            shell = _shell(size) @ np.asarray(rot).T + centre
                        dist = min(dist, float(an._signed(shell, h).min()))
                    if dist < best:
                        best, link = dist, part
                key = (side, labels[side], name)
                if link is not None and best < self.least.get(key, (0.04, None))[0]:
                    self.least[key] = (best, link)
        lo, hi = ({side: np.array([f(h.vertices, axis=0) for _, h in hulls]) for side, hulls in sets.items()} for f in (np.min, np.max))
        apart = np.maximum(lo["left"][:, None, :] - hi["right"][None, :, :], lo["right"][None, :, :] - hi["left"][:, None, :]).max(axis=2)
        key = (labels["left"], labels["right"])
        for i, j in zip(*np.nonzero(apart < 0.04)):  # hulls whose bounding boxes are near each other
            (pl, hl), (pr, hr) = sets["left"][i], sets["right"][j]
            dist = min(float(an._signed(hl.vertices, hr).min()), float(an._signed(hr.vertices, hl).min()))
            if dist < self.arms.get(key, (0.04, None))[0]:
                self.arms[key] = (dist, (pl, pr))

    def table(self, under=0.015):
        for side in ("left", "right"):
            print(f"-- {side} arm: hulls within {under * 1000:.0f} mm of something, by primitive")
            for (s, label, name), (dist, link) in self.least.items():
                if s == side and dist < under:
                    print(f"   {label:44s} {name:10s} {dist * 1000:6.1f} mm  {link}")
        print(f"-- arm to arm, within {under * 1000:.0f} mm")
        for (ll, rl), (dist, pair) in self.arms.items():
            if dist < under:
                print(f"   left '{ll}' / right '{rl}': {dist * 1000:.1f} mm  {pair}")


CLEAR = None
CLEAR_EVERY = 2  # control steps between samples


def push(env, part, force, at=None, body=None):
    WORLD.pushed = bool(np.linalg.norm(force) > 0)


def run(mode, max_steps=4000, hold_s=2.0, clear=False):
    global WORLD, CTX, CLEAR
    WORLD = World()
    CLEAR = Clearance() if clear else None
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(step_dt=DT))
    m.state = lambda e, name: WORLD.state(name)
    m.fingertips = lambda e: WORLD.fingertips()
    m.push, m.set_target = push, (lambda *a, **k: None)
    task.write_pose = lambda *a, **k: None  # the world above starts as the task file says
    task.TRIAL.reset(env)
    CTX = ctx = Ctx(types.SimpleNamespace(_env=env), None, mode)
    ctx.state = lambda name: WORLD.state(name)
    ctx.pos = lambda name: WORLD.state(name).pos
    script = task.demo_script(ctx)
    ended, steps, best = None, 0, 0.0
    while steps < max_steps:
        if ended is None:
            try:
                next(script)
            except StopIteration:
                ended = steps
        WORLD.step(ctx)
        if CLEAR and steps % CLEAR_EVERY == 0:
            CLEAR.sample(WORLD, ctx)
        task.TRIAL.step(env)
        steps += 1
        ctx.t += DT
        best = max(best, task.score(None))
        if task.success(None):
            break
        if ended is not None and steps - ended >= hold_s * d.CONTROL_HZ:
            break
    result = {"score": task.score(None), "best": best, "success": task.success(None), "steps": steps, "ended": ended,
              "ik_mm": max(ctx.left.worst[0], ctx.right.worst[0]), "ik_deg": max(ctx.left.worst[1], ctx.right.worst[1]),
              "zero": task.TRIAL.rule.zero, "notes": list(WORLD.notes), "now": dict(task.TRIAL.rule.now), "tab": WORLD.tab, "on_tab": WORLD.on_tab, "back": WORLD.back,
              "clear": CLEAR}
    print(f"RESULT {mode}: score {task.score(None)} (best on the way {best}), success {task.success(None)}, steps {steps} ({steps * DT:.1f} s), script ended at {ended}, "
          f"IK worst left {ctx.left.worst[0]:.1f} mm {ctx.left.worst[1]:.1f} deg, right {ctx.right.worst[0]:.1f} mm {ctx.right.worst[1]:.1f} deg; "
          f"clip on the tab {WORLD.on_tab}, {WORLD.back * 1000:.1f} mm short; now {task.TRIAL.rule.now}")
    return result


if __name__ == "__main__":
    for mode in (sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["full"]):
        print(f"\n===== {mode}")
        out = run(mode, clear="--clear" in sys.argv)
        if out["clear"]:
            out["clear"].table()
