"""A kinematic stand-in for the simulator, to shake out the demo and the scorer's wiring without simulator time.

Not physics. The arms follow their commands with a lag. A thing follows a hand whose fingers have closed on it and, let go, drops
straight down onto whatever is under it. The cap's lugs are the task's own rule: once the task switches the joint on, the cap stays on
the vial's axis, goes down as the hand pushes and turns as the hand turns, and a vial that is not held hangs from a held cap. The
tablets stay in the vial. The lid is pushed by closed fingertips that came from behind it and falls shut once past upright; the hasp
goes home as fingertips press it; a dial turns as fingertips are drawn across it. It runs the task's real demo_script, Trial and rule,
so it finds wrong calls, unreachable poses, loops that never end, end states that do not score, and how many control steps a run takes.

    PYTHONPATH=<yam-isaac-lab-arena> python3 dryrun.py [modes]
"""
import importlib.util
import math
import pathlib
import sys
import types

import numpy as np

TASK_DIR = str(pathlib.Path(__file__).resolve().parent)
QUIET = "--quiet" in sys.argv
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import demokit as d  # noqa: E402
import mechkit as m  # noqa: E402
import tidy_geometry as g  # noqa: E402
import tidy_rule as r  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN, READY  # noqa: E402

spec = importlib.util.spec_from_file_location("tidy_task", f"{TASK_DIR}/task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
TOP = task.TOP
UP = np.array([0.0, 0.0, TOP])
DT = 1.0 / d.CONTROL_HZ
LAG, FINGER_LAG = 0.35, 0.5
BOXO = task.BOXO + UP  # the lock box's frame, in the world
LID_REACH = float(np.linalg.norm(g.lid_top_point(0.0) - g.LID_HINGE))
HALF = {"vial": g.VIAL_RO, "cap": g.CAP_R, "block": g.BLOCK / 2, "brick_red": g.BRICK[1] / 2, "brick_yellow": g.BRICK[1] / 2}
TALL = {"vial": g.VIAL_H, "cap": g.CAP_H, "block": g.BLOCK, "brick_red": g.BRICK[2], "brick_yellow": g.BRICK[2]}


class State:
    """Stands in for a body's or the lock box's state."""

    def __init__(self, pos, rot, joints=None):
        self.pos, self.rot, self.joints = np.asarray(pos, float), np.asarray(rot, float), joints or {}

    def rotation(self):
        return self.rot

    def deg(self, link):
        return math.degrees(self.joints[f"{link}_joint"])


def yaw_of(rot) -> float:
    return math.degrees(math.atan2(rot[1, 0], rot[0, 0]))


def rot_z(deg: float) -> np.ndarray:
    return g.rot_axis((0, 0, 1), deg)


class World:
    def __init__(self):
        self.q = {s: np.array(READY, float) for s in ("left", "right")}
        self.f = {s: FINGER_OPEN for s in ("left", "right")}
        self.body = {n: [np.array(spec.pos, float), np.eye(3)] for n, spec in (("vial", task.VIAL), ("cap", task.CAP), ("block", task.BLOCK),
                                                                               ("brick_red", task.BRICK_RED), ("brick_yellow", task.BRICK_YELLOW))}
        self.tablets = np.array(g.tablet_starts())  # in the vial's frame
        self.held = {}  # name -> (side, position in the hand's frame, rotation in the hand's frame)
        self.thread_on, self.locked, self.rise, self.cap_yaw = task.START == "cap_on", False, r.CAP_RISE, 0.0
        self.lid = 0.0 if task.START == "lid_down" else g.LID_OPEN_DEG
        self.hasp, self.caught, self.dials = g.HASP_REST_DEG, False, [0.0, 0.0, 0.0]
        self.behind, self.last_tip_z, self.notes = {}, {}, []

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)
            if not QUIET:
                print(f"   [world] {text}", flush=True)

    # -- what the task reads ----------------------------------------------------------------------------------------------
    def tcp(self, side):
        return ik.tcp_pose(self.q[side], np.asarray(station.BASES[side], float))

    def tips(self, side):
        pos, rot = self.tcp(side)
        return pos + rot[:, 2] * 0.013

    def state(self, name):
        if name.startswith("tablet"):
            pos, rot = self.body["vial"]
            return State(pos + rot @ self.tablets[int(name[6:])], rot)
        return State(*self.body[name])

    def box_state(self):
        return State(BOXO, np.eye(3), {"lid_joint": math.radians(self.lid - g.LID_OPEN_DEG),
                                       "hasp_stop_joint": math.radians(min(self.hasp, g.HASP_REST_DEG) - g.HASP_REST_DEG),
                                       "hasp_joint": math.radians(max(self.hasp - g.HASP_REST_DEG, 0.0)),
                                       **{f"dial{k}_joint": math.radians(a) for k, a in enumerate(self.dials)}})

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
        self.grasps()
        self.carry()
        self.lugs()
        self.lid_hasp_dials()

    def grasps(self):
        for name, (pos, rot) in self.body.items():
            if name in self.held:
                side = self.held[name][0]
                if abs(self.f[side]) > HALF[name] + 0.003:
                    del self.held[name]
                    self.let_go(name)
                continue
            for side in ("left", "right"):
                hand_pos, hand = self.tcp(side)
                rel = rot.T @ (hand_pos - pos)  # the TCP in the thing's frame
                off = float(np.hypot(rel[0], rel[1]))
                top = g.CAP_SEAT_Z - 0.002 if name == "vial" and self.thread_on else TALL[name] + 0.008  # a cap that is on covers the vial's top
                if abs(self.f[side]) <= HALF[name] + 0.0005 and off < 0.008 and 0.004 < rel[2] < top and name not in self.held:
                    self.held[name] = (side, hand.T @ (pos - hand_pos), hand.T @ rot)
                    self.note(f"{side} hand holds the {name}: TCP {1000 * rel[2]:.0f} mm up it, {1000 * off:.1f} mm off its axis")

    def support(self, xy) -> float:
        """The height (world) of what is under a point."""
        p = np.array([[xy[0], xy[1], 0.01]])
        if g.IN_BASKET_XY(p)[0]:
            return TOP + g.BASKET_WALL
        if g.IN_LOCKBOX_XY(p)[0]:
            return TOP + (g.BOX_FLOOR if self.lid > 45.0 else g.BOX_H)
        if g.ON_MAT_XY(p)[0]:
            return TOP + g.MAT[2]
        return TOP

    def let_go(self, name):
        if name == "cap" and self.thread_on:
            return  # it stays on the vial
        pos, rot = self.body[name]
        if name == "vial" and self.thread_on and "cap" in self.held:
            return  # it hangs from the held cap
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, rot[2, 2]))))
        z = self.support(pos[:2])
        self.body[name] = [np.array([pos[0], pos[1], z]), rot_z(yaw_of(rot)) if tilt < 45.0 else rot]
        where = "the toy basket" if g.IN_BASKET_XY(pos[None, :])[0] else "the lock box" if g.IN_LOCKBOX_XY(pos[None, :])[0] else "the mat or the table"
        self.note(f"the {name} is let go {1000 * (pos[2] - z):.0f} mm up and comes down in {where}")

    def carry(self):
        for name, (side, p, rr) in self.held.items():
            hand_pos, hand = self.tcp(side)
            self.body[name] = [hand_pos + hand @ p, hand @ rr]

    def lugs(self):
        """The cap on the vial once the task has switched the lugs' joint on."""
        if not self.thread_on:
            return
        vial_pos, vial_rot = self.body["vial"]
        cap_pos, cap_rot = self.body["cap"]
        axis = vial_rot[:, 2]
        if "cap" in self.held:
            if "vial" in self.held:
                want = float((cap_pos - (vial_pos + axis * g.CAP_SEAT_Z)) @ axis)
                self.rise = 0.0 if self.locked else min(max(want, 0.0), 2 * r.CAP_RISE)
            else:  # the vial hangs from the cap
                self.rise = 0.0 if self.locked else r.CAP_RISE
                vial_pos = cap_pos - axis * (g.CAP_SEAT_Z + self.rise)
                floor = self.support(vial_pos[:2])
                if vial_pos[2] < floor:
                    vial_pos = np.array([vial_pos[0], vial_pos[1], floor])
                self.body["vial"] = [vial_pos, vial_rot]
            if not self.locked:
                self.cap_yaw = yaw_of(vial_rot.T @ cap_rot)
        elif not self.locked:
            self.rise = r.CAP_RISE
        self.body["cap"] = [vial_pos + axis * (g.CAP_SEAT_Z + self.rise), vial_rot @ rot_z(self.cap_yaw)]

    def lid_hasp_dials(self):
        hinge = BOXO + g.LID_HINGE
        hasp_hinge = BOXO + g.HASP_HINGE
        pushed = None
        for side in ("left", "right"):
            tips = self.tips(side)
            last_z, self.last_tip_z[side] = self.last_tip_z.get(side, tips[2]), tips[2]
            if abs(self.f[side]) > 0.004:
                continue  # only closed fingers push
            rel = tips - hinge
            rad = math.hypot(rel[0], rel[2])
            if not self.caught and abs(rel[1]) <= g.BOX_W / 2 + 0.01 and 0.03 < rad <= LID_REACH + 0.005:
                ang = math.degrees(math.atan2(rel[2], -rel[0]))  # the tips' angle above shut, seen from the hinge
                face = ang - math.degrees(math.asin(min(1.0, (g.BOX_H - g.BOX_RIM_Z) / rad))) - math.degrees(0.008 / rad)  # the lid's top is 28 mm off the hinge line
                was_behind = self.behind.get(side, False)
                self.behind[side] = face >= self.lid - 1e-6 or (was_behind and face >= self.lid - 15.0)
                if self.lid > 2.0 and was_behind and face < self.lid:
                    pushed = face if pushed is None else min(pushed, face)
            if self.lid <= r.LID_SHUT_DEG and not self.caught:  # the hasp hangs in front of the lock: fingertips press it home
                lever = hasp_hinge[2] - tips[2]
                if abs(tips[1] - hasp_hinge[1]) <= g.HASP[0] / 2 + 0.006 and 0.008 <= lever <= g.HASP[1]:
                    gap = (hasp_hinge[0] - g.HASP[2]) - (tips[0] + 0.004)  # from the hasp's home face to the fingertips' front
                    if gap > -0.02:
                        self.hasp = min(self.hasp, max(0.0, math.degrees(math.asin(min(1.0, max(gap, 0.0) / lever)))))
            for k in range(3):  # fingertips drawn across a dial turn it
                at = BOXO + g.dial_at(k)
                if tips[0] + 0.004 >= at[0] - g.DIAL_R - 0.001 and abs(tips[1] - at[1]) <= g.DIAL_W / 2 + 0.006 and abs(tips[2] - at[2]) <= 0.014:
                    self.dials[k] += math.degrees((last_z - tips[2]) / g.DIAL_R)
                    self.note(f"{side} fingertips are on dial {k}")
        if self.caught:
            self.lid, self.hasp = 0.0, 0.0
            return
        if pushed is not None:
            self.lid = max(pushed, 0.0)
        elif g.LID_OPEN_DEG - 0.01 > self.lid > 90.0:
            self.lid = min(g.LID_OPEN_DEG, self.lid + 6.0)  # short of upright and let go: it falls back open
        if 0.0 < self.lid <= 88.0:
            self.lid = max(0.0, self.lid - 25.0)  # past upright it falls shut, away from the fingers behind it
            if self.lid == 0.0:
                self.note("the lid has fallen shut; the hasp hangs in front of the lock")
        if self.lid > r.LID_SHUT_DEG:
            self.hasp = g.HASP_REST_DEG


class Ctx(d.Ctx):
    def observed(self, side):
        return np.r_[WORLD.q[side], WORLD.f[side] / FINGER_OPEN]


def thread_on(name, on):
    WORLD.thread_on = bool(on)
    if not on:
        WORLD.locked = False
    WORLD.note(f"the task switches the cap's lugs {'on' if on else 'off'}")


def slide_target(name, x):
    if x > 0.0 and WORLD.thread_on:
        WORLD.locked = True
        WORLD.note("the task locks the cap on its seat")


def hold(env, part, link, target, stiffness, effort):
    if link == "hasp":
        WORLD.caught = True
        WORLD.note("the task holds the hasp in the lock, and the lid with it")


def unlock(env, part, link):
    WORLD.caught = False


def run(mode, max_steps=4000, hold_s=2.0):
    global WORLD, CTX
    WORLD = World()
    joints = types.SimpleNamespace(torch=types.SimpleNamespace(clone=lambda: 0.0))
    box = types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=joints, default_joint_vel=joints), write_joint_position_to_sim_index=lambda position: None,
                                write_joint_velocity_to_sim_index=lambda velocity: None)  # the world above starts as TIDY_START says
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(step_dt=DT, scene={"lock_box": box}))
    task.states = lambda e, names: {n: WORLD.state(n) for n in names}
    m.state = lambda e, name: WORLD.box_state()
    m.fingertips = lambda e: WORLD.fingertips()
    m.hold, m.unlock = hold, unlock
    task.b.set_thread_on, task.b.set_slide_target, task.b.set_turn_damping = thread_on, slide_target, lambda name, x: None
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
              "lid": WORLD.lid, "hasp": WORLD.hasp, "dials": list(WORLD.dials), "locked": WORLD.locked, "harmed": task.TRIAL.rule.harmed, "notes": list(WORLD.notes),
              "now": dict(task.TRIAL.rule.now)}
    print(f"RESULT {mode}: score {task.score(None)} (best on the way {best}), success {task.success(None)}, steps {steps} ({steps * DT:.1f} s), script ended at {ended}, "
          f"IK worst left {ctx.left.worst[0]:.1f} mm {ctx.left.worst[1]:.1f} deg, right {ctx.right.worst[0]:.1f} mm {ctx.right.worst[1]:.1f} deg; "
          f"lid {WORLD.lid:.1f} hasp {WORLD.hasp:.1f} dials {np.round(WORLD.dials, 0)}; cap locked {WORLD.locked}; now {task.TRIAL.rule.now}")
    return result


if __name__ == "__main__":
    for mode in (sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["full"]):
        print(f"\n===== {mode}")
        run(mode)
