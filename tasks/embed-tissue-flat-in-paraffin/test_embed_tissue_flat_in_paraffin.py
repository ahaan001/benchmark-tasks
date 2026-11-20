"""Offline checks, no simulator: the embedding rules and the score, and the scripted demo on a kinematic stand-in.

    PYTHONPATH=<yam-isaac-lab-arena> python3 test_embed_tissue_flat_in_paraffin.py

The stand-in replaces mechkit's calls into Isaac Sim. Each arm reaches its command exactly; the forceps go with the left hand once its closed fingers
are round their grip, and drop into the well when let go over it; the mould slides with the right hand once its fingers are closed either side of it;
the piece goes where the rules hold it, or else lies flat on the mould's floor, carried by the mould and dragged by the tips when they press on it.
It runs the real `demo_script`, the real per-step rules and the real scorer.
"""

from __future__ import annotations

import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import demokit as d  # noqa: E402
import embedkit as ek  # noqa: E402
import mechkit as m  # noqa: E402
import task as t  # noqa: E402
from yam_arena import ik  # noqa: E402
from yam_arena.constants import FINGER_OPEN, TCP_OFFSET  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


# -- the rules on hand-built states ----------------------------------------------------------------------


def snap(mould=(0.30, 0.065), piece=(0.0, 0.0), tip=None, tilt=0.0, mould_vel=(0.0, 0.0, 0.0), tips=None, forceps=None) -> ek.Snapshot:
    g = t.GEOM
    mp = np.array([mould[0], mould[1], g.plate_z])
    tp = mp + np.array([piece[0], piece[1], g.floor + g.tissue[2] / 2])
    rot = ek.rot_x(tilt)
    fp = np.array(forceps) if forceps is not None else (tp + np.array([0.0, 0.0, g.tissue[2] / 2]) if tip == "on" else np.array([0.25, 0.145, g.plate_z + 0.001]))
    far = np.array([[0.15, 0.25, 0.25]])
    return ek.Snapshot(mould_pos=mp, mould_rot=np.eye(3), mould_vel=np.array(mould_vel, float), tissue_pos=tp, tissue_rot=rot, forceps_pos=fp,
                       forceps_rot=np.eye(3), tips=far if tips is None else np.atleast_2d(tips))


def run_rules(seq, rules=None):
    rules = rules or ek.Rules(t.GEOM, t.DT)
    held = None
    for s, n in seq:
        for _ in range(n):
            held = rules.step(s)
    return rules, held


def test_rules():
    cold = ((t.COLD[0] + t.COLD[1]) / 2, (t.COLD[2] + t.COLD[3]) / 2)
    r, held = run_rules([(snap(), 5)])
    check("a strip just dropped in is curled", r.floating and held is not None and held[0][2] > t.PLATE_Z + t.FLOOR + t.TISSUE[2] / 2 + 0.0015)
    r, _ = run_rules([(snap(tip="on"), 5)])
    check("pressed by the tips, it lies flat (not held)", not r.floating)
    r, _ = run_rules([(snap(tip="on"), 5), (snap(), 5)])
    check("unpressed for under 0.3 s, it still lies flat", not r.floating)
    r, _ = run_rules([(snap(tip="on"), 5), (snap(), 7)])
    check("unpressed for over 0.3 s, it curls again", r.floating)
    hover = snap()
    hover.forceps_pos = hover.tissue_pos + np.array([0.0, 0.0, t.TISSUE[2] / 2 + 0.0008])
    r, _ = run_rules([(snap(tip="on"), 5), (hover, 10)])
    check("tips hovering 0.8 mm above the strip do not press it: it curls", r.floating)
    beside = snap()
    beside.forceps_pos = beside.tissue_pos + np.array([t.TISSUE[0] / 2 + 0.0008, 0.0, -t.TISSUE[2] / 2])
    r, _ = run_rules([(snap(tip="on"), 5), (beside, 10)])
    check("tips resting on the floor 0.8 mm beside the strip do not press it: it curls", r.floating)
    flat_end = snap()
    flat_end.forceps_pos = flat_end.tissue_pos + np.array([0.0, -0.004, t.TISSUE[2] / 2])
    r, _ = run_rules([(snap(tip="on"), 5), (flat_end, 10)])
    check("tips on the end that lies flat anyway do not hold the curling end: it curls", r.floating)
    half = (t.M0[0], t.COLD[3] + 0.010)  # the mould's near half over the cold block, its far half still on the strip
    r, _ = run_rules([(snap(tip="on"), 5), (snap(mould=half), 10), (snap(mould=half, tip="on"), 5)])
    check("curled with part of the mould over the cold block: the curl is set", r.curl_set)
    r, _ = run_rules([(snap(tip="on"), 5), (snap(mould=cold, tip="on"), int(8.0 / t.DT) + 2)])
    on = snap(mould=cold, tip="on")
    check("pressed on the cold block for 8 s: set, flat, centred (0.75 while the forceps are still on it)", r.set_ and r.flat and r.centred and r.moment(on) == 0.75,
          f"set {r.set_} flat {r.flat} centred {r.centred} moment {r.moment(on)}")
    clean = snap(mould=cold, forceps=(t.WELL[0], t.WELL[1], t.PLATE_Z + 0.001))
    check("... with the forceps in the well and the hands clear: 1.0", r.moment(clean) == 1.0)
    r, _ = run_rules([(snap(tip="on"), 5), (snap(mould=cold, tip="on"), int(7.0 / t.DT)), (snap(mould=cold), int(1.5 / t.DT) + 2)])
    check("forceps lifted on the cold block before the set: the bottom sets round the curl, 0", r.set_ and r.curl_set and not r.flat and r.moment(snap(mould=cold)) == 0.0)
    r, _ = run_rules([(snap(), 25), (snap(mould_vel=(0.0, -0.02, 0.0)), 2), (snap(tip="on"), 5), (snap(mould=cold, tip="on"), int(8.0 / t.DT) + 2)])
    check("curled and moved, then pressed again while molten: sets flat", r.drifted is False and r.set_ and r.flat)
    r, _ = run_rules([(snap(), 25), (snap(mould_vel=(0.0, -0.02, 0.0)), 2), (snap(mould=cold), int(8.0 / t.DT) + 2), (snap(mould=cold, tip="on"), 5)])
    check("curled on the cold block: the curl is set, pressing afterwards does not help, 0", r.curl_set and r.set_ and not r.flat and r.moment(snap(mould=cold)) == 0.0)
    r, _ = run_rules([(snap(tip="on"), 5), (snap(tip="on"), int(20.0 / t.DT))])
    check("pressed but never moved to the cold block: never sets, 0", not r.set_ and r.moment(snap()) == 0.0)
    r, _ = run_rules([(snap(piece=(0.004, 0.0), tip="on"), 5), (snap(mould=cold, piece=(0.004, 0.0), tip="on"), int(8.0 / t.DT) + 2)])
    check("set flat 4 mm off centre: no centre credit", r.set_ and r.flat and not r.centred)
    r, _ = run_rules([(snap(tip="on", tips=[(0.30, 0.065, t.PLATE_Z + 0.004)]), 1)])
    check("a fingertip in the wax: failed for the trial", r.failed)
    r, _ = run_rules([(snap(tip="on", tips=[(0.30, 0.065, t.PLATE_Z + 0.004)]), 1), (snap(tip="on"), 5), (snap(mould=cold, tip="on"), int(8.0 / t.DT) + 2)])
    check("... and it stays failed whatever follows (latched)", r.failed and r.moment(snap(mould=cold)) == 0.0)
    r, _ = run_rules([(snap(tip="on", tips=[(0.30, 0.065 - 0.0215, t.PLATE_Z + 0.005)]), 3)])
    check("fingers outside the mould's walls do not count as touching the wax", not r.failed)
    r.reset()
    check("a reset clears the latch", not r.failed and not r.set_)
    for vals, want in (((True, True, True, True, False), 1.0), ((True, False, True, True, False), 0.0), ((True, True, False, False, False), 0.5),
                       ((False, True, True, True, False), 0.0), ((True, True, True, True, True), 0.0), ((True, False, False, False, False), 0.0), ((True, True, False, True, False), 0.75)):
        check(f"ladder {vals}", ek.score(*vals) == want, str(ek.score(*vals)))
    q = ek.quat_xyzw(ek.rot_x(30.0) @ ek.rot_z(1.0))
    check("rotation to quaternion round trip", np.allclose(m.quat_matrix(q), ek.rot_x(30.0) @ ek.rot_z(1.0), atol=1e-9))


# -- the kinematic stand-in -----------------------------------------------------------------------------------


def tcp(arm):
    return ik.tcp_pose(arm.q, arm.base)


class World:
    def __init__(self):
        g = t.GEOM
        self.mould = np.array([t.M0[0], t.M0[1], g.plate_z])
        self.mould_prev = self.mould.copy()
        self.piece = self.mould + np.array([t.TISSUE_START[0], t.TISSUE_START[1], g.floor + g.tissue[2] / 2])
        self.piece_rot = np.eye(3)
        self.forceps = t.F_START.copy()
        self.forceps_rot = np.eye(3)
        self.f_attach = None  # forceps pose in the left hand's frame while held
        self.m_attach = None  # mould offset from the right TCP while held
        self.held_pose = None
        self.prev_tip = None

    def state(self, env, name):
        pos, rot, vel = {"mould": (self.mould, np.eye(3), (self.mould - self.mould_prev) / t.DT), "tissue": (self.piece, self.piece_rot, np.zeros(3)),
                         "forceps": (self.forceps, self.forceps_rot, np.zeros(3))}[name]
        p = np.asarray(pos, float) + np.array([0.0, 0.0, t.TOP])
        return m.PartState(pos=p, quat_xyzw=ek.quat_xyzw(rot), lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), low=p, high=p, joints={})

    def physics(self, ctx):
        g = t.GEOM
        lp, lr = tcp(ctx.left)
        lp = lp - np.array([0.0, 0.0, t.TOP])
        grip_pt = self.forceps + self.forceps_rot @ np.array([0.0, 0.0, t.F_GRIP])
        l_closed = ctx.left.cmd[6] < 0.5
        if self.f_attach is None and l_closed and np.linalg.norm(lp - grip_pt) < 0.006:
            self.f_attach = (lr.T @ (self.forceps - lp), lr.T @ self.forceps_rot)
        if self.f_attach is not None and not l_closed:
            self.f_attach = None
            if math.hypot(self.forceps[0] - t.WELL[0], self.forceps[1] - t.WELL[1]) < 0.006:
                self.forceps, self.forceps_rot = np.array([t.WELL[0], t.WELL[1], g.plate_z + 0.001]), np.eye(3)
            else:
                self.forceps = self.forceps.copy()
                self.forceps[2] = g.plate_z + 0.002
        if self.f_attach is not None:
            self.forceps, self.forceps_rot = lp + lr @ self.f_attach[0], lr @ self.f_attach[1]
            floor_z = self.mould[2] + g.floor if self.in_mould_xy(self.forceps) else g.plate_z
            rel = self.forceps - self.piece
            if abs(rel[0]) <= g.tissue[0] / 2 + 0.0005 and abs(rel[1]) <= g.tissue[1] / 2 + 0.0005:
                floor_z = max(floor_z, self.piece[2] + g.tissue[2] / 2 - 0.0005)  # the tips rest on the piece, pressing it
            self.forceps[2] = max(self.forceps[2], floor_z)
        rp, _ = tcp(ctx.right)
        rp = rp - np.array([0.0, 0.0, t.TOP])
        r_closed = ctx.right.cmd[6] < 0.5
        pad_tip = rp[2] - 0.013
        near = abs(rp[0] - self.mould[0]) < 0.004 and abs(rp[1] - self.mould[1]) < 0.02 and self.mould[2] < pad_tip < self.mould[2] + g.mould_h
        if self.m_attach is None and r_closed and near:
            self.m_attach = self.mould[:2] - rp[:2]
        if self.m_attach is not None and not r_closed:
            self.m_attach = None
        self.mould_prev = self.mould.copy()
        if self.m_attach is not None:
            self.mould = np.array([rp[0] + self.m_attach[0], rp[1] + self.m_attach[1], self.mould[2]])
        moved = self.mould - self.mould_prev
        if self.held_pose is not None:
            self.piece, self.piece_rot = self.held_pose
            self.held_pose = None
            self.prev_tip = None
            return
        # lying flat on the floor: carried by the mould, dragged by tips pressing on it
        self.piece = self.piece + moved
        self.piece[2] = self.mould[2] + g.floor + g.tissue[2] / 2
        self.piece_rot = np.eye(3)
        tip = self.forceps
        rel = tip - self.piece
        on = abs(rel[0]) <= g.tissue[0] / 2 + 0.001 and abs(rel[1]) <= g.tissue[1] / 2 + 0.001 and rel[2] <= g.tissue[2] / 2 + 0.001
        if on and self.prev_tip is not None:
            dxy = (tip - self.prev_tip) - moved
            self.piece[:2] += dxy[:2]
        self.prev_tip = tip.copy() if on else None

    def in_mould_xy(self, p):
        ix, iy = t.GEOM.mould_in
        return abs(p[0] - self.mould[0]) < ix / 2 and abs(p[1] - self.mould[1]) < iy / 2

    def tips(self, ctx):
        out = []
        for a in (ctx.left, ctx.right):
            p, rot = tcp(a)
            gp = p - rot @ np.array(TCP_OFFSET)
            opening = abs(a.cmd[6] * FINGER_OPEN)
            for sx in (-1.0, 1.0):
                for dy in (-0.018, 0.0, 0.018):
                    for dz in (0.147, 0.117):
                        out.append(gp + rot @ np.array([sx * (opening + 0.0045), dy, dz]))
        return np.array(out)


class FakeCtx(d.Ctx):
    def __init__(self, mode):
        self.emb = SimpleNamespace(_env="env")
        self.obs = None
        self.mode = mode
        self.left, self.right = d.Arm(self, "left"), d.Arm(self, "right")
        self.t = 0.0

    def observed(self, side):
        a = self.arm(side)
        return np.r_[a.q, a.cmd[6]]


def dry_run(mode: str, max_steps: int = 3600) -> dict:
    w = World()
    ctx = FakeCtx(mode)

    def write_pose(env, name, pos, rot):
        w.held_pose = (np.asarray(pos, float) - np.array([0.0, 0.0, t.TOP]), np.asarray(rot, float))

    patches = [(m, "state", w.state), (m, "fingertips", lambda env: w.tips(ctx)), (m, "set_visible", lambda path, vis: None),
               (m, "push", lambda env, part, force, at=None, body=None: None), (ek, "write_pose", write_pose)]
    saved = [(mod, k, getattr(mod, k)) for mod, k, _ in patches]
    for mod, k, v in patches:
        setattr(mod, k, v)
    try:
        t.TRIAL.reset("env")
        script = t.demo_script(ctx)
        ended, steps, success = None, 0, False
        while steps < max_steps:
            if ended is None:
                try:
                    next(script)
                except StopIteration:
                    ended = steps
            t.TRIAL.step("env")
            w.physics(ctx)
            steps += 1
            ctx.t += t.DT
            if t.success(None):
                success = True
                break
            if ended is not None and steps - ended > 60:
                break
        r = t.RULES
        return {"score": 1.0 if success else t.score(None), "steps": steps, "ended": ended, "set": r.set_, "flat": r.flat, "centred": r.centred,
                "failed": r.why, "ik_l": ctx.left.worst, "ik_r": ctx.right.worst}
    finally:
        for mod, k, v in saved:
            setattr(mod, k, v)


def test_demo():
    want = {"full": 1.0, "none": 0.0, "still": 0.0, "dip": 0.0, "lift": 0.0, "nohold": 0.0}
    for mode, w in want.items():
        res = dry_run(mode)
        check(f"demo stand-in, mode {mode}", abs(res["score"] - w) < 1e-9,
              f"score {res['score']} (want {w}); set {res['set']} flat {res['flat']} centred {res['centred']} failed {res['failed'] or '-'}; "
              f"steps {res['steps']}, script ended {res['ended']}; IK worst left {np.round(res['ik_l'], 1)} right {np.round(res['ik_r'], 1)}")
        if mode == "full":
            check("full: within the run limit with room for a lagging arm", res["steps"] < 2400, f"{res['steps']} steps")
            check("full: IK within 5 mm and 3 degrees", max(res["ik_l"][0], res["ik_r"][0]) < 5.0 and max(res["ik_l"][1], res["ik_r"][1]) < 3.0)


if __name__ == "__main__":
    test_rules()
    test_demo()
    print(f"\n{len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
