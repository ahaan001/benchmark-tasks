"""Offline checks, no simulator: the embedding rules and the score, and the scripted demo on a kinematic stand-in.

    PYTHONPATH=<yam-isaac-lab-arena> python3 test_embed_tissue_flat_in_paraffin.py

The stand-in replaces mechkit's calls into Isaac Sim. Each arm reaches its command exactly; the forceps go with the left hand once its closed fingers
are round their grip, and drop into the well when let go over it; the mould slides with the right hand once its fingers are closed either side of it.
The strip stands bowed on the mould's floor, carried by the mould. Tips that come down on it push its crown down as far as they go, to flat at
the lowest, and drag it when they move; when they leave, it bows again over 0.3 s. Tips that do not reach its crown do nothing to it.
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

BOW = math.radians(t.BOW)


def snap(mould=(0.30, 0.065), piece=(0.0, 0.0), tilt=BOW, tips=None, forceps=None) -> ek.Snapshot:
    """The strip standing in the mould with each half tilted `tilt` rad (the rest bow by default; 0 is flat), its fold line `piece` from the mould's middle."""
    g = t.GEOM
    mp = np.array([mould[0], mould[1], g.plate_z])
    pos, rot = ek.bow_pose((mp[0] + piece[0], mp[1] + piece[1]), g.plate_z + g.floor, tilt, g.tissue)
    fp = np.array(forceps) if forceps is not None else np.array([0.25, 0.145, g.plate_z + 0.001])
    far = np.array([[0.15, 0.25, 0.25]])
    return ek.Snapshot(mould_pos=mp, mould_rot=np.eye(3), mould_vel=np.zeros(3), tissue_pos=pos, tissue_rot=rot, fold=g.fold - 2 * tilt, forceps_pos=fp,
                       forceps_rot=np.eye(3), tips=far if tips is None else np.atleast_2d(tips))


FLAT = dict(tilt=0.0)


def run_rules(seq, rules=None):
    rules = rules or ek.Rules(t.GEOM, t.DT)
    held = None
    for s, n in seq:
        for _ in range(n):
            held = rules.step(s)
    return rules, held


def test_rules():
    g = t.GEOM
    cold = ((t.COLD[0] + t.COLD[1]) / 2, (t.COLD[2] + t.COLD[3]) / 2)
    r, held = run_rules([(snap(), 5)])
    depth = r.ink_depth(snap())
    check("a strip just dropped in stands bowed on its two ends: not flat, and nothing is held for it", not r.lies_flat(snap()) and held is None and abs(depth.min()) < 1e-9,
          f"inked face {depth.min() * 1000:.2f} to {depth.max() * 1000:.2f} mm off the floor")
    rise = g.tissue[1] / 2 * math.sin(BOW) - g.tissue[2] * math.sin(BOW) * math.tan(BOW)
    check("at rest the middle of the inked face stands 1.5 to 2.5 mm off the floor", 0.0015 <= rise <= 0.0025, f"{rise * 1000:.2f} mm")
    check("pushed down to the floor, it lies flat", r.lies_flat(snap(**FLAT)))
    check("pushed half way down (its halves still tilted 12 degrees), it does not lie flat", not r.lies_flat(snap(tilt=math.radians(12.0))))
    check("the hinge reads 0 at the rest bow and its full travel when flat", snap().fold == 0.0 and abs(snap(**FLAT).fold - g.fold) < 1e-12)
    w = World()
    crown = g.plate_z + g.floor + ek.crown_height(g.tissue, BOW)
    over = np.array([w.hinge[0], w.hinge[1], crown + 0.0003])
    check("tips 0.3 mm above the bowed strip's crown do not flatten it", w.tilt_under(over, True) == BOW)
    check("tips 2 mm above the floor beside the strip do not flatten it", w.tilt_under(np.array([w.hinge[0] + g.tissue[0] / 2 + 0.0015, w.hinge[1], g.plate_z + g.floor + 0.002]), True) == BOW)
    part = w.tilt_under(np.array([w.hinge[0], w.hinge[1], crown - 0.001]), True)
    check("tips that push the crown 1 mm down leave it bowed", 0.0 < part < BOW and not r.lies_flat(snap(tilt=part)), f"halves tilted {math.degrees(part):.1f} degrees")
    check("tips that push the crown down to the flat strip's top lay it flat", w.tilt_under(np.array([w.hinge[0], w.hinge[1], g.plate_z + g.floor + g.tissue[2]]), True) == 0.0)
    half = (t.M0[0], t.COLD[3] + 0.010)  # the mould's near half over the cold block, its far half still on the strip
    r, _ = run_rules([(snap(**FLAT), 5), (snap(mould=half), 7), (snap(mould=half, **FLAT), 5)])
    check("bowed for 0.3 s with part of the mould over the cold block: the bow is set", r.bow_set)
    r, _ = run_rules([(snap(**FLAT), 5), (snap(mould=half), 4), (snap(mould=half, **FLAT), 5)])
    check("bowed for 0.2 s there, then pushed flat again: not set", not r.bow_set)
    r, _ = run_rules([(snap(), 40)])
    check("bowed on the warm plate, however long: not set", not r.bow_set)
    r, _ = run_rules([(snap(**FLAT), 5), (snap(mould=cold, **FLAT), int(8.0 / t.DT) + 2)])
    on = snap(mould=cold, forceps=(cold[0], cold[1], g.plate_z + g.floor + g.tissue[2]), **FLAT)
    check("held flat on the cold block for 8 s: set, flat, centred (0.75 while the forceps are still on it)", r.set_ and r.flat and r.centred and r.moment(on) == 0.75,
          f"set {r.set_} flat {r.flat} centred {r.centred} moment {r.moment(on)}")
    clean = snap(mould=cold, forceps=(t.WELL[0], t.WELL[1], t.PLATE_Z + 0.001), **FLAT)
    check("... with the forceps in the well and the hands clear: 1.0", r.moment(clean) == 1.0)
    leaning = snap(mould=cold, forceps=(t.WELL[0], t.WELL[1] + 0.009, t.PLATE_Z + 0.001), **FLAT)
    check("... also with the tips leaning on the well's wall, 9 mm off its axis", r.moment(leaning) == 1.0)
    for name, where in (("beside the well", (t.WELL[0] + 0.013, t.WELL[1], t.PLATE_Z + 0.001)), ("above the well's top", (t.WELL[0], t.WELL[1], t.PLATE_Z + t.WELL_H + 0.002))):
        check(f"... but not with the tips {name}: 0.75", r.moment(snap(mould=cold, forceps=where, **FLAT)) == 0.75)
    r, _ = run_rules([(snap(**FLAT), 5), (snap(mould=cold, **FLAT), int(7.0 / t.DT)), (snap(mould=cold), int(1.5 / t.DT) + 2)])
    check("let go on the cold block before the set: the bottom sets round the bow, 0", r.set_ and r.bow_set and not r.flat and r.moment(snap(mould=cold)) == 0.0)
    r, _ = run_rules([(snap(), 25), (snap(mould=(0.30, 0.050)), 2), (snap(**FLAT), 5), (snap(mould=cold, **FLAT), int(8.0 / t.DT) + 2)])
    check("bowed and moved on the warm plate, then pushed flat while molten: sets flat", r.set_ and r.flat)
    r, _ = run_rules([(snap(), 25), (snap(mould=(0.30, 0.050)), 2), (snap(mould=cold), int(8.0 / t.DT) + 2), (snap(mould=cold, **FLAT), 5)])
    check("bowed on the cold block: the bow is set, pushing afterwards does not help, 0", r.bow_set and r.set_ and not r.flat and r.moment(snap(mould=cold)) == 0.0)
    r, _ = run_rules([(snap(**FLAT), 5), (snap(**FLAT), int(20.0 / t.DT))])
    check("held flat but never moved to the cold block: never sets, 0", not r.set_ and r.moment(snap()) == 0.0)
    r, _ = run_rules([(snap(piece=(0.004, 0.0), **FLAT), 5), (snap(mould=cold, piece=(0.004, 0.0), **FLAT), int(8.0 / t.DT) + 2)])
    check("set flat 4 mm off centre: no centre credit", r.set_ and r.flat and not r.centred)
    r, _ = run_rules([(snap(tips=[(0.30, 0.065, t.PLATE_Z + 0.004)], **FLAT), 1)])
    check("a fingertip in the wax: failed for the trial", r.failed)
    r, _ = run_rules([(snap(tips=[(0.30, 0.065, t.PLATE_Z + 0.004)], **FLAT), 1), (snap(**FLAT), 5), (snap(mould=cold, **FLAT), int(8.0 / t.DT) + 2)])
    check("... and it stays failed whatever follows (latched)", r.failed and r.moment(snap(mould=cold)) == 0.0)
    r, _ = run_rules([(snap(tips=[(0.30, 0.065 - 0.0215, t.PLATE_Z + 0.005)], **FLAT), 3)])
    check("fingers outside the mould's walls do not count as touching the wax", not r.failed)
    r.reset()
    check("a reset clears the latch", not r.failed and not r.set_ and not r.bow_set)
    for vals, want in (((True, True, True, True, False), 1.0), ((True, False, True, True, False), 0.0), ((True, True, False, False, False), 0.5),
                       ((False, True, True, True, False), 0.0), ((True, True, True, True, True), 0.0), ((True, False, False, False, False), 0.0), ((True, True, False, True, False), 0.75)):
        check(f"ladder {vals}", ek.score(*vals) == want, str(ek.score(*vals)))
    q = ek.quat_xyzw(ek.rot_x(30.0) @ ek.rot_z(1.0))
    check("rotation to quaternion round trip", np.allclose(m.quat_matrix(q), ek.rot_x(30.0) @ ek.rot_z(1.0), atol=1e-9) and np.allclose(ek.quat_matrix(q), m.quat_matrix(q)))
    pos, rot = t.TISSUE_START_POSE
    start = ek.Snapshot(mould_pos=np.array([t.M0[0], t.M0[1], g.plate_z]), mould_rot=np.eye(3), mould_vel=np.zeros(3), tissue_pos=pos, tissue_rot=rot, fold=0.0,
                        forceps_pos=np.zeros(3), forceps_rot=np.eye(3), tips=np.zeros((1, 3)))
    d0 = r.ink_depth(start)
    check("the strip starts bowed, its ends 0.4 mm above the mould's floor, 6 mm towards the far end", abs(d0.min() - 0.0004) < 1e-9 and np.allclose(r.piece_in_mould(start)[0][:2], t.TISSUE_START))
    usd = t.TISSUE_PART.build()
    check("the strip is written as two bodies on one sprung hinge", all(k in usd for k in ('def Xform "frame"', 'def Xform "fold"', 'def PhysicsRevoluteJoint "fold_joint"', 'uniform token physics:axis = "X"'))
          and dict(t.TISSUE_PART.drives)["fold_joint"][:2] == (t.FOLD_SPRING, t.FOLD_DAMPING))
    hold = t.FOLD_SPRING * t.FOLD * 2 / (g.tissue[1] / 2)
    check("holding it flat takes 0.2 to 0.3 N on the crown (the real strips are gauged at 10 to 50 gram-force)", 0.2 <= hold <= 0.3, f"{hold:.3f} N; weight {2 * t.HALF_MASS * 9.81:.3f} N")
    part_fold = g.fold - 2 * math.asin(ek.FLAT_DEPTH / (g.tissue[1] / 2))
    check("and 0.12 N or more to bring the inked face within the 1 mm that counts as flat", t.FOLD_SPRING * part_fold * 2 / (g.tissue[1] / 2) >= 0.12,
          f"{t.FOLD_SPRING * part_fold * 2 / (g.tissue[1] / 2):.3f} N")


# -- the kinematic stand-in -----------------------------------------------------------------------------------


def tcp(arm):
    return ik.tcp_pose(arm.q, arm.base)


def tilt_for_crown(height: float) -> float:
    """The tilt of the strip's halves (rad) when its crown stands `height` above the floor."""
    lo, hi = 0.0, BOW
    if height >= ek.crown_height(t.TISSUE, hi):
        return hi
    if height <= ek.crown_height(t.TISSUE, lo) + 1e-9:
        return lo
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ek.crown_height(t.TISSUE, mid) < height else (lo, mid)
    return (lo + hi) / 2


class World:
    def __init__(self):
        g = t.GEOM
        self.mould = np.array([t.M0[0], t.M0[1], g.plate_z])
        self.mould_prev = self.mould.copy()
        self.hinge = self.mould[:2] + np.array(t.TISSUE_START)  # the strip's fold line, in plan
        self.tilt = BOW  # of each half: the rest bow
        self.locked = False  # the fold held where it is (the wax set round it)
        self.fixed = False  # the strip set in its block
        self.forceps = t.F_START.copy()
        self.forceps_rot = t.F_REST.copy()
        self.f_attach = None  # forceps pose in the left hand's frame while held
        self.m_attach = None  # mould offset from the right TCP while held
        self.prev_tip = None

    def floor(self) -> float:
        return self.mould[2] + t.GEOM.floor

    def state(self, env, name):
        if name == "tissue":
            pos, rot = ek.bow_pose(self.hinge, self.floor(), self.tilt, t.TISSUE)
            p = pos + np.array([0.0, 0.0, t.TOP])
            return m.PartState(pos=p, quat_xyzw=ek.quat_xyzw(rot), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p, high=p, joints={"fold_joint": t.FOLD - 2 * self.tilt})
        pos, rot, vel = {"mould": (self.mould, np.eye(3), (self.mould - self.mould_prev) / t.DT), "forceps": (self.forceps, self.forceps_rot, np.zeros(3))}[name]
        p = np.asarray(pos, float) + np.array([0.0, 0.0, t.TOP])
        return m.PartState(pos=p, quat_xyzw=ek.quat_xyzw(rot), lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), low=p, high=p, joints={})

    def over_strip(self, tip) -> bool:
        return abs(tip[0] - self.hinge[0]) <= t.TISSUE[0] / 2 + 0.0005 and abs(tip[1] - self.hinge[1]) <= 0.003

    def tilt_under(self, tip_low, held: bool) -> float:
        """The tilt the strip's halves take under the tips' lowest point: the rest bow, unless the tips have come down on its crown."""
        if not held or not self.over_strip(tip_low):
            return BOW
        return tilt_for_crown(tip_low[2] - self.floor())

    def physics(self, ctx):
        g = t.GEOM
        lp, lr = tcp(ctx.left)
        lp = lp - np.array([0.0, 0.0, t.TOP])
        grip_pt = self.forceps + self.forceps_rot @ t.F_GRIP_LOCAL
        l_closed = ctx.left.cmd[6] < 0.5
        if self.f_attach is None and l_closed and np.linalg.norm(lp - grip_pt) < 0.006:
            self.f_attach = (lr.T @ (self.forceps - lp), lr.T @ self.forceps_rot)
        if self.f_attach is not None and not l_closed:
            self.f_attach = None
            if math.hypot(self.forceps[0] - t.WELL[0], self.forceps[1] - t.WELL[1]) < 0.006:
                self.forceps, self.forceps_rot = t.F_START.copy(), t.F_REST.copy()
            else:
                self.forceps = self.forceps.copy()
                self.forceps[2] = g.plate_z + 0.002
        lift = 0.0
        if self.f_attach is not None:
            self.forceps, self.forceps_rot = lp + lr @ self.f_attach[0], lr @ self.f_attach[1]
            corners = np.array([[a * 0.00075, b * 0.001, 0.0] for a in (-1, 1) for b in (-1, 1)]) @ self.forceps_rot.T
            lift = -float(corners[:, 2].min())  # the tilted tips' lowest corner lies this far below their centre
            floor_z = self.floor() if self.in_mould_xy(self.forceps) else g.plate_z
            if self.over_strip(self.forceps):
                stop = ek.crown_height(t.TISSUE, self.tilt) if self.locked else t.TISSUE[2]  # the tips go no lower than the flat strip's top, or a set bow's crown
                floor_z = max(floor_z, self.floor() + stop)
            self.forceps[2] = max(self.forceps[2], floor_z + lift)
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
        self.hinge = self.hinge + moved[:2]  # the strip rides in the mould
        if self.fixed:
            self.prev_tip = None
            return
        tip = self.forceps - np.array([0.0, 0.0, lift])
        held = self.f_attach is not None
        if not self.locked:  # pushed down at once; let go, it bows again within 0.3 s
            self.tilt = min(self.tilt_under(tip, held), self.tilt + BOW * t.DT / 0.3)
        on = held and self.over_strip(tip) and tip[2] <= self.floor() + ek.crown_height(t.TISSUE, self.tilt) + 0.0001
        if on and self.prev_tip is not None:  # dragged by tips that press on it
            self.hinge = self.hinge + ((tip - self.prev_tip) - moved)[:2]
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
    seen = {"fold": 0.0, "flat_steps": 0}

    def write_pose(env, name, pos, rot, body=None):
        w.fixed = t.RULES.set_  # the reset's put-back, or the hold of the set strip

    def lock(env, part, link, stiffness=0.0, effort=0.0):
        w.locked = True

    patches = [(m, "state", w.state), (m, "fingertips", lambda env: w.tips(ctx)), (m, "set_visible", lambda path, vis: None), (m, "lock", lock),
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
            seen["fold"] = max(seen["fold"], t.FOLD - 2 * w.tilt)
            seen["flat_steps"] += w.tilt == 0.0
            if t.success(None):
                success = True
                break
            if ended is not None and steps - ended > 60:
                break
        r = t.RULES
        return {"score": 1.0 if success else t.score(None), "steps": steps, "ended": ended, "set": r.set_, "flat": r.flat, "centred": r.centred, "bow_set": r.bow_set,
                "failed": r.why, "ik_l": ctx.left.worst, "ik_r": ctx.right.worst, "fold": seen["fold"], "flat_steps": seen["flat_steps"]}
    finally:
        for mod, k, v in saved:
            setattr(mod, k, v)


def test_demo():
    want = {"full": 1.0, "none": 0.0, "hover": 0.0, "still": 0.0, "dip": 0.0, "lift": 0.0, "late": 0.0, "nohold": 0.0}
    for mode, w in want.items():
        res = dry_run(mode)
        check(f"demo stand-in, mode {mode}", abs(res["score"] - w) < 1e-9,
              f"score {res['score']} (want {w}); set {res['set']} flat {res['flat']} centred {res['centred']} bow set {res['bow_set']} failed {res['failed'] or '-'}; "
              f"steps {res['steps']}, script ended {res['ended']}; IK worst left {np.round(res['ik_l'], 1)} right {np.round(res['ik_r'], 1)}")
        if mode == "full":
            check("full: within the run limit with room for a lagging arm", res["steps"] < 2400, f"{res['steps']} steps")
            check("full: IK within 5 mm and 3 degrees", max(res["ik_l"][0], res["ik_r"][0]) < 5.0 and max(res["ik_l"][1], res["ik_r"][1]) < 3.0)
            check("full: the tips pushed the fold through its whole travel", abs(res["fold"] - t.FOLD) < 1e-9, f"{math.degrees(res['fold']):.1f} degrees")
        if mode == "late":
            check("late: the bow had set before the second press, which did not flatten it", res["bow_set"] and not res["flat"])
        if mode == "hover":
            check("hover: tips 1.5 mm above the crown never moved the fold, and the bow set on the cold block", res["fold"] == 0.0 and res["flat_steps"] == 0 and res["bow_set"])


if __name__ == "__main__":
    test_rules()
    test_demo()
    print(f"\n{len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
