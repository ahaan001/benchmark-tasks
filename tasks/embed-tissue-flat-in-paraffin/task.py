"""Embed a thin skin strip flat in paraffin: one hand keeps it pressed to the mould floor with warm forceps while the other slides the mould from the
warm plate onto the cold spot, and the press lasts until the bottom layer of wax has set.

A benchtop paraffin embedding station: a warm work plate (60 C) and, flush beside it across a 1 cm insulating strip, a cold block (-5 C). On the warm
plate a stainless base mould (37 x 24 mm inside) holds molten paraffin and one processed skin strip (10 x 8 x 2 mm, cut face inked black), just
dropped in near one end; warm embedding forceps stand tips-down in their heated well. The usual method (NSCC Histotechnology, "Embedding tissues in
paraffin wax" and "Embedding orientation"; ARUP histology terms): lay the piece cut face down in the middle of the mould, press it flat with the
forceps, move the mould onto the cold spot and keep pressing until the wax at the bottom has set and holds it. The cassette and the top-up follow.

The bow is physics. The strip is two rigid halves on a sprung hinge across its middle: left alone it stands bowed, its two ends on the mould's
floor and its middle about 2 mm off it. It goes down only as far as something pushes its crown against the spring: 0.16 N brings the inked
face within the 1 mm that counts as flat, 0.25 N holds it fully down, and damping adds to that while it moves. Let go, it no longer lies flat
within half a second. Nothing but that contact flattens it.

Declared rules (the wax is not simulated): if the strip stands bowed for 0.3 s while any part of the mould is over the cold block, the bottom sets
round the bow (the hinge is locked where it is). After 8 s on the cold block the strip is fixed in the block as it lies.
Thin samples are held against the mould's floor with forceps until the paraffin has cooled (US patent 8,329,120).

    python demokit.py task.py OUT --mode full
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import embedkit as ek  # noqa: E402
import mechkit as m  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
INSTRUCTION = ("Embed the skin strip that lies in the paraffin-filled mould on the warm plate. With the warm forceps, move it to the middle of the "
               "mould and press it flat, inked face down. Keep it pressed with the forceps while the other hand slides the mould across onto the cold "
               "spot, and keep pressing until the wax at the bottom has set. Then put the forceps back in their well and move both hands away.")
SUCCESS_HOLD_S = 0.0  # the scorer counts its own hold
SCORE_OVER = "end"
DT = 0.05  # s: one control step at 20 Hz (run with -E control_hz=20)
HOLD_STEPS = 20
TOP = TABLE_TOP[SCENE]

# -- the station (station frame, metres above the table)
PLATE_Z = 0.06  # the work surface: warm plate, insulating strip and cold block are flush
WARM = (0.22, 0.38, 0.005, 0.165)  # x0, x1, y0, y1
STRIP = (0.22, 0.38, -0.005, 0.005)
COLD = (0.24, 0.36, -0.075, -0.005)
WELL = (0.25, 0.145)  # the heated forceps well, a square tube 20 mm inside, 45 mm tall
WELL_IN, WELL_H = 0.020, 0.045
# -- the mould (its frame: origin at the centre of its floor's underside; x across, y along)
MOULD_IN = (0.024, 0.037)  # inside, x by y (the common 37 x 24 mm stainless base mould)
WALL, FLOOR, MOULD_H = 0.001, 0.001, 0.009
WAX = 0.006  # molten paraffin depth
M0 = np.array([0.30, 0.065])  # its start on the warm plate
# -- the skin strip: two rigid halves on a sprung hinge along x through the middle of its top face
# (its frame: origin at its centre when flat; x across, y along, z up; the inked cut face is -z)
TISSUE = (0.008, 0.010, 0.002)
TISSUE_START = (0.0, 0.006)  # in the mould frame: dropped 6 mm towards the far end
BOW = 30.0  # degrees each half tilts in the rest bow: the middle of the inked face then stands 1.9 mm off the floor
FOLD = math.radians(2 * BOW)  # the hinge's travel from the rest bow to flat
FOLD_SPRING = 6.0e-4  # N m/rad: holding the strip flat takes 0.25 N (25 gram-force) on its crown; it weighs 0.02 N
FOLD_DAMPING = 1.8e-4  # N m s/rad: let go, it rises past flat in 0.2 to 0.3 s; pushed down in half a second, it resists with another 0.15 N
HALF_MASS = 0.001  # kg a half
HALF_INERTIA = 3e-8  # kg m^2: about ten times a half's own, which the contact solver needs at this size
TISSUE_HINGE = np.array([0.0, 0.0, TISSUE[2] / 2])
TISSUE_START_POSE = ek.bow_pose((M0[0] + TISSUE_START[0], M0[1] + TISSUE_START[1]), PLATE_Z + FLOOR + 0.0004, math.radians(BOW), TISSUE)


def _tissue():
    """The strip as it stands bowed: its near half is the frame, its far half hangs on the hinge, authored turned down to the rest bow (joint
    zero, where the spring rests). The hinge turns 60 degrees to flat and stops there. Both halves have weight; the spring carries them."""
    sx, sy, sz = TISSUE

    def half(y0, y1):
        c = (y0 + y1) / 2
        return m.box((sx, sy / 2, sz - 0.0004), (0, c, 0.0002)), m.box((sx, sy / 2, 0.0004), (0, c, -sz / 2 + 0.0002))

    def bowed(mesh):
        return m.Mesh((mesh.v - TISSUE_HINGE) @ ek.rot_x(-2 * BOW).T + TISSUE_HINGE, mesh.f)

    near_body, near_ink = half(-sy / 2, 0.0)
    far_body, far_ink = (bowed(k) for k in half(0.0, sy / 2))
    far_box = bowed(m.box((sx, sy / 2, sz), (0, sy / 4, 0)))
    far = m.hinge("fold", "X", TISSUE_HINGE, -3.0, 2 * BOW, [("body", far_body, "tissue"), ("ink", far_ink, "ink")], [m.col_hull("fold_c", far_box)],
                  mass=HALF_MASS, spring=FOLD_SPRING, damping=FOLD_DAMPING, friction=0.0, armature=HALF_INERTIA, gravity=True,
                  com=tuple(far_box.v.mean(axis=0)), inertia=(HALF_INERTIA,) * 3)
    pos, rot = TISSUE_START_POSE
    return m.mechanism("tissue", (pos[0], pos[1], TOP + pos[2]), [far], LOOKS, frame_parts=[("body", near_body, "tissue"), ("ink", near_ink, "ink")],
                       frame_colliders=[m.col_box("near_c", (sx, sy / 2, sz), (0, -sy / 4, 0))], friction=(1.5, 1.4), free_mass=HALF_MASS,
                       start_quat=tuple(ek.quat_xyzw(rot)), frame_inertia=(HALF_INERTIA,) * 3)


# -- the forceps (their frame: origin at the closed tips; +z along them to the hinge end; the legs spread along x)
F_LEN = 0.150  # along the forceps
F_GRIP = 0.095  # where the hand holds them, along the forceps from the tips
F_TIP = 0.012  # the tips' bent segment
F_BEND = 50.0  # degrees: held with the body tilted this much, the tips stand vertical
_b = math.radians(F_BEND)
F_BODY = np.array([0.0, math.sin(_b), math.cos(_b)])  # the body's direction from the bend, in the forceps' frame (the tips run along +z)
F_GRIP_LOCAL = np.array([0.0, 0.0, F_TIP]) + (F_GRIP - F_TIP) * F_BODY
F_REST = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(_b), -math.sin(_b)], [0.0, math.sin(_b), math.cos(_b)]])  # in the well: body upright, tips bent
F_START = np.array([WELL[0], WELL[1] + F_TIP * math.sin(_b) / 2, PLATE_Z + 0.001])  # the tips' end, the bend straddling the well's axis

LOOKS = {
    "housing": m.Look(m.rgb(0.80, 0.81, 0.80), roughness=0.5),
    "warm": m.Look(m.rgb(0.72, 0.73, 0.75), metallic=0.6, roughness=0.35),
    "cold": m.Look(m.rgb(0.20, 0.38, 0.70), metallic=0.4, roughness=0.4),
    "strip": m.Look(m.rgb(0.15, 0.15, 0.16), roughness=0.7),
    "steel": m.Look(m.rgb(0.78, 0.79, 0.80), metallic=0.85, roughness=0.25),
    "wax": m.Look(m.rgb(0.93, 0.86, 0.60), roughness=0.15, opacity=0.45),
    "set": m.Look(m.rgb(0.96, 0.95, 0.90), roughness=0.6),
    "tissue": m.Look(m.rgb(0.55, 0.30, 0.22), roughness=0.7),
    "ink": m.Look(m.rgb(0.05, 0.05, 0.06), roughness=0.6),
    "tank": m.Look(m.rgb(0.70, 0.71, 0.73), metallic=0.5, roughness=0.4),
    "panel": m.Look(m.rgb(0.12, 0.12, 0.14), roughness=0.5),
    "lamp": m.Look(m.rgb(0.10, 0.60, 0.20), emissive=m.rgb(0.05, 0.50, 0.12)),
}


def _box(name, x0, x1, y0, y1, z0, z1, look):
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    s = (x1 - x0, y1 - y0, z1 - z0)
    return (name, m.box(s, c), look), m.col_box(f"{name}_c", s, c)


def _station():
    parts, cols = [], []
    for spec in (("housing", 0.20, 0.40, -0.09, 0.18, 0.0, 0.055, "housing"),
                 ("warm", *WARM, 0.055, PLATE_Z, "warm"),
                 ("strip", *STRIP, 0.055, PLATE_Z, "strip"),
                 ("cold", *COLD, 0.040, PLATE_Z, "cold"),
                 ("tank", 0.40, 0.46, 0.02, 0.14, 0.0, 0.20, "tank"),
                 ("panel", 0.20, 0.215, -0.09, 0.18, 0.0, 0.05, "panel")):
        p, c = _box(*spec)
        parts.append(p)
        cols.append(c)
    parts.append(("nozzle", m.cylinder(0.005, 0.035, (0.385, 0.08, 0.15), axis="x"), "steel"))
    parts.append(("lamp", m.box((0.002, 0.008, 0.008), (0.199, 0.10, 0.03)), "lamp"))
    # the forceps well: a square steel tube standing on the warm plate
    wx, wy = WELL
    h, t = WELL_H, 0.002
    a = WELL_IN / 2
    for k, (cx, cy, sx, sy) in enumerate(((wx + a + t / 2, wy, t, WELL_IN + 2 * t), (wx - a - t / 2, wy, t, WELL_IN + 2 * t),
                                         (wx, wy + a + t / 2, WELL_IN, t), (wx, wy - a - t / 2, WELL_IN, t))):
        parts.append((f"well{k}", m.box((sx, sy, h), (cx, cy, PLATE_Z + h / 2)), "steel"))
        cols.append(m.col_box(f"well{k}_c", (sx, sy, h), (cx, cy, PLATE_Z + h / 2)))
    return m.fixed("station", (0.0, 0.0, TOP), parts, cols, LOOKS, friction=(0.3, 0.25))


def _mould():
    ix, iy = MOULD_IN
    ox, oy = ix + 2 * WALL, iy + 2 * WALL
    parts, cols = [], []
    for name, size, c in (("floor", (ox, oy, FLOOR), (0, 0, FLOOR / 2)),
                          ("wall_xp", (WALL, oy, MOULD_H), ((ix + WALL) / 2, 0, MOULD_H / 2)),
                          ("wall_xn", (WALL, oy, MOULD_H), (-(ix + WALL) / 2, 0, MOULD_H / 2)),
                          ("wall_yp", (ix, WALL, MOULD_H), (0, (iy + WALL) / 2, MOULD_H / 2)),
                          ("wall_yn", (ix, WALL, MOULD_H), (0, -(iy + WALL) / 2, MOULD_H / 2))):
        parts.append((name, m.box(size, c), "steel"))
        cols.append(m.col_box(f"{name}_c", size, c))
    parts.append(("wax_liquid", m.box((ix - 0.0002, iy - 0.0002, WAX), (0, 0, FLOOR + WAX / 2)), "wax"))
    parts.append(("wax_set", m.box((ix - 0.0002, iy - 0.0002, 0.002), (0, 0, FLOOR + 0.001)), "set", True))
    return m.free("mould", (M0[0], M0[1], TOP + PLATE_Z + 0.0002), parts, cols, LOOKS, mass=0.022, friction=(0.3, 0.25), torsional_patch=None)


def _forceps():
    """Embedding forceps with bent tips: two flat legs side by side along x, closed at the tips (as squeezed). The tip segment (12 mm) runs along +z
    of the part's frame; the body bends 50 degrees from it towards +y. Each leg's blade is 8 mm wide where the fingers hold it, 2 mm at the tips;
    the pair is 10 mm across there, 1.5 mm at the tips."""
    def section(u):
        return 0.0015 + (0.010 - 0.0015) * u, 0.002 + (0.008 - 0.002) * u

    def segment(p0, p1, u0, u1, direction):
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(direction, e1)
        pts = []
        for p, u in ((p0, u0), (p1, u1)):
            w, t = section(u)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pts.append(p + sx * w / 2 * e1 + sy * t / 2 * e2)
        f = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7]]
        return m.Mesh(np.array(pts), np.array(f))

    bend = np.array([0.0, 0.0, F_TIP])
    tip = segment(np.zeros(3), bend, 0.0, F_TIP / F_GRIP, np.array([0.0, 0.0, 1.0]))
    lower = segment(bend, F_GRIP_LOCAL, F_TIP / F_GRIP, 1.0, F_BODY)
    upper = segment(F_GRIP_LOCAL, bend + (F_LEN - F_TIP) * F_BODY, 1.0, 1.0, F_BODY)
    grip = segment(F_GRIP_LOCAL - 0.0125 * F_BODY, F_GRIP_LOCAL + 0.0125 * F_BODY, 1.03, 1.03, F_BODY)  # the serrated finger grip
    cols = [m.col_hull("tip_c", tip), m.col_hull("lower_c", lower), m.col_hull("upper_c", upper)]
    return m.free("forceps", (F_START[0], F_START[1], TOP + F_START[2]), [("tip", tip, "steel"), ("lower", lower, "steel"), ("upper", upper, "steel"),
                  ("grip", grip, "steel")], cols, LOOKS, mass=0.025, friction=(1.2, 1.0), torsional_patch=None, com=tuple(F_GRIP_LOCAL - 0.01 * F_BODY),
                  start_quat=tuple(ek.quat_xyzw(F_REST)))


STATION = _station()
MOULD = _mould()
TISSUE_PART = _tissue()
FORCEPS = _forceps()
OBJECTS = [STATION, MOULD, TISSUE_PART, FORCEPS]


# -- the rules, every control step ---------------------------------------------------------------------


def world(p) -> np.ndarray:
    return np.asarray(p, float) - np.array([0.0, 0.0, TOP])


def snapshot(env) -> ek.Snapshot:
    mo, ti, fo = (m.state(env, n) for n in ("mould", "tissue", "forceps"))
    return ek.Snapshot(mould_pos=world(mo.pos), mould_rot=mo.rotation(), mould_vel=np.asarray(mo.lin_vel, float),
                       tissue_pos=world(ti.pos), tissue_rot=ti.rotation(), fold=ti.joint("fold"), forceps_pos=world(fo.pos), forceps_rot=fo.rotation(),
                       tips=m.fingertips(env) - np.array([0.0, 0.0, TOP]))


GEOM = ek.Geometry(plate_z=PLATE_Z, cold=COLD, well=WELL, well_in=WELL_IN, well_h=WELL_H, mould_in=MOULD_IN, mould_h=MOULD_H, floor=FLOOR, tissue=TISSUE, fold=FOLD,
                   start_mould=(M0[0], M0[1]))
RULES = ek.Rules(GEOM, DT)
RULES_STEP = [0]


def _put_strip(env, pose):
    """The strip's near half at `pose` (station frame), still; its hinge stays as it is."""
    pos, rot = pose
    ek.write_pose(env, "tissue", pos + np.array([0.0, 0.0, TOP]), rot, body="frame")


class Trial:
    def __init__(self):
        self.reset()

    def reset(self, env=None):
        RULES.reset()
        RULES_STEP[0] = 0
        self.hold = 0
        self.full = False
        self.score = 0.0
        self.shown_set = None
        self.locked = self.weightless = False
        if env is not None:
            _put_strip(env, TISSUE_START_POSE)  # a reset puts the arms back, not a free mechanism's root
            self.show(env)

    def step(self, env):
        snap = snapshot(env)
        held = RULES.step(snap)
        if (RULES.bow_set or RULES.set_) and not self.locked:  # the wax has set round the strip as it stands: its fold no longer moves
            m.lock(env, "tissue", "fold", stiffness=5.0, effort=5.0)
            self.locked = True
        if held is not None:  # fixed in the set block: weightless, and put where the block is, every step
            if not self.weightless:
                for body in ("frame", "fold"):
                    m.push(env, "tissue", (0.0, 0.0, HALF_MASS * 9.81), body=body)
                self.weightless = True
            _put_strip(env, held)
        if os.environ.get("EMBED_DEBUG") and RULES_STEP[0] % int(os.environ["EMBED_DEBUG"]) == 0:
            tips = snap.tips
            local = RULES.in_mould(snap, tips)
            print(f"[embed-dbg] {RULES_STEP[0]} " + ", ".join(f"{k} {v}" for k, v in RULES.report(snap).items())
                  + f"; forceps tip {np.round(snap.forceps_pos * 1000, 1).tolist()}, nearest fingertip in mould mm {np.round(local[np.argmin(np.abs(local).sum(1))] * 1000, 1).tolist()}", flush=True)
        RULES_STEP[0] += 1
        mo = RULES.moment(snap)
        self.hold = self.hold + 1 if mo >= 0.999 else 0
        self.full = self.full or self.hold >= HOLD_STEPS
        self.score = 0.0 if RULES.failed else (1.0 if self.full else min(mo, 0.95))
        self.show(env)

    def show(self, env):
        if RULES.set_ == self.shown_set:
            return
        self.shown_set = RULES.set_
        m.set_visible(f"{m.ENV_PATH}/mould/visual/wax_set", RULES.set_)


TRIAL = Trial()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    return TRIAL.full


def score(objects) -> float:
    return TRIAL.score


def _report(ctx, env, label=""):
    snap = snapshot(env)
    r = RULES.report(snap)
    print(f"[embed] t {ctx.t:5.1f} {label}" + ", ".join(f"{k} {v}" for k, v in r.items()) + f"; score {score(None):.2f}", flush=True)


# -- the scripted demo ---------------------------------------------------------------------------------


def demo_script(ctx):
    """Left hand: the forceps. Takes them from the well, tilts them 50 degrees (tips towards the mould, the hand off to its left side), pushes the
    bowed strip's crown down until it lies flat and slides it to the middle of the mould under the tips, then keeps pressing, following the mould
    and keeping the strip in its middle, while the right hand pinches the mould by its near end and slides it 10 cm across onto the cold block.
    Holds the press for 9.5 s, lifts the forceps, puts them back in the well. Reads only poses a camera sees: the mould's, the strip's, the forceps'.
    Failure checks (--mode): none does nothing; hover does everything with the tips 1.5 mm above the strip's crown, never touching it; lift slides
    the mould with the forceps lifted off the strip; late does the same and then presses again once the mould is on the cold block; nohold
    presses and slides but lifts the forceps as soon as the mould arrives; dip lowers the right fingers into the wax; still presses the strip
    flat but never moves the mould."""
    import demokit as d

    env = ctx.emb._env
    l, r = ctx.left, ctx.right
    mode = ctx.mode
    if mode not in ("full", "none", "hover", "lift", "late", "nohold", "dip", "still"):
        raise ValueError(mode)
    for _ in range(10):
        yield
    _report(ctx, env, f"{mode} start: ")
    if mode == "none":
        for _ in range(40):
            yield
        return
    TILT = d.tilted(50.0, -90.0)

    def forceps_tip():
        s = m.state(env, "forceps")
        return world(s.pos)

    def mould_centre():
        s = m.state(env, "mould")
        return world(s.pos)

    def crown():
        """The middle of the strip's fold line, on its top face: the top of the bow, or the middle of the top face once it lies flat."""
        s = m.state(env, "tissue")
        return world(s.pos) + s.rotation() @ TISSUE_HINGE

    def flat_top():
        """How high the strip's top face lies once it is flat on the mould's floor."""
        return mould_centre()[2] + FLOOR + TISSUE[2]

    def hand_for_tip(tip):
        """Where the left TCP goes so that the forceps' tips are at `tip`, once the hand is tilted (the tip segment then stands upright)."""
        return np.asarray(tip, float) + F_GRIP_LOCAL + np.array([0.0, 0.0, TOP])

    # 1. the forceps from the well (fingers across the legs, closing along x)
    well_grip = F_START + F_REST @ F_GRIP_LOCAL  # the grip of the forceps standing in the well
    yield from l.move(well_grip + np.array([0.0, 0.0, TOP + 0.06]), grip=d.down(-90.0), t=2.5, label="over the forceps")
    yield from l.move(well_grip + np.array([0.0, 0.0, TOP]), t=1.0, label="round the forceps")
    yield from l.close(width=0.010, force=30.0, label="take the forceps")
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.07]), t=1.2, label="lift them out of the well")
    start_tip = crown() + np.array([0.0, 0.0, 0.03])
    yield from l.move(hand_for_tip(start_tip), grip=TILT, t=2.5, label="tilted, tips over the strip")
    # 2. the right hand pinches the mould by its long walls at its near end, and holds it still while the forceps work
    yield from r.move(mould_centre() + np.array([0.0, -0.015, 0.05 + TOP]), grip=d.down(90.0), t=2.0, label="the right hand over the mould's near end")
    pinch = mould_centre() + np.array([0.0, -0.015, 0.0])
    yield from r.move(pinch + np.array([0.0, 0.0, 0.03 + 0.016 + TOP]), t=1.0, label="the right hand down to the mould")
    if mode == "dip":
        yield from r.move(mould_centre() + np.array([0.0, 0.0, 0.004 + 0.013 + TOP]), finger=0.0, t=1.5, label="the right fingers into the wax")
        for _ in range(20):
            yield
        _report(ctx, env, "dipped: ")
        return
    yield from r.move(pinch + np.array([0.0, 0.0, 0.0025 + 0.013 + TOP]), t=1.0, label="fingers either side of the mould's long walls")
    yield from r.close(width=MOULD_IN[0] + 2 * WALL, force=12.0, label="pinch the mould")
    # 3. tips down onto the strip's crown until it lies flat; then, pressing, carry it to the middle under the tips (a closed loop on the tips,
    #    the strip and the mould)
    keep = _KeepPressed(l, mould_centre, crown, forceps_tip, flat_top)
    yield from l.move(hand_for_tip(crown() + np.array([0.0, 0.0, 0.012])), t=1.2, label="tips down to about 1 cm over the strip")
    if mode == "hover":
        yield from _over_crown(l, env, crown, forceps_tip, 0.0015, gain=0.15, timeout=4.0)
        _report(ctx, env, "hovering: ")
    else:
        yield from _press(l, env, crown, forceps_tip, flat_top)
        yield from keep.run(4.0, centre=True)
        _report(ctx, env, "centred, pressing: ")
    if mode in ("lift", "late"):
        yield from l.move(l.at() + np.array([0.0, 0.0, 0.03]), t=0.6, label="the forceps lifted off")
    if mode == "still":
        yield from keep.run(12.0)
        _report(ctx, env, "still: ")
        return
    # 4. the right hand slides the mould 10 cm towards -y onto the cold block; the left keeps pressing, carried with the mould
    target_y = (COLD[2] + COLD[3]) / 2
    shift = np.array([0.0, target_y - mould_centre()[1], 0.0])
    if mode in ("lift", "late", "hover"):
        lifted = l.at().copy()
        start = mould_centre().copy()

        def carried():
            for _ in range(int(5.6 / DT)):
                l._solve(lifted + (mould_centre() - start), l.grip.rot)
                yield

        yield from ctx.together(r.move(r.at() + shift, t=5.0, label="slide the mould onto the cold block"), carried())
    else:
        yield from ctx.together(r.move(r.at() + shift, t=5.0, label="slide the mould onto the cold block"), keep.run(5.6, centre=True, follow=True))
    _report(ctx, env, "on the cold block: ")
    if mode == "late":
        yield from _press(l, env, crown, forceps_tip, flat_top)
        yield from keep.run(9.5)
    elif mode == "nohold":
        yield from keep.run(0.5)
    elif mode not in ("lift", "hover"):  # the right hand keeps the mould still on the cold block
        yield from keep.run(2.0, centre=True)  # the wax is still soft: the strip back to the middle if the move shifted it
        yield from keep.run(7.5)
    else:
        yield from ctx.until(lambda: False, 9.5)
    _report(ctx, env, "after the hold: ")
    yield from r.open(t=0.4, label="let go of the mould")
    yield from r.move(r.at() + np.array([0.0, -0.03, 0.06]), t=1.0, label="the right hand up and away")
    # 4. forceps back in the well
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.04]), t=0.8, label="lift the forceps")
    yield from l.move(well_grip + np.array([0.0, 0.0, TOP + 0.07]), grip=d.down(-90.0), t=2.5, label="the forceps upright over the well")
    tips_to_hand = lambda: l.tcp() - (forceps_tip() + np.array([0.0, 0.0, TOP]))  # noqa: E731  (as the forceps now sit in the hand)
    yield from l.servo(lambda tt: F_START + np.array([0.0, 0.0, TOP + 0.03]) + tips_to_hand(), tol=0.0015, timeout=2.0, gain=0.3,
                       read=lambda: forceps_tip() + np.array([0.0, 0.0, TOP]) + tips_to_hand(), label="the tips over the well")
    yield from l.move(l.at() + np.array([0.0, 0.0, -0.028]), t=1.5, label="down into the well")
    yield from l.open(t=0.4, label="let go of the forceps")
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.06]), t=1.0, label="the left hand up")
    yield from ctx.together(l.move((0.18, 0.22, TOP + 0.20), grip=d.down(0.0), t=1.5), r.move((0.18, -0.22, TOP + 0.20), grip=d.down(0.0), t=1.5))
    for k in range(40):
        yield
        if k % 20 == 19:
            _report(ctx, env, "end: ")


class _KeepPressed:
    """Keeps the forceps' tips pressing on the strip's crown, every control step. With `follow`, the hand moves with the mould (so a slide carries
    it). If the tips have drifted more than 2.5 mm along the strip from its fold line, or to within 1 mm of its side edges, the hand shifts level
    by a quarter of the miss, at most 1.5 mm a step. With `centre`, the hand also moves by a quarter of the strip's offset from the mould's
    middle, at most 1 mm a step: the pressed strip goes with the tips. The height creeps down (0.5 mm a step at most, 4 mm below the first press
    at most) until the tips are 1.5 mm below where the flat strip's top lies: the arm's stiffness then holds a steady press."""

    def __init__(self, arm, mould_centre, crown, forceps_tip, flat_top):
        self.arm, self.mould_centre, self.crown, self.forceps_tip, self.flat_top = arm, mould_centre, crown, forceps_tip, flat_top
        self.floor_z = self.floor_ref = None

    def run(self, seconds: float, centre: bool = False, follow: bool = False):
        from yam_arena import ik
        a = self.arm
        sx = TISSUE[0]
        c = ik.tcp_pose(a.q, a.base)[0]
        last = self.mould_centre()
        if self.floor_z is None:
            self.floor_z = c[2] - 0.004  # the hand never goes more than 4 mm below where the first press left it (relative to the mould)
            self.floor_ref = last[2]
        for _ in range(max(1, round(seconds / DT))):
            mould, strip, tip = self.mould_centre(), self.crown(), self.forceps_tip()
            rel = tip[:2] - strip[:2]
            safe = abs(rel[0]) < sx / 2 - 0.001 and abs(rel[1]) < 0.0025
            move = np.zeros(2)
            if not safe:
                move += np.clip(0.25 * (strip[:2] - tip[:2]), -0.0015, 0.0015)
            elif centre:
                move += np.clip(0.25 * (mould[:2] - strip[:2]), -0.001, 0.001)
            dz = float(np.clip(0.25 * ((self.flat_top() - 0.0015) - tip[2]), -0.0005, 0.0005))  # push the tips towards 1.5 mm into the flat strip's top
            c = c + ((mould - last) if follow else 0.0) + np.array([move[0], move[1], dz])
            c[2] = max(c[2], self.floor_z + (mould[2] - self.floor_ref))
            last = mould
            a._solve(c, a.grip.rot)
            yield


def _over_crown(l, env, crown, forceps_tip, above, gain=0.3, timeout=2.5):
    """The tips to `above` over the strip's crown, closed loop on where the tips are."""
    def goal(tt):
        return crown() + np.array([0.0, 0.0, above]) + m.state(env, "forceps").rotation() @ F_GRIP_LOCAL + np.array([0.0, 0.0, TOP])

    def hand_from_tips():
        return forceps_tip() + m.state(env, "forceps").rotation() @ F_GRIP_LOCAL + np.array([0.0, 0.0, TOP])

    yield from l.servo(goal, tol=0.001, timeout=timeout, gain=gain, read=hand_from_tips, label="tips over the strip's crown")


def _press(l, env, crown, forceps_tip, flat_top):
    """The tips to 3 mm above the bowed strip's crown; then straight down, at 4 mm/s, to 3 mm below where the flat strip's top lies: the bent
    tips stand upright, they push the crown down the 2 mm to the floor, and the arm's own stiffness presses."""
    yield from _over_crown(l, env, crown, forceps_tip, 0.003)
    drop = forceps_tip()[2] - flat_top() + 0.003
    yield from l.move(l.at() + np.array([0.0, 0.0, -drop]), t=max(1.5, drop / 0.004), label="tips down onto the crown")


if os.environ.get("EMBED_PRINT_LAYOUT"):
    print(f"[embed] mould at {M0}, cold block {COLD}, well {WELL}", flush=True)
