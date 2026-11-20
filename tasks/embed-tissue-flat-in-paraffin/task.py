"""Embed a thin skin strip flat in paraffin: one hand keeps it pressed to the mould floor with warm forceps while the other slides the mould from the
warm plate onto the cold spot, and the press lasts until the bottom layer of wax has set.

A benchtop paraffin embedding station: a warm work plate (60 C) and, flush beside it across a 1 cm insulating strip, a cold block (-5 C). On the warm
plate a stainless base mould (37 x 24 mm inside) holds molten paraffin and one processed skin strip (10 x 8 x 2 mm, cut face inked black), just
dropped in near one end; warm embedding forceps stand tips-down in their heated well. The usual method (NSCC Histotechnology, "Embedding tissues in
paraffin wax" and "Embedding orientation"; ARUP histology terms): lay the piece cut face down in the middle of the mould, press it flat with the
forceps, move the mould onto the cold spot and keep pressing until the wax at the bottom has set and holds it. The cassette and the top-up follow.

Declared rules (not fluid simulation): thin processed tissue curls; a strip the forceps' tips have not pressed for 0.3 s curls (one end 2 mm up, the
strip tilted 15 degrees, held there), and pressing it again while the wax is molten lays it flat; if the mould moves while it is curled it is left
4 mm behind. Curled on the cold block, the bottom sets round the curl. After 8 s on the cold block the strip is fixed in the block where it is.
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
WELL = (0.25, 0.145)  # the heated forceps well, a square tube 12 mm inside, 45 mm tall
WELL_IN, WELL_H = 0.012, 0.045
# -- the mould (its frame: origin at the centre of its floor's underside; x across, y along)
MOULD_IN = (0.024, 0.037)  # inside, x by y (the common 37 x 24 mm stainless base mould)
WALL, FLOOR, MOULD_H = 0.001, 0.001, 0.009
WAX = 0.006  # molten paraffin depth
M0 = np.array([0.30, 0.065])  # its start on the warm plate
# -- the tissue piece (its frame: origin at its centre; x across, y along, z up; the inked cut face is -z)
TISSUE = (0.008, 0.010, 0.002)
TISSUE_START = (0.0, 0.006)  # in the mould frame: dropped 6 mm towards the far end
# -- the forceps (their frame: origin at the closed tips; +z along them to the hinge end; the legs spread along x)
F_LEN = 0.150
F_GRIP = 0.095  # where the hand holds them, from the tips
F_START = np.array([WELL[0], WELL[1], PLATE_Z + 0.001])

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
    return m.free("mould", (M0[0], M0[1], TOP + PLATE_Z + 0.0002), parts, cols, LOOKS, mass=0.022, friction=(0.2, 0.15), torsional_patch=None)


def _tissue():
    sx, sy, sz = TISSUE
    body = m.box((sx, sy, sz - 0.0004), (0, 0, 0.0002))
    ink = m.box((sx, sy, 0.0004), (0, 0, -sz / 2 + 0.0002))
    start = (M0[0] + TISSUE_START[0], M0[1] + TISSUE_START[1], TOP + PLATE_Z + FLOOR + sz / 2 + 0.0004)
    return m.free("tissue", start, [("body", body, "tissue"), ("ink", ink, "ink")], [m.col_box("tissue_c", TISSUE)], LOOKS, mass=0.002,
                  friction=(0.8, 0.7), torsional_patch=None)


def _forceps():
    # two legs side by side along x, closed at the tips (as squeezed), spread to 10 mm at the grip, joined at the hinge end
    def outline(z):
        w = 0.0015 + (0.010 - 0.0015) * min(z / F_GRIP, 1.0)
        return w

    th = 0.003
    lower = m.Mesh(np.array([[sx * outline(z) / 2, sy * th / 2, z] for z in (0.0, F_GRIP) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]),
                   np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7]]))
    upper = m.box((0.010, th, F_LEN - F_GRIP), (0, 0, (F_LEN + F_GRIP) / 2))
    serr = m.box((0.010, th + 0.0006, 0.025), (0, 0, F_GRIP))  # the serrated finger grip
    cols = [m.col_hull("lower_c", lower), m.col_box("upper_c", (0.010, th, F_LEN - F_GRIP), (0, 0, (F_LEN + F_GRIP) / 2))]
    return m.free("forceps", (F_START[0], F_START[1], TOP + F_START[2]), [("lower", lower, "steel"), ("upper", upper, "steel"), ("grip", serr, "steel")],
                  cols, LOOKS, mass=0.025, friction=(1.2, 1.0), torsional_patch=None, com=(0.0, 0.0, 0.08))


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
                       tissue_pos=world(ti.pos), tissue_rot=ti.rotation(), forceps_pos=world(fo.pos), forceps_rot=fo.rotation(),
                       tips=m.fingertips(env) - np.array([0.0, 0.0, TOP]))


GEOM = ek.Geometry(plate_z=PLATE_Z, cold=COLD, well=WELL, well_h=WELL_H, mould_in=MOULD_IN, mould_h=MOULD_H, floor=FLOOR, tissue=TISSUE,
                   start_mould=(M0[0], M0[1]))
RULES = ek.Rules(GEOM, DT)


def _hold_tissue(env, pose):
    """Make the piece weightless and put it at `pose` (station frame), still: floating, or fixed in the set block."""
    pos, rot = pose
    m.push(env, "tissue", (0.0, 0.0, 0.002 * 9.81))
    ek.write_pose(env, "tissue", pos + np.array([0.0, 0.0, TOP]), rot)


class Trial:
    def __init__(self):
        self.reset()

    def reset(self, env=None):
        RULES.reset()
        self.hold = 0
        self.full = False
        self.score = 0.0
        self.shown_set = None
        self.weightless = False
        if env is not None:
            self.show(env)

    def step(self, env):
        snap = snapshot(env)
        held = RULES.step(snap)
        if held is not None:
            _hold_tissue(env, held)
            self.weightless = True
        elif self.weightless:
            m.push(env, "tissue", (0.0, 0.0, 0.0))
            self.weightless = False
        mo = RULES.moment(snap)
        self.hold = self.hold + 1 if mo >= 0.999 else 0
        self.full = self.full or self.hold >= HOLD_STEPS
        self.score = 0.0 if RULES.failed else (1.0 if self.full else min(mo, 0.95))
        self.show(env)

    def show(self, env):
        if RULES.set_ is self.shown_set:
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
    """Left hand: the forceps. Takes them from the well, tilts them 30 degrees (tips towards the mould, the hand off to its left side), presses the
    piece flat where it lies and slides it to the middle of the mould under the tips, then keeps pressing, following the mould, while the right hand
    pinches the mould by its near end and slides it 10 cm across onto the cold block. Holds the press for 9 s, lifts the forceps, puts them back in
    the well. Reads only poses a camera sees: the mould's, the piece's, the forceps'.
    Failure checks (--mode): none does nothing; lift slides the mould with the forceps lifted off the piece; nohold presses and slides but lifts the
    forceps as soon as the mould arrives; dip lowers the right fingers into the wax; still presses the piece flat but never moves the mould."""
    import demokit as d

    env = ctx.emb._env
    l, r = ctx.left, ctx.right
    mode = ctx.mode
    if mode not in ("full", "none", "lift", "nohold", "dip", "still"):
        raise ValueError(mode)
    for _ in range(10):
        yield
    _report(ctx, env, f"{mode} start: ")
    if mode == "none":
        for _ in range(40):
            yield
        return
    TILT = d.tilted(30.0, -90.0)
    axis = TILT.rot[:, 2]  # the direction the forceps point, from the hand to the tips

    def forceps_tip():
        s = m.state(env, "forceps")
        return world(s.pos)

    def mould_centre():
        s = m.state(env, "mould")
        return world(s.pos)

    def tissue_top():
        s = m.state(env, "tissue")
        return world(s.pos) + np.array([0.0, 0.0, TISSUE[2] / 2])

    def hand_for_tip(tip):
        """Where the left TCP goes so that the forceps' tips are at `tip`."""
        return np.asarray(tip, float) - F_GRIP * axis + np.array([0.0, 0.0, TOP])

    # 1. the forceps from the well (fingers across the legs, closing along x)
    grip_z = F_START[2] + F_GRIP
    yield from l.move((WELL[0], WELL[1], TOP + grip_z + 0.06), grip=d.down(-90.0), t=2.5, label="over the forceps")
    yield from l.move((WELL[0], WELL[1], TOP + grip_z), t=1.0, label="round the forceps")
    yield from l.close(width=0.010, force=15.0, label="take the forceps")
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.07]), t=1.2, label="lift them out of the well")
    start_tip = tissue_top() + np.array([0.0, 0.0, 0.03])
    yield from l.move(hand_for_tip(start_tip), grip=TILT, t=2.5, label="tilted, tips over the piece")
    yield from r.move(mould_centre() + np.array([0.0, -0.010, 0.05 + TOP]), grip=d.down(90.0), t=2.0, label="the right hand over the mould's near end")
    # 2. tips down onto the piece, pressing about 2 mm past its top; then slide it to the middle under the tips
    yield from _press(l, ctx, env, tissue_top, forceps_tip, axis)
    for _ in range(6):
        yield
    _report(ctx, env, "pressed: ")
    yield from _drag_to_centre(l, ctx, env, mould_centre, forceps_tip, tissue_top, axis)
    _report(ctx, env, "centred: ")
    if mode == "lift":
        yield from l.move(l.at() + np.array([0.0, 0.0, 0.03]), t=0.6, label="the forceps lifted off")
    # 3. the right hand pinches the mould's near end; both hands move 10 cm towards -y together, the left following the mould
    pinch = mould_centre() + np.array([0.0, -0.010, 0.0])
    yield from r.move(pinch + np.array([0.0, 0.0, 0.03 + 0.016 + TOP]), t=1.0, label="the right hand down to the mould")
    yield from r.move(pinch + np.array([0.0, 0.0, 0.0025 + 0.013 + TOP]), t=1.0, label="fingers either side of the mould's long walls")
    if mode == "dip":
        yield from r.move(mould_centre() + np.array([0.0, 0.0, 0.004 + 0.013 + TOP]), finger=0.0, t=1.0, label="the right fingers into the wax")
        for _ in range(20):
            yield
        _report(ctx, env, "dipped: ")
        return
    yield from r.close(width=MOULD_IN[0] + 2 * WALL, force=12.0, label="pinch the mould")
    if mode == "still":
        for _ in range(int(12 / DT)):
            yield
        _report(ctx, env, "still: ")
        return
    target_y = (COLD[2] + COLD[3]) / 2
    shift = np.array([0.0, target_y - mould_centre()[1], 0.0])
    offset = forceps_tip() - mould_centre()
    pressing = mode != "lift"

    def follow_target():
        return mould_centre() + offset + (np.array([0.0, 0.0, -0.002]) if pressing else np.array([0.0, 0.0, 0.03])) - F_GRIP * axis + np.array([0.0, 0.0, TOP])

    yield from ctx.together(r.move(r.at() + shift, t=5.0, label="slide the mould onto the cold block"),
                            l.follow(follow_target, t=5.6, speed=0.05, label="the forceps follow the mould, pressing"))
    _report(ctx, env, "on the cold block: ")
    yield from r.open(t=0.4, label="let go of the mould")
    yield from r.move(r.at() + np.array([0.0, -0.03, 0.06]), t=1.0, label="the right hand up and away")
    hold = 9.5 if mode != "nohold" else 0.5
    yield from l.follow(follow_target, t=hold, speed=0.05, label="keep pressing while the wax sets")
    _report(ctx, env, "after the hold: ")
    # 4. forceps back in the well
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.04]), t=0.8, label="lift the forceps")
    yield from l.move((WELL[0], WELL[1], TOP + grip_z + 0.07), grip=d.down(-90.0), t=2.5, label="the forceps upright over the well")
    yield from l.move((WELL[0], WELL[1], TOP + grip_z + 0.004), t=1.5, label="down into the well")
    yield from l.open(t=0.4, label="let go of the forceps")
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.06]), t=1.0, label="the left hand up")
    yield from ctx.together(l.move((0.18, 0.22, TOP + 0.20), grip=d.down(0.0), t=1.5), r.move((0.18, -0.22, TOP + 0.20), grip=d.down(0.0), t=1.5))
    for k in range(40):
        yield
        if k % 20 == 19:
            _report(ctx, env, "end: ")


def _press(l, ctx, env, tissue_top, forceps_tip, axis):
    """Bring the tips down onto the piece's top and 2 mm past it, closed loop on where the tips really are."""
    def goal(tt):
        return tissue_top() + np.array([0.0, 0.0, max(0.02 - 0.015 * tt, -0.002)]) - F_GRIP * axis + np.array([0.0, 0.0, TOP])

    def hand_from_tips():
        return forceps_tip() - F_GRIP * axis + np.array([0.0, 0.0, TOP])

    yield from l.servo(goal, tol=0.0015, timeout=3.0, gain=0.3, read=hand_from_tips, label="tips down onto the piece")


def _drag_to_centre(l, ctx, env, mould_centre, forceps_tip, tissue_top, axis):
    """Slide the pressed piece to the mould's middle under the tips, 2 mm at a time, the tips held 2 mm into its top."""
    def goal(tt):
        tip = forceps_tip()
        c = mould_centre()
        d = np.array([c[0] - tip[0], c[1] - tip[1], 0.0])
        n = float(np.linalg.norm(d))
        step = d if n < 0.002 else d / n * 0.002
        return np.array([tip[0] + step[0], tip[1] + step[1], tissue_top()[2] - 0.002]) - F_GRIP * axis + np.array([0.0, 0.0, TOP])

    def hand_from_tips():
        return forceps_tip() - F_GRIP * axis + np.array([0.0, 0.0, TOP])

    for _ in range(40):
        tip, c = forceps_tip(), mould_centre()
        if np.hypot(tip[0] - c[0], tip[1] - c[1]) < 0.0012:
            break
        yield from l.servo(goal, tol=0.001, timeout=0.15, gain=0.3, read=hand_from_tips, label="slide the piece to the middle")


if os.environ.get("EMBED_PRINT_LAYOUT"):
    print(f"[embed] mould at {M0}, cold block {COLD}, well {WELL}", flush=True)
