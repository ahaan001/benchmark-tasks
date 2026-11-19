"""Offline tests: the weld rule against hand-built traces (the full sequence, doing nothing and the red-team cases), the latches, the geometry and the
field limits. NumPy only for the rule; the reach test needs yam_arena on the path.

    PYTHONPATH=../yam-isaac-lab-arena python3 tasks/weld-blade-waxes-into-a-casting-cluster/test_weld_blade_waxes.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import cluster as c  # noqa: E402
import weld_rule as w  # noqa: E402

DT = 0.05
FAILED = []
AWAY = np.tile([0.0, 0.0, 0.6], (24, 1))  # both hands far above everything


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) and not isinstance(want, bool) else got == want
    print(f"  {'ok ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILED.append(label)


def knife_at(k: int, inserted: bool, side: int | None = None, dy: float = 0.0, dz: float = 0.0):
    """The knife level, its blade between gate end and sprue at slot k (inserted) or 8 cm out to the side it comes from."""
    s = c.SLOT_FROM[k] if side is None else side
    R = np.eye(3) if s > 0 else c.rot_z(180.0)
    p = np.array([c.CX + (0.0 if inserted else s * 0.08), c.FACE_Y - c.BLADE[1] / 2 + dy, c.SLOTS_Z[k] + dz])
    return R, p


KNIFE_HOME = (c.KNIFE_REST_R, c.KNIFE_REST.copy())


def gap_pose(k: int, gap: float):
    """A pattern square on slot k with its gate end `gap` off the sprue's face."""
    R, p = c.slot_pose(k)
    return R, p - np.array([0.0, gap, 0.0])


def holding_tips(pose):
    """The right hand's fingers either side of a pattern's platform (closing along x), the left hand away."""
    R, p = pose
    plat = c.to_world(R, p, [[0.0, c.Y_PLATFORM, 0.0]])[0]
    tips = AWAY.copy()
    k = 12
    for sx in (-1.0, 1.0):
        for dy in (-0.009, 0.0, 0.009):
            for dz in (0.0, 0.03):
                tips[k] = plat + np.array([sx * (c.PLATFORM[0] / 2 + 0.0045), dy, dz])
                k += 1
    return tips


class Trace:
    def __init__(self):
        self.rule = w.Rule()
        self.t = 0.0
        self.loose = [c.loose_start(i) for i in range(3)]
        self.neigh = [(r, p) for _, p, r in c.NEIGHBOURS]
        self.knife = KNIFE_HOME
        self.tips = AWAY.copy()
        self.opening = (0.04, c.PLATFORM[0] / 2)

    def obs(self):
        return w.Obs(self.t, self.knife, list(self.loose), list(self.neigh), self.tips, self.opening)

    def run(self, seconds: float, **set_):
        for k, v in set_.items():
            setattr(self, k, v)
        for _ in range(max(1, round(seconds / DT))):
            self.rule.step(self.obs())
            self.t += DT
        return self

    def score(self):
        return self.rule.score(self.obs())

    def weld(self, i: int, k: int, dose: float = 3.0, hold: float = 11.0, release: bool = True, side: int | None = None):
        """The real sequence for pattern i at slot k: gate held 0.8 mm off the face, blade between for `dose`, blade out, gate on, held, let go."""
        g = gap_pose(k, c.BLADE[1] + 0.0002)
        self.loose[i] = g
        self.run(0.5, tips=holding_tips(g), knife=knife_at(k, False, side))
        self.run(dose, knife=knife_at(k, True, side))
        self.run(0.5, knife=knife_at(k, False, side))
        on = c.slot_pose(k)
        on = (on[0], on[1] + np.array([0.0, 0.0001, 0.0]))
        self.loose[i] = on
        self.run(hold, tips=holding_tips(on))
        for idx in list(self.rule.locks):
            self.rule.locks.remove(idx)
        if release:
            self.run(0.5, tips=AWAY.copy())
        return self


def full():
    tr = Trace()
    for i in range(3):
        tr.weld(i, i)
    tr.run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    return tr


def test_ladder():
    print("ladder")
    check("full sequence, knife home, hands clear 1 s", full().score(), 1.0)
    check("doing nothing", Trace().run(30.0).score(), 0.0)
    tr = Trace()
    tr.weld(0, 0).run(2.0, tips=AWAY.copy())
    check("one pattern welded, knife left out", tr.score(), 0.25 + 0.10)
    tr = Trace()
    tr.weld(0, 0)
    tr.run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    check("one pattern welded, knife home, clear", tr.score(), 0.50)
    tr = Trace().run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    check("knife home and hands clear with nothing welded", tr.score(), 0.0)


def test_red_team():
    print("red team (prompt 00, 8.2)")
    tr = Trace()  # hover: blade in the gap, 2 mm off both faces
    g = gap_pose(0, 0.005)
    tr.loose[0] = g
    tr.run(5.0, tips=holding_tips(g), knife=knife_at(0, True, dy=-0.002))
    tr.loose[0] = c.slot_pose(0)
    tr.run(11.0)
    tr.run(1.0, tips=AWAY.copy())
    check("hover: blade between, touching neither face", tr.rule.joints[0].welded, False)
    tr = Trace()  # one face only: the blade on the sprue, the gate 3 mm back
    g = gap_pose(0, 0.004)
    tr.loose[0] = g
    tr.run(10.0, tips=holding_tips(g), knife=knife_at(0, True))
    check("blade on the sprue only, 10 s: no heat in the gate", tr.rule.joints[0].dose, 0.0)
    tr = Trace()  # the right action at the wrong place: the blade high, over the gate's top edge only
    g = gap_pose(0, c.BLADE[1] + 0.0002)
    tr.loose[0] = g
    tr.run(5.0, tips=holding_tips(g), knife=knife_at(0, True, dz=0.0095))
    check("blade not over the middle of the gate face: no heat", tr.rule.joints[0].dose, 0.0)
    tr = Trace()  # contact in a single step
    tr.loose[0] = gap_pose(0, c.BLADE[1] + 0.0002)
    tr.run(0.05, knife=knife_at(0, True))
    tr.run(0.5, knife=knife_at(0, False))
    tr.loose[0] = c.slot_pose(0)
    tr.run(11.0, tips=holding_tips(tr.loose[0]))
    check("one step of contact: no weld", tr.rule.joints[0].welded, False)
    tr = Trace()  # wrong order: gate pressed on first, then the knife: the blade cannot get between, nothing heats
    tr.loose[0] = c.slot_pose(0)
    tr.run(5.0, tips=holding_tips(tr.loose[0]), knife=knife_at(0, True, dy=-0.0009))
    tr.run(11.0, knife=knife_at(0, False))
    check("gate on first, knife after: no weld", tr.rule.joints[0].welded, False)
    tr = Trace()
    tr.weld(0, 0, hold=11.0, release=True)
    tr.loose[0] = (tr.loose[0][0], tr.loose[0][1])
    check("the real sequence welds", tr.rule.joints[0].welded, True)


def test_set_and_dose():
    print("setting and dose")
    tr = Trace()
    tr.weld(0, 0, hold=4.0, release=False)
    tr.loose[0] = (tr.loose[0][0], tr.loose[0][1] - np.array([0.0, 0.0, 0.08]))  # let go early: it drops
    tr.run(8.0, tips=AWAY.copy())
    check("let go after 4 s: no weld", tr.rule.joints[0].welded, False)
    tr = Trace()
    tr.weld(0, 0, hold=9.0, release=False)
    tr.loose[0] = (tr.loose[0][0], tr.loose[0][1] + np.array([0.0, 0.0, 0.003]))
    tr.run(3.0)
    check("moved 3 mm at 9 s: cracked, cold", (tr.rule.joints[0].welded, tr.rule.joints[0].dose), (False, 0.0))
    tr = Trace()
    tr.weld(0, 0, dose=9.0)
    tr.run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    check("dose 9 s: welded but the gate melted back", tr.score(), 0.25 - 0.10 + 0.25)
    tr = Trace()
    tr.weld(0, 0, dose=1.5)
    check("dose 1.5 s: too cold to close", tr.rule.joints[0].welded, False)
    tr = Trace()  # blade out, then a wait past T_OPEN before closing
    g = gap_pose(0, c.BLADE[1] + 0.0002)
    tr.loose[0] = g
    tr.run(3.0, tips=holding_tips(g), knife=knife_at(0, True))
    tr.run(5.0, knife=knife_at(0, False))
    tr.loose[0] = c.slot_pose(0)
    tr.run(11.0, tips=holding_tips(tr.loose[0]))
    check("closed 5 s after the blade left: cold", tr.rule.joints[0].welded, False)


def test_latches():
    print("latches")
    tr = full()
    R, p = tr.neigh[0]
    tr.neigh[0] = (R, p + np.array([0.003, 0.0, 0.0]))
    tr.run(0.1)
    tr.neigh[0] = (R, p)
    tr.run(2.0)
    check("a neighbour knocked 3 mm and put back: 0", tr.score(), 0.0)
    tr = full()
    tr2 = Trace()
    tr2.run(0.1, opening=(0.04, c.PLATFORM[0] / 2 - 0.0008), tips=holding_tips(tr2.loose[1]))
    tr2.run(2.0, opening=(0.04, 0.04), tips=AWAY.copy())
    check("fingers closed 1.6 mm past a platform: crushed, 0", (tr2.rule.crush, tr2.score()), (True, 0.0))
    tr = Trace()
    tr.weld(0, 0)
    Rk, pk = knife_at(1, False)
    tr.run(0.2, knife=(Rk, np.array([c.CX - 0.008, c.FACE_Y + c.Y_ROOT, c.SLOTS_Z[0]])))  # the blade on the welded pattern's root
    tr.run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    check("blade on a welded pattern's root: minus 0.10", tr.score(), 0.25 + 0.15 + 0.10 - 0.10)
    tr = Trace()
    tr.run(0.2, knife=(c.KNIFE_REST_R, c.NEIGHBOURS[0][1] + np.array([0.0, c.Y_PLATFORM, 0.011 + 0.0065])))  # on the lower neighbour's platform top
    check("blade on a neighbour's platform: marked", tr.rule.neighbour_marked[0], True)
    tr = full()
    tips = AWAY.copy()
    tips[3] = c.to_world(*KNIFE_HOME, [[0.0, 0.002, 0.0]])[0]
    tr.run(0.1, tips=tips)
    tr.run(2.0, tips=AWAY.copy())
    check("the blade against a finger pad: 0", tr.score(), 0.0)
    tr = full()
    tr.run(0.0)
    tr.t = 0.0  # the clock goes back: a real reset
    tr.run(0.05)
    check("a new episode clears the memory", (tr.rule.joints[0].welded, tr.rule.knock), (False, False))
    tr = Trace()
    tr.weld(0, 0, release=False)
    tr.run(2.0, knife=KNIFE_HOME)
    check("welded but still held at the end: that pattern 0", tr.score(), 0.0)
    tr = Trace()
    tr.weld(0, 1)  # the lowest pattern welded at the middle mark: fine, it is on a mark
    tr.weld(1, 1)  # a second one onto the same mark cannot score twice
    tr.run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    check("two patterns on one mark score once", tr.score(), 0.25 + 0.15 + 0.10)
    tr = Trace()
    g = c.slot_pose(0)
    tr.weld(0, 0)
    tr.loose[0] = (c.rot_y(4.0), tr.loose[0][1])
    tr.rule.joints[0].weld_pose = tr.loose[0]
    tr.run(2.0, knife=KNIFE_HOME, tips=AWAY.copy())
    check("welded 4 degrees out of square: 0 for it", tr.score(), 0.0)


def test_geometry():
    print("geometry")
    pts = []
    for pose in [c.slot_pose(k) for k in range(3)] + [(r, p) for _, p, r in c.NEIGHBOURS]:
        corners = []
        for _, ctr, half, rot in w.MARKED:
            for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                corners.append(ctr + rot @ (s * half))
        pts.append(c.to_world(*pose, corners))
    gaps = [float(np.min(np.linalg.norm(pts[a][:, None] - pts[b][None], axis=2))) for a in range(6) for b in range(a + 1, 6)]
    print(f"  closest two patterns (roots, platforms, airfoils): {min(gaps) * 1000:.1f} mm")
    check("every two patterns on the finished cluster at least 10 mm apart, gates aside (Baosheng: 10 to 15 mm)", min(gaps) >= 0.010, True)
    low = min(float(p[:, 2].min()) for p in pts)
    check("the lowest pattern clears the stand", low > c.STAND[2], True)
    check("the top mark is on the sprue", c.SLOTS_Z[-1] + c.GATE[2] / 2 < c.SPRUE_Z1, True)
    for k in range(3):
        R, p = knife_at(k, True)
        blade = c.to_world(R, p, c.BLADE_PTS)
        other = [n for n in range(3) if n != k]
        hit = any(w.Rule.near_boxes(blade, (r, q), c.BOXES, 0.005) for _, q, r in c.NEIGHBOURS)
        check(f"slot {k}: the blade in its gap is 5 mm or more from every neighbour", hit, False)


def test_text():
    print("task.json")
    f = HERE / "task.json"
    if not f.exists():
        print("  (no task.json yet)")
        return
    d = json.loads(f.read_text())
    for k, lim in (("title", 120), ("instruction", 1000), ("context", 1000), ("fidelity", 500), ("scores", 300), ("run_limit", 200)):
        check(f"{k} within {lim}", len(d.get(k, "")) <= lim, True)
    for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        check(f"{k} filled", bool(d.get(k)), True)
    import task as t
    check("instruction is the task file's", d["instruction"], t.INSTRUCTION)
    for k, v in d.items():
        if isinstance(v, str):
            check(f"{k}: no em or en dash", chr(0x2014) in v or chr(0x2013) in v, False)
    for label, text, phrase in (("dose window", d["fidelity"], f"{w.D_MIN:g} to {w.D_MAX:g} s"), ("open time", d["fidelity"], f"within {w.T_OPEN:g} s"),
                                ("set band", d["fidelity"], f"{w.SET_POS * 1000:g} mm and {w.SET_ANG:g} degrees for {w.T_SET:g} s"),
                                ("knock", d["scoring"], f"{w.KNOCK_POS * 1000:g} mm or {w.KNOCK_ANG:g} degrees"), ("crush", d["scoring"], f"{w.CRUSH * 1000:g} mm or more"),
                                ("mark", d["scoring"], f"within {w.MARK_TOL * 1000:g} mm"), ("square", d["scoring"], f"within {w.SQUARE_TOL:g} degrees"),
                                ("held", d["scoring"], f"within {w.HELD * 100:g} cm"), ("clear", d["scoring"], f"{w.CLEAR * 100:g} cm or more"),
                                ("overheat", d["scoring"], f"over {w.D_MAX:g} s"), ("slots", d["setup"], "7.5, 11.0 and 14.5 cm"),
                                ("run limit", d["run_limit"], "4,800 steps")):
        check(f"text states the code's {label} ({phrase})", phrase in text, True)


def gripper_points(tcp, close_x: bool, half_gap: float) -> np.ndarray:
    """Points over one gripper's two fingers and its body (measured profile: memory of the URDF meshes): the fingers 17 mm wide at the pad tips,
    13 mm below the TCP, widening to 68 mm 50 mm up; each finger's outer face 4.5 mm past its pad, 0.29 mm more per mm up; the body 102 x 68 mm
    (the long side along the closing axis) from 71 mm above the TCP."""
    tcp = np.asarray(tcp, float)
    pts = []
    for h in np.linspace(0.0, 0.050, 11):
        half_w = 0.0085 + (0.034 - 0.0085) * h / 0.050
        for s in (-1.0, 1.0):
            for a in np.linspace(half_gap, half_gap + 0.0045 + 0.29 * h, 3):
                for b in np.linspace(-half_w, half_w, 5):
                    p = (s * a, b) if close_x else (b, s * a)
                    pts.append(tcp + np.array([p[0], p[1], -0.013 + h]))
    for h in np.linspace(0.071, 0.12, 4):
        for a in np.linspace(-0.051, 0.051, 7):
            for b in np.linspace(-0.034, 0.034, 5):
                p = (a, b) if close_x else (b, a)
                pts.append(tcp + np.array([p[0], p[1], h]))
    return np.array(pts)


def obstacles(knife_pose=None, welded=0):
    """Boxes (centre, half, rotation) in the table frame: the sprue, the patterns on the cluster, the knife if given."""
    out = [(w.SPRUE_C, w.SPRUE_H, np.eye(3))]
    poses = [(r, p) for _, p, r in c.NEIGHBOURS] + [c.slot_pose(k) for k in range(welded)]
    for R, p in poses:
        out += [(c.to_world(R, p, [ctr])[0], h, R @ Rb) for _, ctr, h, Rb in c.BOXES]
    if knife_pose is not None:
        R, p = knife_pose
        for ctr, h in ((np.zeros(3), np.array(c.BLADE) / 2), (np.array([c.BLADE[0] / 2 + c.SHANK[0] / 2, 0, 0]), np.array(c.SHANK) / 2),
                       (np.array([c.HANDLE_X0 + c.HANDLE_L / 2, 0, 0]), np.array([c.HANDLE_L / 2, c.HANDLE_R, c.HANDLE_R]))):
            out.append((c.to_world(R, p, [ctr])[0], h, R))
    return out


def gap_to(points, boxes) -> float:
    return min(float(c.box_distance(points, ctr, h, R)[0].min()) for ctr, h, R in boxes)


def demo_poses():
    """The demo's key TCP poses (table frame), from the same constants the demo uses."""
    import task as t
    R = c.KNIFE_REST_R
    knife_tcp = lambda blade: np.asarray(blade) + R @ np.array([c.GRASP_X, 0.0, 0.0]) + [0.0, 0.0, t.KNIFE_TCP]  # noqa: E731
    plat = lambda gate_end: np.asarray(gate_end) + [0.0, c.Y_PLATFORM, t.PLAT_TCP]  # noqa: E731
    right, left = [], []
    for i in range(3):
        _, p = c.loose_start(i)
        right += [(f"over pattern {i}", plat(p) + [0, 0, 0.06]), (f"on pattern {i}", plat(p))]
    for k, z in enumerate(c.SLOTS_Z):
        right += [(f"slot {k}: 4 cm off", plat([c.CX, c.FACE_Y - t.GAP_OPEN - 0.04, z])), (f"slot {k}: gap open", plat([c.CX, c.FACE_Y - t.GAP_OPEN, z])),
                  (f"slot {k}: gate on", plat([c.CX, c.FACE_Y + t.LEAN, z])), (f"slot {k}: let go, back", plat([c.CX, c.FACE_Y - 0.03, z]) + [0, 0, 0.04 if k < 2 else 0.0])]
        left += [(f"slot {k}: knife parked", knife_tcp([c.CX - 0.07, c.FACE_Y - 0.0055, z + 0.03])),
                 (f"slot {k}: blade 5 cm out", knife_tcp([c.CX - 0.05, c.FACE_Y - t.GAP_OPEN / 2, z])),
                 (f"slot {k}: blade in the gap", knife_tcp([c.CX + t.BLADE_IN, c.FACE_Y - c.BLADE[1] / 2, z]))]
    left = [("over the knife", knife_tcp(c.KNIFE_REST) + [0, 0, 0.06]), ("on the knife", knife_tcp(c.KNIFE_REST))] + left + \
           [("knife over its rest", knife_tcp(c.KNIFE_REST) + [0, 0, 0.05]), ("left clear", t.LEFT_CLEAR)]
    right += [("right clear", t.RIGHT_CLEAR)]
    return {"left": left, "right": right}


def test_reach():
    print("reach (IK from the ready pose, through the demo's key poses in order)")
    import math
    import demokit as d
    from yam_arena import ik, station
    from yam_arena.constants import READY
    import task as t
    grips = {"left": d.down(0), "right": d.down(90)}
    top = t.TOP
    for side, poses in demo_poses().items():
        q = np.array(READY, float)
        worst, least = 0.0, 180.0
        for label, p in poses:
            q, pe, re_ = ik.solve(np.asarray(p, float) + [0.0, 0.0, top], ik.matrix_to_quat_xyzw(grips[side].rot), q, base_pos=np.asarray(station.BASES[side], float), iters=300)
            margin = float(np.degrees(np.min(np.minimum(q - ik.LOWER, ik.UPPER - q))))
            worst, least = max(worst, pe * 1000), min(least, margin)
            if pe * 1000 > 0.5 or margin < 8.0:
                print(f"    {side} '{label}': IK {pe * 1000:.2f} mm, {math.degrees(re_):.2f} deg, joint margin {margin:.1f} deg")
        check(f"{side} arm: IK under 0.5 mm at every key pose", worst < 0.5, True)
        check(f"{side} arm: joint margin 8 deg or more (least {least:.1f})", least >= 8.0, True)


def test_clearance():
    print("clearance of the two grippers at the joint (finger and body profile from the URDF)")
    import task as t
    for k, z in enumerate(c.SLOTS_Z):
        R = c.KNIFE_REST_R
        knife = (R, np.array([c.CX + t.BLADE_IN, c.FACE_Y - c.BLADE[1] / 2, z]))
        hold = gripper_points(np.array([c.CX, c.FACE_Y + c.Y_PLATFORM, z + t.PLAT_TCP]), True, c.PLATFORM[0] / 2)
        kh = gripper_points(knife[1] + R @ np.array([c.GRASP_X, 0.0, 0.0]) + [0.0, 0.0, t.KNIFE_TCP], False, c.HANDLE_R)
        g1 = gap_to(hold, obstacles(knife, welded=k))
        g2 = gap_to(kh, obstacles(None, welded=k) + [(np.mean(hold, axis=0), np.ptp(hold, axis=0) / 2, np.eye(3))])
        g3 = gap_to(kh, [(c.to_world(*c.slot_pose(k), [ctr])[0], h, Rb) for _, ctr, h, Rb in c.BOXES])
        print(f"    slot {k}: holding hand to sprue, knife and cluster {g1 * 1000:.1f} mm; knife hand to cluster and the holding hand's box {g2 * 1000:.1f} mm; "
              f"to the held pattern {g3 * 1000:.1f} mm")
        check(f"slot {k}: holding hand 10 mm or more from everything but its pattern", g1 >= 0.010, True)
        check(f"slot {k}: knife hand 10 mm or more from everything", min(g2, g3) >= 0.010, True)
        shank_end = c.to_world(R, knife[1], [[c.BLADE[0] / 2, 0.0, 0.0]])[0]  # where the shank starts, toward -x
        check(f"slot {k}: the 3 mm shank stays 1 mm or more off the sprue's face (its end {(c.CX - c.SPRUE / 2 - shank_end[0]) * 1000:.1f} mm short)",
              shank_end[0] <= c.CX - c.SPRUE / 2 - 0.001, True)
        blade_x = sorted([c.to_world(R, knife[1], [[s_ * c.BLADE[0] / 2, 0, 0]])[0][0] for s_ in (-1, 1)])
        check(f"slot {k}: the blade covers the gate's middle", blade_x[0] <= c.CX - w.COVER and blade_x[1] >= c.CX + w.COVER, True)


TESTS = [test_ladder, test_red_team, test_set_and_dose, test_latches, test_geometry, test_reach, test_clearance, test_text]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\n{len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
