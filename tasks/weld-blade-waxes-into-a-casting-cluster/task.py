"""Wax-weld three turbine-blade patterns into the gaps of a casting cluster: one hand holds each pattern with its gate end at a mark on the sprue, the
other passes a hot wax knife between the gate end and the sprue so both faces melt; the knife comes out, the gate goes onto the sprue square, and the
pattern is held still until the wax sets.

An investment-casting wax room: a wax sprue standing on its pour cup on an assembly stand already carries three blade patterns, one under the three empty
marks and two on the next face beside the lower two. Three loose blade patterns lie in a foam tray; an electric wax knife lies in its rest.
Puffer and Phipps (ACEEE 2003): "a manual process using a hot knife blade passed between the two wax surfaces is used to attach the wax patterns to the
runners". A wax-welding patent (US 11,084,113): the tip is inserted in the weld plane, then withdrawn so the melted wax flows.

Model (declared in code, `weld_rule.py`): melting and setting are bookkeeping, not physics. Each pattern hangs on a chain of six free joints (three
slides, three hinges) from a hidden anchor, so it moves as a free body, falls when let go and is pushed only by contacts; a weld locks the chain where it
is. The patterns already on the cluster are locked from the start, with a spring that a knock can move.

    python demokit.py task.py OUT --mode full
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cluster as c  # noqa: E402
import mechkit as m  # noqa: E402
import weld_rule as w  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
INSTRUCTION = ("Weld the three loose blue wax blade patterns onto the sprue at the three empty scribed marks. For each: hold it by the two ends of its platform with its gate end "
               "against the mark, pass the hot knife between the gate end and the sprue so both faces melt, take the knife out, press the gate onto the sprue "
               "square, and keep the pattern still until the wax sets before you let go. The knife touches only the gate end and the sprue, never any pattern's "
               "airfoil, platform or root, and nothing may knock the patterns already on the cluster. When all three are on, lay the knife back in its rest and "
               "move both hands clear.")
SUCCESS_HOLD_S = 0.0  # the scorer counts its own hold
HOLD_STEPS = 20  # 1 s at 20 Hz
DT = 0.05
TOP = TABLE_TOP[SCENE]
UP = np.array([0.0, 0.0, TOP])

LOOKS = {
    "wax": m.Look(color=m.rgb(0.16, 0.36, 0.78), roughness=0.45),
    "sprue": m.Look(color=m.rgb(0.20, 0.55, 0.52), roughness=0.5),
    "mark": m.Look(color=m.rgb(0.05, 0.05, 0.06), roughness=0.8),
    "steel": m.Look(color=m.rgb(0.62, 0.63, 0.66), roughness=0.35, metallic=0.9),
    "foam": m.Look(color=m.rgb(0.45, 0.46, 0.48), roughness=0.95),
    "alu": m.Look(color=m.rgb(0.78, 0.79, 0.80), roughness=0.4, metallic=0.8),
    "handle": m.Look(color=m.rgb(0.10, 0.10, 0.11), roughness=0.6),
    "blade": m.Look(color=m.rgb(0.80, 0.80, 0.82), roughness=0.25, metallic=0.95),
    "rest": m.Look(color=m.rgb(0.20, 0.20, 0.22), roughness=0.6),
}

LINKS = ("sx", "sy", "sz", "rz", "ry", "rx")
LOOSE = [f"pattern{i}" for i in range(3)]
NEIGH = [f"on_cluster{k}" for k in range(3)]
K_SLIDE, K_HINGE = 3000.0, 20.0  # a set weld: N/m and N m/rad
E_SLIDE, E_HINGE = 1.5, 0.10  # what it takes to knock one: N and N m
D_SLIDE, D_HINGE = 2.0, 0.02


# -- meshes ---------------------------------------------------------------------------------------------------------------------------------------
def _swap_xz(mesh: m.Mesh) -> m.Mesh:
    return m.Mesh(mesh.v[:, [2, 1, 0]], mesh.f[:, ::-1])


def _root_mesh() -> m.Mesh:
    """A three-lobed fir-tree root: its profile in y (radial) and z, drawn through the root's 18 mm along x."""
    h, d = c.ROOT[2] / 2, c.ROOT[1]
    y1 = -c.GATE[1]
    prof = [(y1, h)]
    for k in range(3):
        y = y1 - d * (k + 0.5) / 3
        prof += [(y + d / 12, h - 0.0025 - 0.001 * k), (y, h - 0.001 * k)]
    prof += [(y1 - d, h)]
    outline = [(-z, y) for y, z in prof[::-1]] + [(z, y) for y, z in prof]
    centre = np.mean(np.array(outline), axis=0)
    pts = np.array(outline) - centre
    order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))
    mesh = m.prism(np.array(outline)[order], -c.ROOT[0] / 2, c.ROOT[0] / 2)
    return _swap_xz(mesh)


def _airfoil_mesh(n_span: int = 9, n_chord: int = 14) -> m.Mesh:
    """A cambered, twisted airfoil from the platform outwards (-y): 4% camber, its thickness by the NACA four-digit shape."""
    xs = (1 - np.cos(np.linspace(0.0, math.pi, n_chord))) / 2
    t = 5 * (c.THICK / c.CHORD) * (0.2969 * np.sqrt(xs) - 0.126 * xs - 0.3516 * xs ** 2 + 0.2843 * xs ** 3 - 0.1015 * xs ** 4)
    camber = 0.04 * (1 - (2 * xs - 1) ** 2)
    upper = np.c_[xs - 0.5, camber + t] * c.CHORD
    lower = np.c_[xs - 0.5, camber - t][::-1][1:-1] * c.CHORD
    ring = np.vstack([upper, lower])
    ring[:, 1] -= 0.04 * c.CHORD / 2
    rings = []
    for k in range(n_span):
        u = k / (n_span - 1)
        R = c.rot_y(c.TWIST * (u - 0.5))
        y = c.Y_AIR0 - u * c.SPAN + (0.0005 if k == 0 else 0.0)
        rings.append(np.array([R @ np.array([px, 0.0, pz]) for px, pz in ring]) + np.array([0.0, y, 0.0]))
    n = len(ring)
    v = np.vstack(rings + [rings[0].mean(axis=0, keepdims=True), rings[-1].mean(axis=0, keepdims=True)])
    f = []
    for k in range(n_span - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            f += [[a, b, b + n], [a, b + n, a + n]]
    c0, c1 = len(v) - 2, len(v) - 1
    for i in range(n):
        f.append([c0, (i + 1) % n, i])
        f.append([c1, (n_span - 1) * n + i, (n_span - 1) * n + (i + 1) % n])
    return m.Mesh(v, np.array(f)[:, ::-1])


def pattern_parts():
    parts = [("gate", m.box(c.GATE, (0.0, -c.GATE[1] / 2, 0.0)), "wax"),
             ("root", _root_mesh(), "wax"),
             ("platform", m.box(c.PLATFORM, (0.0, c.Y_PLATFORM, 0.0)), "wax"),
             ("airfoil", _airfoil_mesh(), "wax")]
    return parts


def pattern_colliders(prefix: str):
    return [m.col_box(f"{prefix}_{n}_c", tuple(2 * h), tuple(ctr), rot=None if np.allclose(R, np.eye(3)) else R) for n, ctr, h, R in c.BOXES]


def _volume_com():
    vol = np.array([np.prod(2 * h) for _, _, h, _ in c.BOXES])
    ctr = np.array([ctr for _, ctr, _, _ in c.BOXES])
    return tuple(float(v) for v in (vol[:, None] * ctr).sum(axis=0) / vol.sum())


PATTERN_COM = _volume_com()


def chain(prefix: str, set_: bool):
    """Six joints from a hidden anchor at the pattern's own origin to the pattern: free, or set (a weld) with a spring a knock can move."""
    ks, kh = (K_SLIDE, K_HINGE) if set_ else (0.0, 0.0)
    es, eh = (E_SLIDE, E_HINGE) if set_ else (50.0, 5.0)
    ds, dh = (D_SLIDE, D_HINGE) if set_ else (0.002, 1e-5)
    tiny = dict(mass=0.002, friction=0.0, armature=1e-5, inertia=(1e-7, 1e-7, 1e-7))
    links = [m.slider("sx", "X", -0.20, 0.20, (), (), spring=ks, damping=ds, effort=es, **tiny),
             m.slider("sy", "Y", -0.15, 0.45, (), (), parent="sx", spring=ks, damping=ds, effort=es, **tiny),
             m.slider("sz", "Z", -0.10, 0.30, (), (), parent="sy", spring=ks, damping=ds, effort=es, **tiny),
             m.hinge("rz", "Z", (0.0, 0.0, 0.0), -179.0, 179.0, (), (), parent="sz", spring=kh, damping=dh, effort=eh, **tiny),
             m.hinge("ry", "Y", (0.0, 0.0, 0.0), -85.0, 85.0, (), (), parent="rz", spring=kh, damping=dh, effort=eh, **tiny),
             m.hinge("rx", "X", (0.0, 0.0, 0.0), -179.0, 179.0, pattern_parts(), pattern_colliders(prefix), parent="ry", spring=kh, damping=dh, effort=eh,
                     mass=c.PATTERN_MASS, friction=0.0, armature=1e-5, gravity=True, com=PATTERN_COM, inertia=(1.8e-5, 1.5e-6, 1.8e-5))]
    return links


def _yaw(R) -> float:
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _bench():
    """The stand, cup and sprue with its marks; the drip tray; the foam tray; the knife's rest with its blade guard."""
    parts, cols = [], []

    def b(name, size, ctr, look, collide=True):
        parts.append((name, m.box(size, ctr), look))
        if collide:
            cols.append(m.col_box(name + "_c", size, ctr))

    b("drip_tray", (0.16, 0.16, 0.002), (c.CX, c.CY, 0.001), "alu")
    b("stand", c.STAND, (c.CX, c.CY, 0.002 + c.STAND[2] / 2), "steel")
    z0 = 0.002 + c.STAND[2]
    cup = m.lathe([(0.0, 0.0), (c.CUP_R, 0.0), (c.SPRUE * 0.6, c.CUP_H), (0.0, c.CUP_H)], 32).moved((c.CX, c.CY, z0))
    parts.append(("cup", cup, "sprue"))
    cols.append(m.col_hull("cup_c", m.lathe([(0.0, 0.0), (c.CUP_R, 0.0), (c.SPRUE * 0.6, c.CUP_H), (0.0, c.CUP_H)], 16).moved((c.CX, c.CY, z0))))
    b("sprue", (c.SPRUE, c.SPRUE, c.SPRUE_Z1 - c.SPRUE_Z0), (c.CX, c.CY, (c.SPRUE_Z0 + c.SPRUE_Z1) / 2), "sprue")
    for k, z in enumerate(c.SLOTS_Z):
        for dz in (-c.GATE[2] / 2 - 0.0015, c.GATE[2] / 2 + 0.0015):
            b(f"mark{k}_{'lu'[dz > 0]}", (0.016, 0.0004, 0.0008), (c.CX, c.FACE_Y - 0.0002, z + dz), "mark", collide=False)
    # foam tray: a slab and three bars across it, under the gates, the roots and the airfoils
    xs = (min(c.TRAY_X) - 0.03, max(c.TRAY_X) + 0.03)
    ys = (c.TRAY_Y - c.LENGTH - 0.01, c.TRAY_Y + 0.008)
    b("tray", (xs[1] - xs[0], ys[1] - ys[0], c.TRAY_TOP), ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, c.TRAY_TOP / 2), "foam")
    for name, y, depth in c.TRAY_BARS:
        top = c.LIE_Z - depth - 0.001
        b(name, (xs[1] - xs[0], 0.008, top - c.TRAY_TOP), ((xs[0] + xs[1]) / 2, c.TRAY_Y + y, (c.TRAY_TOP + top) / 2), "foam")
    # the knife's rest: two cradles with side rails under the handle, and a guard round the blade
    R = c.KNIFE_REST_R
    for k, x in enumerate(c.REST_X):
        p = c.KNIFE_REST + R @ np.array([x, 0.0, 0.0])
        top = c.KNIFE_REST[2] - c.HANDLE_R - 0.0005
        b(f"cradle{k}", (0.02, 0.03, top), (p[0], p[1], top / 2), "rest")
        for s in (-1, 1):
            b(f"cradle{k}_rail{'pn'[s > 0]}", (0.02, 0.004, 0.008), (p[0], p[1] + s * (c.HANDLE_R + 0.003), top + 0.004), "rest")
    tip = c.KNIFE_REST + R @ np.array([-c.BLADE[0] / 2, 0.0, 0.0])
    floor = c.KNIFE_REST[2] - c.BLADE[2] / 2 - 0.003
    gx = (tip[0] - 0.004, c.KNIFE_REST[0] + c.BLADE[0] / 2 + 0.004)
    b("guard_floor", (gx[1] - gx[0] + 0.008, 0.024, floor), ((gx[0] + gx[1]) / 2, c.KNIFE_REST[1], floor / 2), "rest")
    b("guard_end", (0.004, 0.024, 0.020), (gx[1] + 0.006, c.KNIFE_REST[1], floor + 0.010), "rest")
    for s in (-1, 1):
        b(f"guard_side{'pn'[s > 0]}", (gx[1] - gx[0] + 0.008, 0.003, 0.016), ((gx[0] + gx[1]) / 2, c.KNIFE_REST[1] + s * 0.010, floor + 0.008), "rest")
    return parts, cols


_bp, _bc = _bench()
BENCH = m.fixed("bench", tuple(UP), _bp, _bc, LOOKS, friction=(0.9, 0.8))


def _knife():
    parts = [("blade", m.box(c.BLADE), "blade"),
             ("shank", m.box(c.SHANK, (c.BLADE[0] / 2 + c.SHANK[0] / 2, 0.0, 0.0)), "blade"),
             ("handle", m.cylinder(c.HANDLE_R, c.HANDLE_L, (c.HANDLE_X0 + c.HANDLE_L / 2, 0.0, 0.0), axis="x", segments=20), "handle"),
             ("cord_boot", m.cylinder(0.004, 0.015, (c.HANDLE_X0 + c.HANDLE_L + 0.0075, 0.0, 0.0), axis="x", segments=12), "handle")]
    cols = [m.col_box("blade_c", c.BLADE),
            m.col_box("shank_c", c.SHANK, (c.BLADE[0] / 2 + c.SHANK[0] / 2, 0.0, 0.0)),
            m.col_hull("handle_c", m.cylinder(c.HANDLE_R, c.HANDLE_L, (c.HANDLE_X0 + c.HANDLE_L / 2, 0.0, 0.0), axis="x", segments=12))]
    return parts, cols


_kp, _kc = _knife()
KNIFE = m.free("knife", tuple(c.KNIFE_REST + UP + np.array([0.0, 0.0, 0.001])), _kp, _kc, LOOKS, mass=c.KNIFE_MASS, friction=(0.9, 0.8),
               yaw_deg=_yaw(c.KNIFE_REST_R), com=(c.GRASP_X, 0.0, 0.0))

PATTERNS = []
for _i in range(3):
    _R, _p = c.loose_start(_i)
    PATTERNS.append(m.mechanism(LOOSE[_i], tuple(_p + UP + np.array([0.0, 0.0, 0.001])), chain(LOOSE[_i], False), LOOKS, friction=(0.9, 0.8), yaw_deg=_yaw(_R)))
for _k, (_n, _p, _R) in enumerate(c.NEIGHBOURS):
    _out = _R @ np.array([0.0, -0.0003, 0.0])  # 0.3 mm off the sprue's face, so that nothing is authored touching
    PATTERNS.append(m.mechanism(NEIGH[_k], tuple(_p + _out + UP), chain(NEIGH[_k], True), LOOKS, friction=(0.9, 0.8), yaw_deg=_yaw(_R)))

OBJECTS = [BENCH, KNIFE, *PATTERNS]


# -- the scorer -----------------------------------------------------------------------------------------------------------------------------------
def pose(env, name):
    R, p = m.link_pose(env, name, "rx")
    return R, p - UP


def knife_pose(env):
    k = m.state(env, "knife")
    return k.rotation(), k.pos - UP


class Trial:
    def __init__(self):
        self.rule = w.Rule()
        self.reset()

    def reset(self, env=None):
        self.rule.reset()
        self.n = 0
        self.hold = 0
        self.full = False
        self.score = 0.0

    def obs(self, env) -> w.Obs:
        return w.Obs(self.n * DT, knife_pose(env), [pose(env, n) for n in LOOSE], [pose(env, n) for n in NEIGH], m.fingertips(env) - UP, m.finger_opening(env))

    def step(self, env):
        o = self.obs(env)
        was = (self.rule.knock, self.rule.crush, self.rule.pad)
        self.rule.step(o)
        now = (self.rule.knock, self.rule.crush, self.rule.pad)
        if now != was:
            print(f"[weld] t {o.t:5.1f} latched: knock {int(now[0])} crush {int(now[1])} pad {int(now[2])}; openings {np.round(np.array(o.opening) * 1000, 1)} mm; "
                  f"neighbours off by {[round(float(np.linalg.norm(p - ref[1])) * 1000, 1) for (_, p), ref in zip(o.neighbours, c.NEIGHBOURS)]} mm", flush=True)
        for i in self.rule.locks:
            for link in LINKS:
                k, e, d = (K_SLIDE, E_SLIDE, D_SLIDE) if link[0] == "s" else (K_HINGE, E_HINGE, D_HINGE)
                m.set_gains(env, LOOSE[i], link, k, d)
                m.lock(env, LOOSE[i], link, stiffness=k, effort=e)
        self.rule.locks.clear()
        self.n += 1
        s = self.rule.score(o)
        self.hold = self.hold + 1 if s >= 0.999 else 0
        self.full = self.full or self.hold >= HOLD_STEPS
        self.score = 1.0 if self.full else min(s, 0.95)


TRIAL = Trial()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    return TRIAL.full


def score(objects) -> float:
    return TRIAL.score


def _report(ctx, env, label=""):
    o = TRIAL.obs(env)
    r = TRIAL.rule
    js = " ".join(f"[{i}: dose {j.dose:.1f} wet {'-' if j.wet_t is None else round(o.t - j.wet_t, 1)} welded {int(j.welded)} slot {r.on_mark(o.loose[i])} "
                  f"melt {int(j.melt_back)} mark {int(j.marked)}]" for i, j in enumerate(r.joints))
    print(f"[weld] t {o.t:5.1f} {label}{js} knock {int(r.knock)} crush {int(r.crush)} pad {int(r.pad)} marks(n) {r.neighbour_marked} rest {r.rest_t:.1f} "
          f"knife home {r.knife_home(o)}; score {r.score(o):.2f}", flush=True)
    if os.environ.get("WELD_DEBUG"):
        for i, (R, p) in enumerate(o.loose):
            print(f"[weld]   pattern {i} at {np.round(p * 1000, 1)} mm, {c.angle_deg(R, np.eye(3)):.1f} deg; knife {np.round(o.knife[1] * 1000, 1)}", flush=True)


# -- the scripted demo ----------------------------------------------------------------------------------------------------------------------------
PLAT_TCP = 0.007  # the holding TCP above the platform's centre: the pad tips 6 mm below its middle, 5 mm above its lower edge
KNIFE_TCP = 0.010  # the knife hand's TCP above the handle's axis
GAP_OPEN = 0.004  # the gate end's distance off the sprue while the blade goes in
LEAN = 0.0005  # how far past touching each push aims
LEFT_CLEAR, RIGHT_CLEAR = np.array([0.20, 0.30, 0.16]), np.array([0.20, -0.22, 0.15])  # where the hands end, clear of everything


def demo_script(ctx):
    """Right: takes each loose pattern by its platform's two ends and brings its gate end 4 mm off the next mark, lowest first. Left: takes the knife by its handle,
    slides the blade into the gap from -x, presses it onto the sprue; the right brings the gate onto the blade; 3 s; the knife slides out; the right
    presses the gate onto the sprue and holds 11 s, then lets go. Last, the knife goes back in its rest and both hands move clear.
    Failure modes (--mode): none does nothing; hover keeps the blade in the 4 mm gap touching neither face; early lets go of the first pattern 4 s
    after it closes; order presses the gate on first and then tries the knife."""
    import demokit as d

    env = ctx.emb._env
    l, r = ctx.left, ctx.right
    mode = ctx.mode
    if mode not in ("full", "none", "hover", "early", "order"):
        raise ValueError(mode)
    for _ in range(10):
        yield
    _report(ctx, env, f"{mode} start: ")
    if mode == "none":
        for _ in range(40):
            yield
        return
    g0 = d.down(0)  # fingers closing along y: the knife's handle
    g90 = d.down(90)  # fingers closing along x: a pattern's platform, by its two ends
    plat_at = lambda i: pose(env, LOOSE[i])[0] @ np.array([0.0, c.Y_PLATFORM, 0.0]) + pose(env, LOOSE[i])[1] + UP  # noqa: E731
    gate_end = lambda i: pose(env, LOOSE[i])[1] + UP  # noqa: E731
    blade_c = lambda: knife_pose(env)[1] + UP  # noqa: E731
    grasp_at = lambda: knife_pose(env)[0] @ np.array([c.GRASP_X, 0.0, 0.0]) + knife_pose(env)[1] + UP + np.array([0.0, 0.0, KNIFE_TCP])  # noqa: E731

    # the knife: taken once, held level, blade toward +x
    def take(arm, where, height, width, grip, force, label, part_at):
        """A pick that checks itself: settle over the part and at it with long servos, close, lift 3 cm, and see that the part came up; else open,
        back off and try again (twice at most). Returns whether the part is held."""
        from yam_arena.constants import FINGER_OPEN
        for attempt in range(3):
            up = np.array([0.0, 0.0, height])
            yield from arm.move(where() + up + [0.0, 0.0, 0.06], grip=grip, finger=FINGER_OPEN, t=2.5 if attempt == 0 else 1.5, label=f"{label}: over")
            yield from arm.servo(where() + up + [0.0, 0.0, 0.06], tol=0.002, timeout=3.0, gain=0.3, label=f"{label}: over")
            yield from arm.move(where() + up, t=1.2, label=f"{label}: down")
            yield from arm.servo(lambda tt: where() + up, tol=0.0015, timeout=3.0, gain=0.3, label=f"{label}: down")
            z0 = float(part_at()[2])
            yield from arm.close(width, force, label=f"{label}: close")
            yield from arm.wait(0.3)
            yield from arm.move(arm.at() + [0.0, 0.0, 0.03], t=1.0, label=f"{label}: lift")
            yield from arm.wait(0.3)
            rose = float(part_at()[2]) - z0
            print(f"[weld] {label}: attempt {attempt + 1}, the part rose {rose * 1000:.0f} mm; hand {np.round((arm.tcp() - where() - up) * 1000, 1)} mm off", flush=True)
            if rose > 0.02:
                return True
            yield from arm.open(t=0.4, label=f"{label}: let go, try again")
            yield from arm.move(arm.at() + [0.0, 0.0, 0.04], t=0.8, label=f"{label}: up")
        return False

    knife_at = lambda: knife_pose(env)[1] + UP  # noqa: E731
    ok = yield from take(l, grasp_at, 0.0, 2 * c.HANDLE_R, g0, 20.0, "take the knife", knife_at)
    if not ok:
        return
    park = lambda k: np.array([c.CX - 0.07, c.FACE_Y - 0.0015 - 0.004, c.SLOTS_Z[k] + 0.03]) + UP  # noqa: E731
    yield from l.move(park(0) + np.array([-c.GRASP_X, 0.0, KNIFE_TCP]), t=2.5, label="the knife beside the lowest mark")
    for k in range(3):
        i = k
        ok = yield from take(r, lambda: plat_at(i), PLAT_TCP, c.PLATFORM[0], g90, 15.0, f"take pattern {i}", lambda: gate_end(i))
        if not ok:
            break
        yield from r.move(r.at() + [0.0, 0.0, 0.02], t=0.6, label=f"pattern {i} up")
        goal = np.array([c.CX, c.FACE_Y - GAP_OPEN, c.SLOTS_Z[k]]) + UP
        yield from r.move(goal + np.array([0.0, -0.04, 0.0]) + (plat_at(i) - gate_end(i)) + np.array([0.0, 0.0, PLAT_TCP]), t=3.0, label=f"pattern {i} 4 cm off its mark")
        yield from r.servo(goal + np.array([0.0, -0.02, 0.0]), tol=0.0015, timeout=2.0, gain=0.3, read=lambda: gate_end(i), label="gate square to the mark, 2 cm off")
        if mode == "order":
            yield from r.servo(goal + np.array([0.0, GAP_OPEN + LEAN, 0.0]), tol=0.0008, timeout=2.0, gain=0.3, read=lambda: gate_end(i), label="the gate onto the sprue first")
        else:
            yield from r.servo(goal, tol=0.0008, timeout=2.0, gain=0.3, read=lambda: gate_end(i), label="gate 4 mm off the mark")
        # the blade in from -x, level with the gate, mid-gap; then onto the sprue's face
        lane = np.array([c.CX - 0.05, c.FACE_Y - GAP_OPEN / 2, c.SLOTS_Z[k]]) + UP
        yield from l.servo(lane, tol=0.0015, timeout=2.5, gain=0.3, read=blade_c, label="blade level with the gap, 5 cm out")
        yield from l.servo(lambda tt: lane + np.array([min(0.054, 0.025 * tt), 0.0, 0.0]), tol=0.0008, timeout=3.5, gain=0.3, read=blade_c, label="slide the blade into the gap")
        if mode == "hover":
            yield from l.wait(5.0)
        else:
            yield from l.servo(np.array([c.CX + 0.004, c.FACE_Y - c.BLADE[1] / 2 + LEAN, c.SLOTS_Z[k]]) + UP, tol=0.0006, timeout=2.0, gain=0.3, read=blade_c,
                               label="blade onto the sprue")
            yield from r.servo(np.array([c.CX, c.FACE_Y - c.BLADE[1] + LEAN, c.SLOTS_Z[k]]) + UP, tol=0.0006, timeout=2.0, gain=0.3, read=lambda: gate_end(i),
                               label="gate onto the blade")
            yield from l.wait(3.0)
        _report(ctx, env, f"heated {i}: ")
        s0 = blade_c()
        yield from l.servo(lambda tt: s0 + np.array([-min(0.07, 0.035 * tt), -0.0003, 0.0]), tol=0.002, timeout=2.5, gain=0.3, read=blade_c,
                           label="slide the blade out along the gap")
        yield from r.servo(np.array([c.CX, c.FACE_Y + LEAN, c.SLOTS_Z[k]]) + UP, tol=0.0006, timeout=2.0, gain=0.3, read=lambda: gate_end(i), label="gate onto the sprue")
        yield from l.move(park(min(k + 1, 2)) + np.array([-c.GRASP_X, 0.0, KNIFE_TCP]), t=1.5, label="the knife up to the next mark")
        if mode == "early" and k == 0:
            yield from r.wait(4.0)
        else:
            yield from r.wait(11.0)  # held still while the wax sets
        _report(ctx, env, f"held {i}: ")
        yield from r.open(t=0.4, label="let go")
        yield from r.move(r.at() + np.array([0.0, -0.03, 0.0]), t=0.8, label="the fingers back off the platform")
        if k < 2:  # over the top mark the arm is near its limits: back off level instead
            yield from r.move(r.at() + np.array([0.0, 0.0, 0.04]), t=0.8, label="the right hand up")
    # the knife back in its rest
    rest = c.KNIFE_REST + UP
    yield from l.move(rest + c.KNIFE_REST_R @ np.array([c.GRASP_X, 0.0, 0.0]) + np.array([0.0, 0.0, KNIFE_TCP + 0.05]), t=3.0, label="the knife over its rest")
    yield from l.servo(rest + np.array([0.0, 0.0, 0.02]), tol=0.002, timeout=2.0, gain=0.3, read=blade_c, label="the blade over its guard")
    yield from l.servo(lambda tt: rest + np.array([0.0, 0.0, max(0.02 - 0.02 * tt, 0.001)]), tol=0.002, timeout=2.5, gain=0.3, read=blade_c, label="the knife down into its rest")
    yield from l.open(t=0.4, label="let go of the knife")
    yield from l.move(l.at() + np.array([0.0, 0.0, 0.08]), t=1.0, label="the left hand up")
    yield from ctx.together(l.move(LEFT_CLEAR + UP, t=2.0, label="left clear"), r.move(RIGHT_CLEAR + UP, t=2.0, label="right clear"))
    for k in range(60):
        yield
        if k % 30 == 29:
            _report(ctx, env, "end: ")
