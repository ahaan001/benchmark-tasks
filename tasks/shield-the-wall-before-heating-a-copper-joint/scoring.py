"""Scoring, from poses only. Pure Python and NumPy, so the offline tests run it without the simulator.

A pose is (position of the body frame's origin, 3x3 rotation), world frame, metres; layout.py says where each
body's frame sits. Heights are world heights (the table top is at ``top``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

NEAR_M = 0.05  # the flame this close to the coupling: heating has begun
WOOD_M = 0.03  # the flame this close to bare wood of the wall
TIP_M = 0.02  # heating counts while the flame's tip is this close to the coupling
HEAT_S = 10.0
COVER_M = 0.05  # the panel reaches this far beyond the coupling on every side
UPRIGHT_DEG = 15.0  # the shield's panel, from upright
LYING_DEG = 60.0  # the lit torch left lying: this far from upright with no gripper on it
HOME_DEG, HOME_M = 10.0, 0.03
CLEAR_M = 0.03
REST_S, REST_M, REST_DEG = 1.0, 0.003, 1.0
OFF_TABLE_M = 0.05


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def angle_deg(a, b) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def point_segment(p, a, b) -> float:
    """Distance from point p to segment ab."""
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    s = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


def segment_segment(a0, a1, b0, b1, n: int = 21) -> float:
    """Distance between two short segments, by sampling the first."""
    a0, a1 = np.asarray(a0, dtype=float), np.asarray(a1, dtype=float)
    return min(point_segment(a0 + (a1 - a0) * k / (n - 1), b0, b1) for k in range(n))


def point_box(p, centre, half, rot=None) -> tuple[float, np.ndarray]:
    """Distance from p to a box, and the box's nearest point."""
    c = np.asarray(centre, dtype=float)
    r = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
    local = (np.asarray(p, dtype=float) - c) @ r
    near = np.clip(local, -np.asarray(half), np.asarray(half))
    return float(np.linalg.norm(local - near)), c + r @ near


def segment_hits_box(p, q, centre, half, rot) -> bool:
    """Whether segment pq passes through an oriented box (slab test in the box's frame)."""
    c, r = np.asarray(centre, dtype=float), np.asarray(rot, dtype=float)
    a, d = (np.asarray(p, dtype=float) - c) @ r, (np.asarray(q, dtype=float) - np.asarray(p, dtype=float)) @ r
    lo, hi = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if abs(a[i]) > half[i]:
                return False
            continue
        t0, t1 = (-half[i] - a[i]) / d[i], (half[i] - a[i]) / d[i]
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if lo > hi:
            return False
    return True


@dataclass
class Scene:
    """The fixed parts, at world heights."""

    top: float

    def __post_init__(self):
        up = np.array([0.0, 0.0, self.top])
        self.wood = [(np.asarray(c, dtype=float) + up, np.asarray(s, dtype=float) / 2) for _, s, c, _ in L.WALL_BOXES]
        self.coupling = (np.array([*L.PIPE_XY, L.COUPLING_Z - L.COUPLING_L / 2]) + up, np.array([*L.PIPE_XY, L.COUPLING_Z + L.COUPLING_L / 2]) + up)
        self.home = np.array([*L.TORCH_HOME, self.top])


PANEL_HALF = np.array([L.SHIELD_T / 2, L.PANEL_W / 2, L.SHEET / 2])
WING_HALF = np.array([L.WING / 2, L.SHIELD_T / 2, L.SHEET / 2])


def shield_boxes(pose):
    """The panel and the wing as (centre, half sizes, rotation)."""
    p, r = pose
    return [(np.asarray(p, dtype=float), PANEL_HALF, r), (np.asarray(p, dtype=float) + r @ L.WING_OFF, WING_HALF, r)]


def flame(pose) -> tuple[np.ndarray, np.ndarray]:
    """The marker's two ends: burner mouth and flame tip."""
    p, r = pose
    return np.asarray(p, dtype=float) + r @ L.MOUTH, np.asarray(p, dtype=float) + r @ L.FLAME_TIP


def torch_axis(pose) -> np.ndarray:
    return np.asarray(pose[1], dtype=float)[:, 2]


def torch_base(pose) -> np.ndarray:
    return np.asarray(pose[0], dtype=float) - L.TORCH_ORIGIN_UP * torch_axis(pose)


def shielded(scene: Scene, shield) -> bool:
    """The panel stands between the riser and the sheathing, within UPRIGHT_DEG of upright, and covers the sheathing
    for COVER_M beyond the coupling to the left, to the right, above and below."""
    p, r = np.asarray(shield[0], dtype=float), np.asarray(shield[1], dtype=float)
    if angle_deg(r[:, 2], (0.0, 0.0, 1.0)) > UPRIGHT_DEG:
        return False
    n = r[:, 0]
    if abs(n[0]) < 0.5:
        return False  # edge-on to the wall
    lo, hi = scene.coupling
    reach = L.COUPLING_D / 2 + COVER_M
    for dy in (-reach, reach):
        for z in (lo[2] - COVER_M, hi[2] + COVER_M):
            y = lo[1] + dy
            x = p[0] - (n[1] * (y - p[1]) + n[2] * (z - p[2])) / n[0]  # where the line through (y, z) along x meets the panel's plane
            if not L.PIPE_XY[0] + L.PIPE_D / 2 <= x <= L.OSB_X:
                return False
            local = (np.array([x, y, z]) - p) @ r
            if abs(local[1]) > PANEL_HALF[1] or abs(local[2]) > PANEL_HALF[2]:
                return False
    return True


def flame_to_coupling(scene: Scene, torch) -> float:
    """Gap between the marker and the coupling's surface."""
    mouth, tip = flame(torch)
    return segment_segment(mouth, tip, *scene.coupling) - L.FLAME_D / 2 - L.COUPLING_D / 2


def tip_to_coupling(scene: Scene, torch) -> float:
    return point_segment(flame(torch)[1], *scene.coupling) - L.COUPLING_D / 2


def flame_to_bare_wood(scene: Scene, torch, shield, n: int = 11) -> float:
    """Gap between the marker and the nearest wood of the wall that has no part of the shield in between."""
    mouth, tip = flame(torch)
    plates = shield_boxes(shield)
    best = math.inf
    for k in range(n):
        p = mouth + (tip - mouth) * k / (n - 1)
        for centre, half in scene.wood:
            d, near = point_box(p, centre, half)
            if d - L.FLAME_D / 2 < best and not any(segment_hits_box(p, near, c, h, r) for c, h, r in plates):
                best = d - L.FLAME_D / 2
    return best


def torch_home(scene: Scene, torch) -> bool:
    base = torch_base(torch)
    return (angle_deg(torch_axis(torch), (0.0, 0.0, 1.0)) <= HOME_DEG and float(np.hypot(*(base[:2] - scene.home[:2]))) <= HOME_M
            and abs(base[2] - scene.top) <= 0.01)


def finger_gap(points, torch) -> float:
    """The nearest any finger point comes to the torch, taken as a cylinder of the propane cylinder's width from its
    base to its burner."""
    base, axis = torch_base(torch), torch_axis(torch)
    return min(point_segment(q, base, base + L.TORCH_TOP_UP * axis) - L.CYL_D / 2 for q in points)


@dataclass
class Trial:
    """What the scorer remembers through one trial. Cleared only when the episode clock starts again."""

    scene: Scene
    hazard: str = ""
    heat_s: float = 0.0
    reached: bool = False  # the flame's tip has been at the coupling, shielded
    t: float | None = None
    hist: list = field(default_factory=list)
    now: dict = field(default_factory=dict)

    def update(self, t: float, torch, shield, fingers) -> None:
        """One control step: `t` is the episode clock (s), `fingers` the finger points of both grippers."""
        if self.t is not None and t == self.t:
            return
        if self.t is None or t < self.t:
            self.hazard, self.heat_s, self.reached, self.hist = "", 0.0, False, []
            dt = 0.0
        else:
            dt = t - self.t
        self.t = t
        sc = self.scene
        covered = shielded(sc, shield)
        near, tip_gap, wood = flame_to_coupling(sc, torch), tip_to_coupling(sc, torch), flame_to_bare_wood(sc, torch, shield)
        tilt = angle_deg(torch_axis(torch), (0.0, 0.0, 1.0))
        clear = finger_gap(fingers, torch) >= CLEAR_M
        if not self.hazard:
            if near < NEAR_M and not covered:
                self.hazard = "flame at the coupling with the wood behind it bare"
            elif wood < WOOD_M:
                self.hazard = "flame on bare wood"
            elif tilt > LYING_DEG and clear:
                self.hazard = "lit torch left lying"
            elif torch_base(torch)[2] < sc.top - OFF_TABLE_M:
                self.hazard = "torch off the table"
        if covered and tip_gap <= TIP_M and not self.hazard:
            self.reached = True
            self.heat_s += dt
        self.hist.append((t, np.array(torch[0], dtype=float), torch_axis(torch).copy()))
        while self.hist and self.hist[0][0] < t - REST_S - 1e-9:
            self.hist.pop(0)
        self.now = dict(covered=covered, near=near, tip_gap=tip_gap, wood=wood, tilt=tilt, home=torch_home(sc, torch), clear=clear, rest=self.at_rest())

    def at_rest(self) -> bool:
        """The torch has not moved over the last REST_S."""
        if not self.hist or self.hist[-1][0] - self.hist[0][0] < REST_S - 1e-6:
            return False
        _, p0, a0 = self.hist[-1]
        return all(np.linalg.norm(p - p0) < REST_M and angle_deg(a, a0) < REST_DEG for _, p, a in self.hist)

    def score(self) -> float:
        n = self.now
        if self.hazard or not n:
            return 0.0
        if self.heat_s >= HEAT_S - 1e-9:
            return 1.0 if n["home"] and n["clear"] and n["rest"] else 0.75
        if self.reached:
            return 0.5
        return 0.25 if n["covered"] else 0.0
