"""The wax model and the score, as plain arithmetic on poses: no simulator, so the tests can drive it by hand.

Wax cannot be simulated here, so it follows a declared rule (task.json, fidelity). The rule changes only how stiff the
wax is; every motion is the robot's own:

- the seat under the stone yields at SEAT_HOLD_N and stays where it is pressed;
- a claw springs back while it is cold; once the warm pen's tip has rested on it for SOFT_AFTER_S it is soft, turns
  freely where it is pushed and stays there; it sets as it cools, melts after MELT_AFTER_S of touch, and breaks if
  forced COLD_BREAK_DEG while cold;
- the pen's tip resting MARK_AFTER_S on any other wax, or on the stone, leaves a melt mark.

The times and forces are estimates, to be calibrated on real wax (README).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import ring_geometry as g

SOFT_AFTER_S = 1.5
MELT_AFTER_S = 5.0
COOL_RATE = 1.0  # seconds of heat lost per second with the tip away
MARK_AFTER_S = 0.5
COLD_BREAK_DEG = 25.0
COLD_RATE = 0.004  # N m/rad: a cold claw's spring-back (0.2 N at its tip moves it about 0.2 mm)
COLD_DAMP = 4.5e-5  # N m s/rad
SOFT_DAMP = 2.0e-5  # N m s/rad: all that holds a soft claw: 0.01 N on its tip turns it 60 degrees a second
SEAT_RATE = 8000.0  # N/m
SEAT_HOLD_N = 0.6
SEAT_YIELD = SEAT_HOLD_N / SEAT_RATE  # m
TOUCH_GAP = 0.00025  # the tip counts as on a surface within a quarter millimetre of it
FINGER_HALF = 0.0045  # from a finger's centre line to its faces
FINGER_TOUCH = 0.0065  # a finger's centre line this near a surface: the finger is on it
CLOSED_OVERLAP = 0.001
HELD_MIN_CLAWS = 3
IN_HEAD_OFF = 0.0012
IN_HEAD_UP = (-0.0006, 0.0026)
IN_HEAD_TILT_DEG = 30.0
HOME_TABLE_OVER_RIM = 0.002  # pressed home: the table stands less than this above the head's rim
HOME_UP = HOME_TABLE_OVER_RIM - g.TABLE_OVER_GIRDLE - g.GIRDLE_H / 2
LEVEL_DEG = 10.0
CLEAR = 0.05
OFF_TABLE = 0.02
REST_S, REST_SHIFT, REST_TURN_DEG = 1.0, 0.0003, 0.5  # at rest: over this long, no loose part has moved this far or turned this much
TIPPED_GAP = 0.001  # the mandrel's thin end this near the table with no hand on the mandrel: it has tipped over its V-block
PEN_DOWN_AFTER_S = 2.0
PEN_HOME = (0.006, 0.004, 10.0)  # in its stand: within this of the stand's axis, of its resting height, and of upright (deg)
TOOL_HOME = (0.006, 0.005, 12.0)
MANDREL_HOME = (0.008, 0.003, 5.0)
W_STONE, W_HOME, W_CLAW, W_PUT_BACK = 0.1, 0.1, 0.15, 0.2
MARK_CAP = 0.5


@dataclass
class Pose:
    pos: np.ndarray
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))

    def world(self, local) -> np.ndarray:
        return self.pos + self.rot @ np.asarray(local, float)

    def local(self, world) -> np.ndarray:
        return self.rot.T @ (np.asarray(world, float) - self.pos)

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(self.rot[2, 2], -1.0, 1.0))))


@dataclass
class Snapshot:
    """One control step as the scorer sees it. Positions are in the station frame, heights above the table top."""

    dt: float
    mandrel: Pose
    claws: tuple  # four joint angles, rad, 0 as moulded, positive toward the stone
    seat: float  # m, 0 before pressing, negative as pressed
    stone: Pose
    pen: Pose
    tweezers: Pose
    pusher: Pose
    fingers: dict  # arm -> (finger a's points, finger b's points), each (n, 3)
    at_rest: bool = True


MANDREL_START = Pose(np.array([g.RING_XY[0], g.RING_XY[1], g.AXIS_H]))
STONE_START = Pose(np.array([g.STONE_BLOCK_XY[0], g.STONE_BLOCK_XY[1], g.STONE_REST_H]))
PEN_START = Pose(np.array([g.PEN_STAND_XY[0], g.PEN_STAND_XY[1], g.PEN_REST_Z]))
TWEEZERS_START = Pose(np.array([g.TWEEZER_XY[0], g.TWEEZER_XY[1], g.TW_REST_Z]))
PUSHER_START = Pose(np.array([g.PUSHER_XY[0], g.PUSHER_XY[1], g.PU_REST_Z]))
# What a gripper must stand clear of at the end, as rods in each part's own frame: (from, to, radius).
RODS = {
    "pen": (((0.0, 0.0, 0.0), (0.0, 0.0, g.PEN_TIP_LEN), g.PEN_TIP_R), ((0.0, 0.0, g.PEN_TIP_LEN), (0.0, 0.0, g.PEN_LEN), g.PEN_HANDLE_R)),
    "tweezers": (((0.0, 0.0, 0.0), (0.0, 0.0, g.TW_LEN), 0.008),),
    "pusher": (((0.0, 0.0, 0.0), (0.0, 0.0, g.PU_SHANK), 0.003), ((0.0, 0.0, g.PU_SHANK), (0.0, 0.0, g.PU_LEN), g.PU_KNOB_R)),
    "mandrel": (((0.0, g.Y_THIN, 0.0), (0.0, g.Y_THICK, 0.0), g.STICK_R_THICK), ((0.0, 0.0, 0.0), (0.0, 0.0, g.Z_S + g.CLAW_H), g.CUP_R1)),
    "stone": (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), g.STONE_R),),
}


def clearance(points: np.ndarray, poses: dict) -> float:
    """The smallest gap between the fingers (points down their centre lines) and the mandrel, the ring, the stone and the tools."""
    best = math.inf
    for name, rods in RODS.items():
        loc = (points - poses[name].pos) @ poses[name].rot
        for a, b, r in rods:
            a, ab = np.asarray(a, float), np.asarray(b, float) - np.asarray(a, float)
            t = np.clip((loc - a) @ ab / max(float(ab @ ab), 1e-18), 0.0, 1.0)
            best = min(best, float(np.linalg.norm(loc - (a + t[:, None] * ab), axis=1).min()) - r - FINGER_HALF)
    return best


def _near_start(pose: Pose, start: Pose, tol) -> bool:
    d = pose.pos - start.pos
    return bool(math.hypot(d[0], d[1]) <= tol[0] and abs(d[2]) <= tol[1] and pose.tilt_deg() <= tol[2])


def pen_held(pen: Pose, fingers: dict) -> bool:
    """One hand's two fingers on opposite sides of the pen's handle."""
    for a, b in fingers.values():
        sides = []
        for pts in (a, b):
            loc = (np.atleast_2d(pts) - pen.pos) @ pen.rot
            on = (loc[:, 2] > g.PEN_TIP_LEN) & (loc[:, 2] < g.PEN_LEN + 0.01) & (np.hypot(loc[:, 0], loc[:, 1]) < g.PEN_HANDLE_R + 0.012)
            sides.append(loc[on][:, :2].mean(axis=0) if on.any() else None)
        if sides[0] is not None and sides[1] is not None and float(sides[0] @ sides[1]) < 0.0:
            return True
    return False


def settled(trail: list, dt: float) -> bool:
    """Whether every loose part has kept its place for the last REST_S. `trail` holds {name: Pose}, one a control step, newest last, and is trimmed here.
    Rest is read from where the parts have been, not from their speeds: a part held between others (the stone under four claws) can show a speed and
    go nowhere."""
    span = max(2, round(REST_S / dt))
    del trail[:-span]
    if len(trail) < span:
        return False
    now = trail[-1]
    for old in trail[:-1]:
        for name, pose in now.items():
            turn = math.degrees(math.acos(float(np.clip((np.trace(old[name].rot.T @ pose.rot) - 1.0) / 2.0, -1.0, 1.0))))
            if float(np.linalg.norm(old[name].pos - pose.pos)) >= REST_SHIFT or turn >= REST_TURN_DEG:
                return False
    return True


def mandrel_held(mandrel: Pose, fingers: dict) -> bool:
    """One hand's two fingers on opposite sides of the stick, on its thick half."""
    for a, b in fingers.values():
        sides = []
        for pts in (a, b):
            loc = (np.atleast_2d(pts) - mandrel.pos) @ mandrel.rot
            on = (loc[:, 1] > g.REST_Y) & (loc[:, 1] < g.Y_THICK + 0.01) & (np.hypot(loc[:, 0], loc[:, 2]) < g.STICK_R_THICK + 0.012)
            sides.append(float(loc[on][:, 0].mean()) if on.any() else None)
        if sides[0] is not None and sides[1] is not None and sides[0] * sides[1] < 0.0:
            return True
    return False


class Wax:
    """The wax's state over one trial. `reset` is called on a real episode reset only; nothing here is cleared because
    parts are back where they started."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        n = g.N_CLAWS
        self.heat = [0.0] * n
        self.target = [0.0] * n  # where each claw's drive rests: its set angle
        self.soft = [False] * n
        self.melted = [False] * n
        self.broken = [False] * n
        self.closed = [False] * n
        self.closed_at_home = [False] * n
        self.seat_target = 0.0
        self.mark_time = 0.0
        self.mark = False
        self.fingered = False
        self.pen_free = 0.0
        self.pen_down = False
        self.tipped = False
        self.off_table = False
        self.now: dict = {"score": 0.0}

    # -- one control step ---------------------------------------------------------------------------------------------

    def step(self, s: Snapshot) -> dict:
        m = s.mandrel
        hot = [m.local(s.pen.world(h)) for h in g.PEN_HOT]
        reach = g.PEN_TIP_R + TOUCH_GAP
        off, up, tilt = g.stone_in_head(m.local(s.stone.pos), m.rot.T @ s.stone.rot[:, 2])
        in_head = off <= IN_HEAD_OFF and IN_HEAD_UP[0] <= up <= IN_HEAD_UP[1] and tilt <= IN_HEAD_TILT_DEG
        home = in_head and up <= HOME_UP and tilt <= LEVEL_DEG

        for i in range(g.N_CLAWS):
            if self.melted[i] or self.broken[i]:
                continue
            q = float(s.claws[i])
            box = g.claw_box(i, q)
            touch = any(g.box_gap(p, *box) <= reach for p in hot)
            self.heat[i] = self.heat[i] + s.dt if touch else max(0.0, self.heat[i] - COOL_RATE * s.dt)
            if self.heat[i] >= MELT_AFTER_S:
                self.melted[i], self.soft[i], self.closed[i] = True, False, False
                continue
            self.soft[i] = self.heat[i] >= SOFT_AFTER_S
            if self.soft[i]:
                self.target[i] = q  # its spring is off (the task file sets the joint's gains): it will set where it stands
            elif abs(q - self.target[i]) > math.radians(COLD_BREAK_DEG):
                self.broken[i], self.closed[i] = True, False
                continue
            closed = not self.soft[i] and g.claw_overlap(q) >= CLOSED_OVERLAP
            if closed and not self.closed[i]:
                self.closed_at_home[i] = home
            self.closed[i] = closed

        if self.seat_target - s.seat > SEAT_YIELD:
            self.seat_target = max(s.seat + SEAT_YIELD, -g.SEAT_LIMIT)

        on_other = any(g.wax_body_gap(p) <= reach for p in hot) or any(g.stone_gap(s.stone.local(s.pen.world(h))) <= reach for h in g.PEN_HOT)
        self.mark_time = self.mark_time + s.dt if on_other else max(0.0, self.mark_time - COOL_RATE * s.dt)
        self.mark = self.mark or self.mark_time >= MARK_AFTER_S

        pts = np.vstack([np.atleast_2d(p) for pair in s.fingers.values() for p in pair])
        if not self.fingered:  # only the points near enough to matter are measured exactly
            near = (np.linalg.norm(pts - s.stone.pos, axis=1) < g.STONE_R + FINGER_TOUCH + 0.002) | (np.linalg.norm(pts - m.pos, axis=1) < g.Z_S + g.CLAW_H + g.CUP_R1 + FINGER_TOUCH)
            for p in pts[near]:
                if g.stone_gap(s.stone.local(p)) <= FINGER_TOUCH or g.wax_gap(m.local(p), s.claws) <= FINGER_TOUCH:
                    self.fingered = True
                    break

        pen_home = _near_start(s.pen, PEN_START, PEN_HOME)
        self.pen_free = 0.0 if pen_home or pen_held(s.pen, s.fingers) else self.pen_free + s.dt
        self.pen_down = self.pen_down or self.pen_free >= PEN_DOWN_AFTER_S
        tail = m.world((0.0, g.Y_THIN + g.STICK_R_THIN, 0.0))[2] - g.STICK_R_THIN  # how high the thin end's underside stands
        if tail < TIPPED_GAP and not mandrel_held(m, s.fingers):
            self.tipped = True  # pressed on with nobody holding it, the mandrel has tipped over its V-block until its thin end is on the table
        if min(s.stone.pos[2], s.pen.pos[2], s.tweezers.pos[2], s.pusher.pos[2], s.mandrel.pos[2]) < -OFF_TABLE:
            self.off_table = True

        clear = clearance(pts, {"pen": s.pen, "tweezers": s.tweezers, "pusher": s.pusher, "mandrel": m, "stone": s.stone}) >= CLEAR
        level = math.degrees(math.asin(float(np.clip(abs(m.rot[2, 1]), 0.0, 1.0)))) <= MANDREL_HOME[2]
        d = m.pos - MANDREL_START.pos
        mandrel_home = math.hypot(d[0], d[1]) <= MANDREL_HOME[0] and abs(d[2]) <= MANDREL_HOME[1] and level
        tools_home = _near_start(s.tweezers, TWEEZERS_START, TOOL_HOME) and _near_start(s.pusher, PUSHER_START, TOOL_HOME)
        put_back = bool(pen_home and tools_home and mandrel_home and clear and s.at_rest)

        n_closed = sum(1 for i in range(g.N_CLAWS) if self.closed[i] and self.closed_at_home[i]) if home else 0
        held = home and n_closed >= HELD_MIN_CLAWS
        zero = any(self.melted) or any(self.broken) or self.fingered or self.pen_down or self.tipped or self.off_table
        score = W_STONE * in_head + W_HOME * home + W_CLAW * n_closed + W_PUT_BACK * (held and put_back)
        if self.mark:
            score = min(score, MARK_CAP)
        score = 0.0 if zero else round(score, 4)
        self.now = {"score": score, "in_head": in_head, "home": home, "n_closed": n_closed, "held": held, "put_back": put_back, "clear": clear,
                    "pen_home": pen_home, "tools_home": tools_home, "mandrel_home": mandrel_home, "at_rest": bool(s.at_rest), "zero": zero, "mark": self.mark, "melted": any(self.melted), "broken": any(self.broken),
                    "fingered": self.fingered, "pen_down": self.pen_down, "tipped": self.tipped, "off_table": self.off_table, "up": up, "off": off, "tilt": tilt}
        return self.now

    @property
    def score(self) -> float:
        return float(self.now["score"])

    @property
    def full(self) -> bool:
        return self.score >= 1.0 - 1e-9
