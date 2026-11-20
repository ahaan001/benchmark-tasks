"""The thread rule and the score. Pure Python and NumPy: nothing here needs the simulator, and every line is tested on its own.

The thread rule (declared in the fidelity note). A nut turns on its screw on a hinge and runs along it on a slide. Thread friction of
T_RUN on the hinge carries the nut round with its screw until something stops it. From then on every turn of the screw in the nut moves
the slide's target by one pitch, on a spring of K_THREAD, so that a nut held back by the slot's lips is pulled against them harder with
every turn, and the torque the screw takes to turn goes up with that pull (the usual 0.2 x diameter x clamp load). Nothing here places
or turns a nut: whether it is in the slot, how far it swings and what stops it are all contact.

The score reads the end state only: where the two nuts are, how far each thread is wound, where the foot is, and whether the grippers and
the driver have let go of the bracket.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import tslot_geometry as g

DT = 0.05
HOLD_STEPS = 20  # 1 s at 20 Hz

# -- the thread --------------------------------------------------------------------------------------------------------------------------------
K_THREAD = 40000.0  # N/m: the pull on a held-back nut per metre the thread is wound past it (bracket, lips and screw together; est)
T_RUN = 0.005  # N m: what carries a free nut round with its screw, also when its boss rubs on a lip (est; a turn of thread-seal tape on a real screw)
T_HEAD = 0.001  # N m: the screw's own drag in the foot
NUT_FACTOR = 0.2  # tightening torque = 0.2 x thread diameter x clamp load
THREAD_D = 0.006
T_MAX = 0.25  # N m: more than the gripper passes through the T-handle
FRICTION_STEP = 0.0005  # N m: a nut's friction is written to the simulator when it has changed by this much

# -- what counts ---------------------------------------------------------------------------------------------------------------------------------
GRIP_TURNS = 0.25  # a nut grips once its screw has turned this far past the point where the nut met the lips
TIGHT_TURNS = 0.5  # where the scripted demo stops turning
UNDER_LIP = 0.001  # the middle of each end of the flange lies at least this far under its lip (a nut turned 40 degrees or more)
BEARING = 0.00025  # slack in reading where the lips' underside is
IN_SLOT = 0.001  # "in the slot": the flange's top at least this far below the bar's top face, over the slot's opening
LINE_FULL, LINE_PART = 0.002, 0.005  # the notch's point from the middle of the line, along the bar
POINTS_IN = 10.0  # degrees: the arm lies over the inside of the frame, square to the bar within this
CLEAR = 0.010  # every finger and the driver at least this far from the bracket
STILL = (0.0003, 0.5)  # the bracket has not moved this many metres or turned this many degrees over the last second


class Thread:
    """One nut on its screw: follows the nut's turn on the screw and gives the slide's target, the lag and the nut's friction."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.turn, self.last, self.friction = 0.0, None, T_RUN

    def step(self, angle: float, travel: float) -> tuple:
        """`angle`: the nut's hinge on its screw (rad, anticlockwise from above positive; the reading may wrap). `travel`: its slide (m, up the
        screw positive). Returns (the slide's target, the lag of the nut behind it, the friction the hinge should now have)."""
        if self.last is None:
            self.turn = float(angle)
        else:
            self.turn += math.remainder(float(angle) - self.last, 2.0 * math.pi)
        self.last = float(angle)
        target = min(max(g.PITCH * self.turn / (2.0 * math.pi), g.TRAVEL[0]), g.TRAVEL[1])
        lag = target - float(travel)
        self.friction = min(T_RUN + NUT_FACTOR * THREAD_D * K_THREAD * abs(lag), T_MAX)
        return target, lag, self.friction


# -- the end state ---------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Nut:
    """Where one nut is against the marked bar's slot."""

    along: float  # along the bar (station x, m)
    across: float  # from the slot's centre line (m)
    top: float  # the flange's top face below the bar's top face (m)
    turn: float  # from lengthwise (degrees, 0 to 90)
    lag: float  # how far the thread is wound past where the nut stopped (m)

    @property
    def in_slot(self) -> bool:
        return self.top >= IN_SLOT and abs(self.across) <= g.SLOT_OPEN / 2 and g.SIDE_SPAN[0] <= self.along <= g.SIDE_SPAN[1]

    @property
    def under_lips(self) -> bool:
        return self.in_slot and self.top >= g.LIP - BEARING and min(g.wing_reach(self.turn, self.across)) >= UNDER_LIP

    @property
    def grips(self) -> bool:
        """Under both lips and wound a quarter turn against them: by its flange's top or, if it stopped short of square, by its boss."""
        return self.under_lips and self.top <= g.LIP + g.NUT_E + BEARING and self.lag >= GRIP_TURNS * g.PITCH


@dataclass(frozen=True)
class Obs:
    nuts: tuple  # (Nut, Nut)
    origin: np.ndarray  # the middle of the foot's underside, station frame, z above the table
    rot: np.ndarray  # the bracket's rotation
    finger_gap: float  # the least distance from any finger to the bracket (nan: not known, which counts as holding)
    driver_gap: float  # and from the driver to the bracket


def off_line(obs: Obs) -> float:
    """The notch's point from the middle of the line, along the bar (m)."""
    return abs(float((obs.origin + obs.rot @ g.NOTCH_POINT)[0]) - g.LINE_X)


def points_in(obs: Obs) -> bool:
    """The arm lies over the inside of the frame: it points to the robot's left, square to the bar."""
    return float(obs.rot[1, 1]) >= math.cos(math.radians(POINTS_IN)) and float(obs.rot[2, 2]) >= math.cos(math.radians(POINTS_IN))


def ladder(obs: Obs) -> float:
    """The score of an end state in which the grippers and the driver are clear and the bracket is still."""
    n = sum(nut.grips for nut in obs.nuts)
    if n == 2:
        if points_in(obs) and off_line(obs) <= LINE_FULL:
            return 1.0
        if points_in(obs) and off_line(obs) <= LINE_PART:
            return 0.75
        return 0.5
    if n == 1:
        return 0.5
    if all(nut.in_slot for nut in obs.nuts):
        return 0.25
    return 0.0


class Rule:
    """Keeps the last second of the bracket's poses and scores the state now."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.poses: deque = deque(maxlen=HOLD_STEPS + 1)
        self.now: dict = {}

    def still(self) -> bool:
        if len(self.poses) <= HOLD_STEPS:
            return False
        p0, r0 = self.poses[-1]
        for p, r in self.poses:
            turn = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(r0.T @ r)) - 1.0) / 2.0))))
            if float(np.linalg.norm(p - p0)) > STILL[0] or turn > STILL[1]:
                return False
        return True

    def step(self, obs: Obs) -> float:
        self.poses.append((np.array(obs.origin, float), np.array(obs.rot, float)))
        clear = bool(obs.finger_gap >= CLEAR) and bool(obs.driver_gap >= CLEAR)  # a nan gap is not clear
        still = self.still()
        rung = ladder(obs)
        self.now = {"grips": tuple(n.grips for n in obs.nuts), "under": tuple(n.under_lips for n in obs.nuts), "in_slot": tuple(n.in_slot for n in obs.nuts),
                    "off_line_mm": round(off_line(obs) * 1000, 2), "points_in": points_in(obs), "clear": clear, "still": still, "rung": rung}
        return rung if clear and still else 0.0
