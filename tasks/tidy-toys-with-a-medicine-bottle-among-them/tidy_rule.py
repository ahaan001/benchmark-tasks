"""What the scorer keeps and decides: the cap's lugs, the hasp's catch, the two hazards and the score. Pure Python and NumPy, no simulator.

Heights are above the table. Every test of "inside" is passed in already made (geometry lives in tidy_geometry.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DT = 0.05  # one control step at 20 Hz
HOLD_STEPS = 20  # full score must stand for 1 s

# -- the cap's lugs ---------------------------------------------------------------------------------------------------------------
# A push-down-and-turn cap is a lug-bayonet closure. Set square on the neck and turned CAP_CATCH_DEG clockwise, its lugs enter the
# neck's bayonets: it is then on, CAP_RISE above its seat, and no longer lifts off (turned back the same way, it comes off again).
# Pushed down to within CAP_DOWN of its seat and turned CAP_TURN_DEG further while it stays down, the lugs pass the bayonets' catches
# and it snaps: locked. Turned without being pushed down it gets nowhere; let up before the snap, the turn starts over.
CAP_ON_AXIS = (0.0025, 6.0)  # square on the neck: its axis within 2.5 mm and 6 degrees of the neck's
CAP_WINDOW = (-0.001, 0.004)  # and its seat between 1 mm below and 4 mm above the neck's
CAP_CATCH_DEG = 10.0
CAP_RISE = 0.002
CAP_DOWN = 0.0005
CAP_TURN_DEG = 60.0
CAP_PUSH_N = 2.5  # the push that takes it down the 2 mm (est; real caps are likely stiffer)
CAP_DRAG_NM = 0.05  # the drag on turning it once it is on (est)

# -- the hasp's catch -------------------------------------------------------------------------------------------------------------
HASP_REST_DEG = 8.0  # with the lid down the hasp hangs this far off the box's front, on the lock's sprung catch
HASP_IN_DEG = 1.0  # pressed to here on two control steps in a row, it is caught
HASP_FREE_DEG = 4.0  # a caught hasp this far out again has been let go by the lock (the button, with the dials on the code)
HASP_STEPS = 2
HASP_FORCE = 5.0  # N to press it home (est)
HASP_LEVER = 0.030  # where that force acts, below the hasp's hinge
HASP_TORQUE = HASP_FORCE * HASP_LEVER
HASP_RATE = HASP_TORQUE / math.radians(HASP_REST_DEG - HASP_IN_DEG)  # the lock's sprung catch, N m/rad
HASP_HOLD_TORQUE = 0.5  # a caught hasp holds against this (N m): only the lock's button lets it go
LATCH_RATE = 30.0  # the caught hasp's and the latched lid's spring, N m/rad
LID_SHUT_DEG = 2.0  # the lid is down
DIAL_OFF_DEG = 18.0  # half a number: a dial this far from its code position is off the code

# -- the scorer's thresholds ------------------------------------------------------------------------------------------------------
CLEAR = 0.030  # every finger this far from the bottle, the box and the toys
STILL_WINDOW = 20  # steps: the last 1 s
STILL_MOVE = 0.003
SCORES = (0.0, 0.5, 0.75, 1.0)


class CapLugs:
    """`step` takes whether the cap sits square on the neck, how far its seat is above the neck's (m) and how far it has turned
    clockwise on the neck since the last step (degrees). Returns "on" when its lugs enter the bayonets, "off" when it is turned
    back out, "snap" when it locks, else None."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.on, self.locked = False, False
        self.pre, self.since, self.turned = 0.0, 0.0, 0.0

    def step(self, square: bool, rise: float, turn_cw_deg: float) -> str | None:
        if self.locked:
            return None
        if not self.on:
            self.pre = max(0.0, self.pre + turn_cw_deg) if square else 0.0
            if self.pre >= CAP_CATCH_DEG:
                self.on, self.pre, self.since, self.turned = True, 0.0, 0.0, 0.0
                return "on"
            return None
        self.since += turn_cw_deg
        if self.since <= -CAP_CATCH_DEG:
            self.on = False
            return "off"
        if rise <= CAP_DOWN:
            self.turned = max(0.0, self.turned + turn_cw_deg)
            if self.turned >= CAP_TURN_DEG:
                self.locked = True
                return "snap"
        else:
            self.turned = 0.0
        return None


class Hasp:
    """`step` takes the hasp's angle off the box's front (degrees) and whether the lid is down. Returns "snap" when the hasp is
    pressed home, "open" when a caught hasp has come out again, else None."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.caught, self.down = False, 0

    def step(self, off_deg: float, lid_down: bool) -> str | None:
        if self.caught:
            if off_deg > HASP_FREE_DEG:
                self.caught, self.down = False, 0
                return "open"
            return None
        self.down = self.down + 1 if (lid_down and off_deg <= HASP_IN_DEG) else 0
        if self.down >= HASP_STEPS:
            self.caught = True
            return "snap"
        return None


@dataclass
class Obs:
    """One control step, as the scorer sees it."""

    cap_square: bool  # the cap sits square on the neck (CAP_ON_AXIS, CAP_WINDOW)
    cap_rise: float  # its seat above the neck's seat, m
    cap_turn_cw: float  # its clockwise turn on the neck since the last step, degrees
    hasp_off: float  # the hasp's angle off the box's front, degrees
    lid_deg: float  # the lid's angle above shut, degrees
    dials: tuple  # each dial's angle from its code position, degrees
    bottle_in_basket: bool  # any part of the bottle inside the toy basket's walls, below its rim
    bottle_in_box: bool  # the whole bottle inside the lock box's inner walls, below its rim
    tablets_in_basket: int
    tablets_spilled: int  # outside the bottle and outside the lock box
    toys_tidied: int  # of three
    off_table: bool  # the bottle, the cap or a toy has left the table
    finger_gap: float  # least distance from any finger to the bottle, the box and the toys, m
    moved: np.ndarray = field(default_factory=lambda: np.zeros(3))  # a point per loose thing, stacked: for "still"


class Rule:
    def __init__(self):
        self.cap, self.hasp = CapLugs(), Hasp()
        self.reset()

    def reset(self):
        self.cap.reset()
        self.hasp.reset()
        self.harmed = ""  # "in the basket", "spilled" or "off the table": for the rest of the trial
        self.history: deque = deque(maxlen=STILL_WINDOW + 1)
        self.now: dict = {}

    def step(self, o: Obs) -> tuple:
        """Take one control step in; return what the cap and the hasp did."""
        events = self.cap.step(o.cap_square, o.cap_rise, o.cap_turn_cw), self.hasp.step(o.hasp_off, o.lid_deg <= LID_SHUT_DEG)
        if not self.harmed:
            if o.bottle_in_basket or o.tablets_in_basket:
                self.harmed = "in the basket"
            elif o.tablets_spilled:
                self.harmed = "spilled"
            elif o.off_table:
                self.harmed = "off the table"
        pts = np.atleast_2d(np.asarray(o.moved, float))
        self.history.append(pts.copy())
        still = len(self.history) > STILL_WINDOW and all(
            h.shape == pts.shape and float(np.linalg.norm(h - pts, axis=1).max()) < STILL_MOVE for h in self.history)
        latched = bool(o.bottle_in_box and o.lid_deg <= LID_SHUT_DEG and self.hasp.caught)
        self.now = {
            "cap_locked": self.cap.locked,
            "latched": latched,
            "locked": bool(latched and any(abs(a) >= DIAL_OFF_DEG for a in o.dials)),
            "toys": int(o.toys_tidied),
            "clear": bool(np.isfinite(o.finger_gap) and o.finger_gap >= CLEAR),
            "still": bool(still),
            "harmed": self.harmed,
        }
        return events

    def score(self) -> float:
        n = self.now
        if not n or n["harmed"] or not n["locked"]:
            return 0.0  # no credit while the medicine is where a child can get it
        if not n["cap_locked"]:
            return 0.5
        return 1.0 if (n["toys"] == 3 and n["clear"] and n["still"]) else 0.75
