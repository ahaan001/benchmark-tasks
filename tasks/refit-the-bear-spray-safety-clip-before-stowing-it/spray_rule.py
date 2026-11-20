"""What the scorer keeps and decides: the two whole-trial zeros and the score. Pure Python and NumPy, no simulator.

Every geometric test is passed in already made (geometry lives in spray_geometry.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

DT = 0.05  # one control step at 20 Hz
FIRED_DEG = 4.5  # the trigger tab past half its travel
FIRED_STEPS = 2  # on two control steps in a row: one step alone can be a solver's twitch
CLEAR = 0.030  # every finger this far from the can and its clip
STILL_WINDOW = 20  # steps: the last 1 s
STILL_MOVE = 0.003  # nothing has moved further than this in that second
SCORES = (0.0, 0.5, 1.0)


@dataclass
class Obs:
    """One control step, as the scorer sees it."""

    tab_deg: float  # how far the trigger tab is pressed, degrees
    seated: bool  # the clip is home on the tab (spray_geometry.seated)
    can_in_bin: bool  # every part of the can inside the bin's inner walls, seen from above
    finger_gap: float  # least distance from any finger to the can and its clip, m
    moved: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # the can's two ends and the clip, stacked: for "at rest"


class Rule:
    def __init__(self):
        self.reset()

    def reset(self):
        """A new trial. Called on the embodiment's reset and nowhere else: a zero is for the whole trial."""
        self.zero = ""  # "fired" or "stowed bare"
        self.pressed = 0
        self.clear_run = 0
        self.history: deque = deque(maxlen=STILL_WINDOW + 1)
        self.now: dict = {}

    def step(self, o: Obs) -> None:
        self.pressed = self.pressed + 1 if o.tab_deg >= FIRED_DEG else 0
        clear = bool(np.isfinite(o.finger_gap) and o.finger_gap >= CLEAR)
        self.clear_run = self.clear_run + 1 if clear else 0
        pts = np.atleast_2d(np.asarray(o.moved, float))
        self.history.append(pts.copy())
        still = len(self.history) > STILL_WINDOW and all(
            h.shape == pts.shape and float(np.linalg.norm(h - pts, axis=1).max()) < STILL_MOVE for h in self.history)
        at_rest = bool(o.can_in_bin and still and self.clear_run > STILL_WINDOW)  # in the bin, let go, for a second
        if not self.zero:
            if self.pressed >= FIRED_STEPS:
                self.zero = "fired"
            elif at_rest and not o.seated:
                self.zero = "stowed bare"
        self.now = {"seated": bool(o.seated), "in_bin": bool(o.can_in_bin), "clear": clear, "still": bool(still), "at_rest": at_rest, "zero": self.zero}

    def score(self) -> float:
        n = self.now
        if not n or n["zero"] or not n["seated"]:
            return 0.0
        return 1.0 if n["at_rest"] else 0.5
