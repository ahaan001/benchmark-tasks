"""deal_demo: the closed-loop scripted demo that yam_scripted runs for this task. NumPy only.

It reads the cards' poses from the simulator each control step (a scripted demo may) and commands joint targets.
One card at a time, hands pointing straight down:

1. Fingers 4 cm apart: both fingertips come down on the back half of the top card and push it forward until most of
   it hangs over the pack's front edge, the tips stopping short of the edge so they never lever on the overhang.
2. The tips lift. The card tips over the edge and comes to rest leaning, its front edge on the felt.
3. The fingers close into one fingertip. It comes down on the card near that front edge, on its centre line, and
   draws it straight along its own length until its back end is 3 cm past the pack and it lies flat on the felt.
4. The fingertip lifts, comes down again 2 cm from the card's middle, on the side the card has to go, and slides it
   straight to its box. Led from in front of its middle a card follows like a trailer and keeps its heading; pushed
   from behind its middle, or drawn under two fingertips that do not press alike, it swings round.
5. The fingertip lifts straight up.

``DEAL_MODE`` (environment variable): ``full`` (four cards), ``one`` (one card to box 2), and two poor deals to
check the scorer. ``light``: the first touch stops 1 mm above the card, so the push carries nothing; the demo then
goes on to its second touch with the card still on the pack and drags it straight over the pack's edge under the
fingertips, which takes the second card with it. ``edge``: the tips come down 1 cm behind the pack and 1.5 mm below
its top, so they push on the back edges of the top cards and shear the pack.
"""

from __future__ import annotations

import math
import os

import numpy as np

import cardkit
import layout as L
from yam_arena import ik
from yam_arena.constants import PAD_TIP_PAST_TCP, READY
from yam_arena.demo import grip_rotation

MODE = os.environ.get("DEAL_MODE", "full")
DEBUG = bool(os.environ.get("TASK_DEBUG"))
HALF_GAP = 0.020  # each fingertip this far from the card's centre line
GRIP = (HALF_GAP - 0.0015) / 0.047  # share of fully open: the tips' inner faces are 47 mm from the centre when open
PRESS = -0.001 if MODE == "light" else 0.0007  # the tips are commanded this far into the surface they ride on
PRESS_DRAW = 0.0005  # second touch, on the leaning card
PRESS_FELT = 0.0008  # once the card lies on the felt
CLEAR_PACK = 0.03  # the card is drawn until its back edge is this far past the pack
LEAD = 0.020  # the one fingertip that carries a card rides this far ahead of the card's middle
BACK = 0.032  # the tips land this far behind the card's centre
SHORT = 0.008  # ... and stop this far short of the pack's front edge
FRONT = 0.015  # second touch: this far behind the card's front edge
FREE, SLOW, PUSH, DRAG = 0.30, 0.02, 0.04, 0.06  # m/s
HOVER = 0.03
ONE_TIP = ("over the front", "touch the front", "draw", "rise", "over the lead", "touch the lead", "slide", "place",
           "settle", "lift")  # phases with the fingers closed into one fingertip
DOWN = ik.matrix_to_quat_xyzw(grip_rotation(0.0))


def jobs(cards: list[str]) -> list[tuple[str, str, tuple[float, float]]]:
    """(arm, card, box centre) in dealing order."""
    if MODE in ("one", "light", "edge"):
        return [("left", cards[0], L.BOXES[int(os.environ.get("DEAL_BOX", "2")) - 1])]
    # each arm deals its outer box first, so no card is carried past one already dealt
    return [("left", cards[0], L.BOXES[0]), ("left", cards[1], L.BOXES[1]),
            ("right", cards[2], L.BOXES[3]), ("right", cards[3], L.BOXES[2])]


class Dealer:
    def __init__(self, bases: dict, dt: float, cards: list[str], pack: str, read=cardkit.poses, todo=None,
                 phase: str = "over") -> None:
        self.bases = {a: np.asarray(b, float) for a, b in bases.items()}
        self.dt, self.read, self.pack = dt, read, pack
        self.q = {a: np.array(READY, float) for a in self.bases}
        self.tcp = {a: ik.tcp_pose(self.q[a], self.bases[a])[0].copy() for a in self.bases}
        self.grip = dict.fromkeys(self.bases, 1.0)
        self.todo = jobs(cards) if todo is None else list(todo)
        self.engaged: set = set()  # arms that have left the ready pose (pointing straight down since)
        self.swing: dict = {}  # arm -> [from joints, to joints, elapsed, duration]: a joint-space move
        self.lead = np.zeros(2)  # unit vector from the card to its box, fixed when the fingertip is set down
        self.homing = None  # an arm on its way back to the ready pose, out of the other arm's way
        self.phase, self.clock, self.last, self.still, self.anchor = phase, 0.0, None, 0, None
        self.worst_ik = 0.0

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _move(self, arm: str, target, speed: float) -> bool:
        d = np.asarray(target, float) - self.tcp[arm]
        dist = float(np.linalg.norm(d))
        step = speed * self.dt
        self.tcp[arm] = np.asarray(target, float).copy() if dist <= step else self.tcp[arm] + d * step / dist
        return dist <= step

    def _swing(self, arm: str, q_to, seconds: float = 1.6) -> bool:
        """Joint-space move (to or from the ready pose, whose hand is not pointing down); True when done."""
        if arm not in self.swing:
            self.swing[arm] = [self.q[arm].copy(), np.asarray(q_to, float), 0.0, seconds]
        q0, q1, t, total = self.swing[arm]
        t = min(t + self.dt, total)
        self.swing[arm][2] = t
        k = ik.smoothstep(t / total)
        self.q[arm] = q0 + k * (q1 - q0)
        self.tcp[arm] = ik.tcp_pose(self.q[arm], self.bases[arm])[0].copy()
        if t >= total:
            del self.swing[arm]
            return True
        return False

    def _reach(self, arm: str, target, speed: float) -> bool:
        """Cartesian move, or a joint-space swing if the arm is still at its ready pose."""
        if arm in self.engaged:
            return self._move(arm, target, speed)
        if arm not in self.swing:
            self.goal_q, _, _ = ik.solve(target, DOWN, np.array(READY, float), base_pos=self.bases[arm], iters=300)
        if self._swing(arm, self.goal_q):
            self.engaged.add(arm)
            self.tcp[arm] = np.asarray(target, float).copy()
            return True
        return False

    def _wait(self, seconds: float) -> bool:
        self.clock += self.dt
        return self.clock >= seconds

    def _next(self, phase: str) -> None:
        if DEBUG:
            print(f"[demo] {phase}", flush=True)
        self.phase, self.clock, self.last, self.still, self.anchor = phase, 0.0, None, 0, None

    @staticmethod
    def _plane(pos, rot):
        """Upper face of a card: a point on it, its upward normal, its long axis pointing away from the robot."""
        n = rot[:, 2] if rot[2, 2] > 0 else -rot[:, 2]
        axis = rot[:, 0] if rot[0, 0] > 0 else -rot[:, 0]
        return pos + n * L.CARD_T / 2, n, axis

    def _surface(self, pos, rot, x: float, y: float) -> float:
        top, n, _ = self._plane(pos, rot)
        return float(top[2] - (n[0] * (x - top[0]) + n[1] * (y - top[1])) / n[2])

    def _tips(self, x: float, y: float, surface: float, press: float, lift: float = 0.0) -> np.ndarray:
        return np.array([x, y, surface + PAD_TIP_PAST_TCP - press + lift])

    # -- one control step ------------------------------------------------------------------------------------------
    def act(self, observation=None):
        if self.homing is not None:
            if self._swing(self.homing, READY):
                self.engaged.discard(self.homing)
                self.grip[self.homing] = 1.0
                self.homing = None
            return self._action(), False
        if not self.todo:
            return self._action(), True
        arm, card, box = self.todo[0]
        state = self.read([card, self.pack])
        if card not in state:  # before the first simulator step
            return self._action(), False
        (pos, rot), (pack_pos, _) = state[card], state[self.pack]
        top, _, axis = self._plane(pos, rot)
        self.grip[arm] = 0.0 if self.phase in ONE_TIP else GRIP
        edge = pack_pos[0] + L.CARD_L / 2
        if MODE == "edge":  # a poor deal: beside the pack, on the edges of its top cards
            top = top - [0.0, 0.0, 0.0015]
        if self.phase == "over":
            x = pos[0] - BACK if MODE != "edge" else pack_pos[0] - L.CARD_L / 2 - 0.010
            if self._reach(arm, self._tips(x, pos[1], top[2], PRESS, HOVER), FREE):
                self._next("touch")
        elif self.phase == "touch":
            if self._move(arm, self._tips(self.tcp[arm][0], self.tcp[arm][1], top[2], PRESS), SLOW) and self._wait(0.25):
                self._next("push")
        elif self.phase == "push":
            if self._move(arm, self._tips(edge - SHORT, self.tcp[arm][1], top[2], PRESS), PUSH):
                self._next("let go")
        elif self.phase == "let go":
            if self.anchor is None:
                self.anchor = self.tcp[arm].copy()
            up = self._move(arm, self.anchor + [0.0, 0.0, 0.02], SLOW)
            if self.tcp[arm][2] - self.anchor[2] > 0.008:  # clear of the card: the fingers close into one fingertip
                self.grip[arm] = 0.0
            self.still = self.still + 1 if self.last is not None and np.linalg.norm(pos - self.last) < 1e-4 else 0
            self.last = pos.copy()
            waited = self._wait(0.5)
            if up and waited and (self.still >= 4 or self.clock > 2.0):
                self._next("over the front")
        elif self.phase == "over the front":
            g = top + axis * (L.CARD_L / 2 - FRONT)
            if self._move(arm, self._tips(g[0], g[1], g[2], PRESS_DRAW, 0.012), FREE / 2):
                self._next("touch the front")
        elif self.phase == "touch the front":
            x, y = self.tcp[arm][:2]
            if self._move(arm, self._tips(x, y, self._surface(pos, rot, x, y), PRESS_DRAW), SLOW) and self._wait(0.2):
                self._next("draw")
        elif self.phase == "draw":  # straight along the card's own length, until it lies flat clear of the pack
            x, y = self.tcp[arm][:2] + axis[:2] / np.linalg.norm(axis[:2]) * PUSH * self.dt
            self.tcp[arm] = self._tips(x, y, self._surface(pos, rot, x, y), PRESS_DRAW)
            back = pos[0] - abs(axis[0]) * L.CARD_L / 2 - abs(axis[1]) * L.CARD_W / 2
            flat = abs(rot[2, 2]) > math.cos(math.radians(2.0)) and top[2] < pack_pos[2]
            if (flat and back > edge + CLEAR_PACK) or self._wait(5.0):
                self._next("rise")
        elif self.phase == "rise":  # the fingers close into one fingertip on the way up
            if self.anchor is None:
                self.anchor = self.tcp[arm].copy()
            if self._move(arm, self.anchor + [0.0, 0.0, 0.015], 0.08):
                self._next("over the lead")
        elif self.phase == "over the lead":
            if self.anchor is None:
                self.anchor = True
                d = np.asarray(box, float) - pos[:2]
                self.lead = d / np.linalg.norm(d)
            x, y = pos[:2] + LEAD * self.lead
            if self._reach(arm, self._tips(x, y, top[2], PRESS_FELT, 0.012), FREE / 2) and self._wait(0.4):
                self._next("touch the lead")
        elif self.phase == "touch the lead":
            x, y = self.tcp[arm][:2]
            if self._move(arm, self._tips(x, y, self._surface(pos, rot, x, y), PRESS_FELT), SLOW) and self._wait(0.2):
                self._next("slide")
        elif self.phase == "slide":  # straight to the box; the card trails behind the fingertip
            goal = np.asarray(box, float) + LEAD * self.lead
            d = goal - self.tcp[arm][:2]
            dist = float(np.linalg.norm(d))
            x, y = goal if dist <= DRAG * self.dt else self.tcp[arm][:2] + d / dist * DRAG * self.dt
            self.tcp[arm] = self._tips(x, y, self._surface(pos, rot, x, y), PRESS_FELT)
            if dist <= DRAG * self.dt:
                self._next("place")
        elif self.phase == "place":  # the last few millimetres, on the card's own position
            err = np.asarray(box, float) - pos[:2]
            dist = float(np.linalg.norm(err))
            step = min(DRAG / 2 * self.dt, dist)
            x, y = self.tcp[arm][:2] + (err / dist * step if dist > 1e-6 else 0.0)
            self.tcp[arm] = self._tips(x, y, self._surface(pos, rot, x, y), PRESS_FELT)
            if dist < 0.002 or self._wait(2.0):
                self._next("settle")
        elif self.phase == "settle":
            if self._wait(0.3):
                self._next("lift")
        elif self.phase == "lift":
            if self.anchor is None:
                self.anchor = self.tcp[arm].copy()
            if self._move(arm, self.anchor + [0.0, 0.0, 0.04], 0.08):
                self.todo.pop(0)
                if not self.todo or self.todo[0][0] != arm:
                    self.homing = arm
                self._next("over")
        return self._action(), False

    def _joints(self) -> dict:
        out = {}
        for arm in ("left", "right"):
            if arm in self.engaged and arm not in self.swing:
                q, pe, _ = ik.solve(self.tcp[arm], DOWN, self.q[arm], base_pos=self.bases[arm], iters=60)
                self.q[arm], self.worst_ik = q, max(self.worst_ik, pe)
            out[arm] = np.r_[self.q[arm], self.grip[arm]]
        return out

    def _action(self):
        return pack_action(self._joints())


def pack_action(joints: dict):
    try:
        from inspect_robots_yam.packing import pack
    except ImportError:  # offline tests
        return np.r_[joints["left"], joints["right"]]
    return pack(joints["left"], joints["right"])
