"""deal_demo: the closed-loop scripted demo that yam_scripted runs for this task. NumPy only.

It reads the cards' poses from the simulator each control step (a scripted demo may) and commands joint targets.
One card at a time, with the fingers 4 cm apart and pointing straight down:

1. Both fingertips come down on the back half of the top card and push it forward until most of it hangs over the
   pack's front edge, the tips stopping short of the edge so they never lever on the overhang.
2. The tips lift. The card tips over the edge and comes to rest leaning, its front edge on the felt.
3. The tips come down on the card near that front edge and slide it across the felt to its box. Its back end leaves
   the pack on the way and the card drops flat.
4. The tips lift straight up.

``DEAL_MODE`` (environment variable): ``full`` (four cards), ``one`` (one card to box 2), and two poor deals to
check the scorer. ``light``: the first touch stops 1 mm above the card, so the push carries nothing; the demo then
goes on to its second touch with the card still on the pack and drags it straight over the pack's edge under the
fingertips, which takes the second card with it. ``edge``: the tips come down 1 cm behind the pack and 1.5 mm below
its top, so they push on the back edges of the top cards and shear the pack.
"""

from __future__ import annotations

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
PRESS = -0.001 if MODE == "light" else 0.0004  # the tips are commanded this far into the surface they ride on
PRESS_FELT = 0.0006
BACK = 0.032  # the tips land this far behind the card's centre
SHORT = 0.008  # ... and stop this far short of the pack's front edge
FRONT = 0.015  # second touch: this far behind the card's front edge
FREE, SLOW, PUSH, DRAG = 0.30, 0.02, 0.04, 0.06  # m/s
HOVER = 0.03
DOWN = ik.matrix_to_quat_xyzw(grip_rotation(0.0))


def jobs(cards: list[str]) -> list[tuple[str, str, tuple[float, float]]]:
    """(arm, card, box centre) in dealing order."""
    if MODE in ("one", "light", "edge"):
        return [("left", cards[0], L.BOXES[1])]
    return [("left", cards[0], L.BOXES[0]), ("left", cards[1], L.BOXES[1]),
            ("right", cards[2], L.BOXES[2]), ("right", cards[3], L.BOXES[3])]


class Dealer:
    def __init__(self, bases: dict, dt: float, cards: list[str], pack: str, read=cardkit.poses) -> None:
        self.bases = {a: np.asarray(b, float) for a, b in bases.items()}
        self.dt, self.read, self.pack = dt, read, pack
        self.q = {a: np.array(READY, float) for a in self.bases}
        self.tcp = {a: ik.tcp_pose(self.q[a], self.bases[a])[0].copy() for a in self.bases}
        self.grip = dict.fromkeys(self.bases, 1.0)
        self.todo = jobs(cards)
        self.engaged: set = set()  # arms that have left the ready pose (pointing straight down since)
        self.swing: dict = {}  # arm -> [from joints, to joints, elapsed, duration]: a joint-space move
        self.phase, self.clock, self.last, self.still, self.anchor = "over", 0.0, None, 0, None
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
        if not self.todo:
            done = all([self._swing(a, READY) for a in sorted(self.engaged)])
            if done:
                self.engaged.clear()
            return self._action(), done
        arm, card, box = self.todo[0]
        state = self.read([card, self.pack])
        if card not in state:  # before the first simulator step
            return self._action(), False
        (pos, rot), (pack_pos, _) = state[card], state[self.pack]
        top, _, axis = self._plane(pos, rot)
        self.grip[arm] = GRIP
        edge = pack_pos[0] + L.CARD_L / 2
        if MODE == "edge":  # a poor deal: beside the pack, on the edges of its top cards
            top = top - [0.0, 0.0, 0.0015]
        if self.phase == "over":
            x = pos[0] - BACK if MODE != "edge" else pack_pos[0] - L.CARD_L / 2 - 0.010
            hover = self._tips(x, pos[1], top[2], PRESS, HOVER)
            if arm not in self.engaged:
                if self.anchor is None:
                    self.anchor, _, _ = ik.solve(hover, DOWN, np.array(READY, float), base_pos=self.bases[arm], iters=300)
                if self._swing(arm, self.anchor):
                    self.engaged.add(arm)
                    self.tcp[arm] = hover
                    self._next("touch")
            elif self._move(arm, hover, FREE):
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
            self.still = self.still + 1 if self.last is not None and np.linalg.norm(pos - self.last) < 1e-4 else 0
            self.last = pos.copy()
            waited = self._wait(0.5)
            if up and waited and (self.still >= 4 or self.clock > 2.0):
                self._next("over the front")
        elif self.phase == "over the front":
            g = top + axis * (L.CARD_L / 2 - FRONT)
            if self._move(arm, self._tips(g[0], g[1], g[2], PRESS_FELT, 0.012), FREE / 2):
                self._next("touch the front")
        elif self.phase == "touch the front":
            x, y = self.tcp[arm][:2]
            if self._move(arm, self._tips(x, y, self._surface(pos, rot, x, y), PRESS_FELT), SLOW) and self._wait(0.2):
                self._next("slide")
        elif self.phase == "slide":
            err = np.asarray(box, float) - pos[:2]
            dist = float(np.linalg.norm(err))
            step = min(DRAG * self.dt, dist)
            x, y = self.tcp[arm][:2] + (err / dist * step if dist > 1e-6 else 0.0)
            self.tcp[arm] = self._tips(x, y, self._surface(pos, rot, x, y), PRESS_FELT)
            if dist < 0.002 or self._wait(12.0):
                self._next("settle")
        elif self.phase == "settle":
            if self._wait(0.3):
                self._next("lift")
        elif self.phase == "lift":
            if self.anchor is None:
                self.anchor = self.tcp[arm].copy()
            if self._move(arm, self.anchor + [0.0, 0.0, 0.04], 0.08):
                self.todo.pop(0)
                self._next("over")
        return self._action(), False

    def _action(self):
        out = {}
        for arm in ("left", "right"):
            if arm in self.engaged and arm not in self.swing:
                q, pe, _ = ik.solve(self.tcp[arm], DOWN, self.q[arm], base_pos=self.bases[arm], iters=60)
                self.q[arm], self.worst_ik = q, max(self.worst_ik, pe)
            out[arm] = np.r_[self.q[arm], self.grip[arm]]
        try:
            from inspect_robots_yam.packing import pack
        except ImportError:  # offline tests
            return np.r_[out["left"], out["right"]]
        return pack(out["left"], out["right"])
