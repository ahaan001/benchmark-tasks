"""Scoring for dealing cards off the top of a pack: pure numpy, no simulator.

A card is a pose (centre, 3x3 rotation). Everything a person checks on a real trial is a function here: is the card
face up, is it wholly inside a box, is it still in the pack, is a finger near it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L


@dataclass
class Card:
    pos: np.ndarray
    rot: np.ndarray  # columns: the card's long axis, short axis and back normal in the world

    @property
    def face_up(self) -> bool:
        return bool(self.rot[2, 2] < 0.0)

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.acos(float(np.clip(abs(self.rot[2, 2]), 0.0, 1.0))))

    def corners(self) -> np.ndarray:
        half = np.array([[sx * L.CARD_L / 2, sy * L.CARD_W / 2, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
        return half @ self.rot.T + self.pos

    def distance(self, point) -> float:
        local = (np.asarray(point, float) - self.pos) @ self.rot
        half = np.array([L.CARD_L, L.CARD_W, L.CARD_T]) / 2
        return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


def in_box(card: Card, box_xy, felt_top: float) -> bool:
    """Flat on the felt with nothing of it over the outer edge of the box's tape."""
    if card.tilt_deg > L.FLAT_DEG or abs(card.pos[2] - (felt_top + L.CARD_T / 2)) > L.ON_FELT:
        return False
    c = card.corners()[:, :2] - np.asarray(box_xy, float)
    return bool(np.all(np.abs(c[:, 0]) <= L.BOX_L / 2 + L.TAPE) and np.all(np.abs(c[:, 1]) <= L.BOX_W / 2 + L.TAPE))


def in_pack(card: Card, pack_pos) -> bool:
    d = card.pos - np.asarray(pack_pos, float)
    return bool(abs(d[0]) <= L.IN_PACK and abs(d[1]) <= L.IN_PACK and d[2] > 0.0 and not card.face_up)


@dataclass
class Trial:
    """Whole-trial memory: cleared only by :meth:`reset` (a real episode reset)."""

    pack_ref: np.ndarray | None = None
    shown: set = field(default_factory=set)  # cards whose face has shown
    spoiled: str = ""  # why the trial is 0 whatever follows
    history: deque = field(default_factory=deque)

    def reset(self) -> None:
        self.pack_ref, self.spoiled = None, ""
        self.shown.clear()
        self.history.clear()

    def update(self, cards: dict[str, Card], pack_pos, table_top: float, dt: float) -> None:
        pack_pos = np.asarray(pack_pos, float)
        if self.pack_ref is None:
            self.pack_ref = pack_pos.copy()
        if not self.spoiled and np.linalg.norm((pack_pos - self.pack_ref)[:2]) > L.PACK_MOVED:
            self.spoiled = "pack moved"
        low = min([pack_pos[2]] + [c.pos[2] for c in cards.values()])
        if not self.spoiled and low < table_top - L.OFF_TABLE:
            self.spoiled = "off the table"
        self.shown.update(n for n, c in cards.items() if c.face_up)
        self.history.append(np.vstack([pack_pos] + [cards[n].pos for n in sorted(cards)]))
        while len(self.history) > max(2, round(L.REST_S / dt) + 1):
            self.history.popleft()

    def at_rest(self, dt: float) -> bool:
        if len(self.history) < round(L.REST_S / dt) + 1:
            return False
        h = np.stack(self.history)
        return bool(np.linalg.norm(h.max(axis=0) - h.min(axis=0), axis=1).max() <= L.REST_M)


def tally(trial: Trial, cards: dict[str, Card], pack_pos, felt_top: float) -> dict:
    """Which box holds which cards, which cards are dealt, which are strays."""
    boxes = [[n for n, c in cards.items() if in_box(c, xy, felt_top)] for xy in L.BOXES]
    dealt = [b[0] if len(b) == 1 and b[0] not in trial.shown else None for b in boxes]
    kept = {n for n, c in cards.items() if in_pack(c, pack_pos)}
    spare = {b[0] for b in boxes if len(b) > 1}  # the first card of a shared box is neither dealt nor a stray
    strays = [n for n in cards if n not in kept and n not in dealt and n not in spare]
    return dict(boxes=boxes, dealt=dealt, strays=strays)


def score(trial: Trial, cards: dict[str, Card], pack_pos, felt_top: float, fingers, dt: float) -> float:
    if trial.spoiled:
        return 0.0
    t = tally(trial, cards, pack_pos, felt_top)
    n = sum(d is not None for d in t["dealt"])
    value = max(0.0, 0.25 * n - 0.25 * len(t["strays"]))
    if n == len(L.BOXES) and not t["strays"]:
        points = np.asarray(fingers, float).reshape(-1, 3)
        if np.isnan(points).any():
            return 0.75  # the fingers could not be read: no full score
        near = min((cards[d].distance(p) for d in t["dealt"] for p in points), default=math.inf)
        return 1.0 if near >= L.CLEAR and trial.at_rest(dt) else 0.75
    return value
