"""The declared wax-weld rule and the scorer's memory. NumPy only: the task file feeds it poses each control step, and the tests feed it traces.

A joint heats only while the knife's blade touches the pattern's gate end face and the sprue at once, with the blade over the middle of the gate's face.
After the blade leaves, the gate must come onto the sprue within T_OPEN with enough heat; in the first T_ADJUST it may be set square (the set starts
again), and it must then stay within SET_POS and SET_ANG for T_SET, and is then welded (the task file locks its joints). Leaving the band, or waiting too long, makes the joint cold: heat it again.
Past D_MAX the gate end melts back. Nothing here moves, holds or lets go of a part.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import cluster as c

D_MIN, D_MAX = 2.0, 8.0  # s of blade contact on both faces (E)
T_OPEN = 4.0  # s from the blade leaving to the gate on the sprue (E)
T_SET = 10.0  # s held still for the wax to set (E)
T_ADJUST = 2.0  # s after the faces meet while the wax is still liquid: the pattern may be set square, and the set starts again (E)
SET_POS, SET_ANG = 0.001, 2.0  # m, deg
CONTACT = 0.0005  # m: a blade point this close to a face touches it
COVER = 0.002  # m: the blade must cover the gate face's centre by this much
MARK_TOL, SQUARE_TOL = 0.003, 3.0  # m, deg
CRUSH = 0.001  # m of finger closure past a root's width
KNOCK_POS, KNOCK_ANG = 0.002, 2.0
PAD = 0.003  # m: the blade this close to a finger point burns the pad
HELD = 0.010  # m: a finger point this close to a pattern still touches it
CLEAR = 0.03
REST_S = 1.0
KNIFE_HOME_POS, KNIFE_HOME_ANG = 0.02, 10.0

SPRUE_C = np.array([c.CX, c.CY, (c.SPRUE_Z0 + c.SPRUE_Z1) / 2])
SPRUE_H = np.array([c.SPRUE / 2, c.SPRUE / 2, (c.SPRUE_Z1 - c.SPRUE_Z0) / 2])
GATE_FACE = np.array([[0.0, 0.0, 0.0]] + [[sx * c.GATE[0] / 2, 0.0, sz * c.GATE[2] / 2] for sx in (-1, 1) for sz in (-1, 1)])
MARKED = [b for b in c.BOXES if b[0] != "gate"]  # the knife must never touch these


@dataclass
class Obs:
    """One control step: the episode clock, the knife's pose, the three loose patterns' and three neighbours' poses (R, p), both arms' finger points
    (24, 3: left 12, right 12) and their openings (half the gap between the pads, per arm)."""

    t: float
    knife: tuple
    loose: list
    neighbours: list
    tips: np.ndarray
    opening: tuple = (0.04, 0.04)


@dataclass
class Joint:
    dose: float = 0.0
    slot: int | None = None  # the mark the joint was heated at
    off_t: float | None = None  # when the blade last left a hot joint
    wet_t: float | None = None
    set_t: float | None = None  # when the pattern was last put where it now sets
    anchor: tuple | None = None
    welded: bool = False
    weld_pose: tuple | None = None
    melt_back: bool = False
    marked: bool = False


@dataclass
class Rule:
    joints: list = field(default_factory=lambda: [Joint() for _ in range(3)])
    neighbour_marked: list = field(default_factory=lambda: [False] * 3)
    knock: bool = False
    crush: bool = False
    pad: bool = False
    t_prev: float | None = None
    rest_t: float = 0.0
    locks: list = field(default_factory=list)  # loose pattern indices to lock this step: the task file empties it
    events: list = field(default_factory=list)  # what happened to each joint, for the log: the task file empties it

    def reset(self) -> None:
        self.__init__()

    # -- geometry
    @staticmethod
    def blade_world(o: Obs) -> np.ndarray:
        R, p = o.knife
        return c.to_world(R, p, c.BLADE_PTS)

    @staticmethod
    def touches_gate_end(blade: np.ndarray, pose) -> bool:
        R, p = pose
        q = c.to_local(R, p, blade)
        name, ctr, half, rot = c.BOXES[0]
        d, near = c.box_distance(q, ctr, half, rot)
        on_end = (d <= CONTACT) & (near[:, 1] >= half[1] - 1e-6)
        if not np.any(on_end):
            return False
        # the blade's face must cover the middle of the gate's face
        bx, bz = q[:, 0], q[:, 2]
        return bool(bx.min() <= -COVER and bx.max() >= COVER and bz.min() <= -COVER and bz.max() >= COVER)

    @staticmethod
    def touches_sprue(points: np.ndarray, need: int = 1) -> bool:
        d, _ = c.box_distance(points, SPRUE_C, SPRUE_H, np.eye(3))
        return int(np.sum(d <= CONTACT)) >= need

    @staticmethod
    def near_boxes(points: np.ndarray, pose, boxes, dist: float) -> bool:
        R, p = pose
        q = c.to_local(R, p, points)
        return any(np.any(c.box_distance(q, ctr, half, rot)[0] <= dist) for _, ctr, half, rot in boxes)

    @staticmethod
    def moved(a, b, pos: float, ang: float) -> bool:
        return float(np.linalg.norm(np.asarray(a[1]) - np.asarray(b[1]))) > pos or c.angle_deg(a[0], b[0]) > ang

    # -- one control step
    def step(self, o: Obs) -> None:
        if self.t_prev is not None and o.t < self.t_prev - 1e-9:
            self.reset()  # a new episode: the clock went back
        if self.t_prev is not None and abs(o.t - self.t_prev) < 1e-9:
            return  # this step was already seen
        dt = 0.0 if self.t_prev is None else o.t - self.t_prev
        self.t_prev = o.t
        blade = self.blade_world(o)
        on_sprue = self.touches_sprue(blade)
        for i, pose in enumerate(o.loose):
            j = self.joints[i]
            if self.near_boxes(blade, pose, MARKED, CONTACT):
                j.marked = True
            if j.welded:
                if self.moved(pose, j.weld_pose, KNOCK_POS, KNOCK_ANG):
                    self.knock = True
                continue
            at = self.at_mark(pose)
            heating = on_sprue and at is not None and (j.slot in (None, at)) and self.touches_gate_end(blade, pose)
            if heating:
                j.dose += dt
                j.slot = at
                j.off_t = None
                j.wet_t = None
                if j.dose > D_MAX:
                    j.melt_back = True
            elif j.dose > 0.0 and j.wet_t is None:
                j.off_t = o.t if j.off_t is None else j.off_t
                gate_on = self.touches_sprue(c.to_world(*pose, GATE_FACE), need=3)
                if gate_on and j.dose >= D_MIN and o.t - j.off_t <= T_OPEN and self.at_mark(pose) == j.slot:
                    j.wet_t, j.set_t, j.anchor = o.t, o.t, pose
                    self.events.append(f"{i}: closed with {j.dose:.1f} s of heat")
                elif o.t - j.off_t > T_OPEN:
                    self.events.append(f"{i}: cold, {T_OPEN:g} s after the blade left without the gate on the sprue")
                    j.dose, j.off_t, j.slot = 0.0, None, None  # cold: heat it again
            if j.wet_t is not None and not heating:
                if self.moved(pose, j.anchor, SET_POS, SET_ANG):
                    if o.t - j.wet_t <= T_ADJUST and self.touches_sprue(c.to_world(*pose, GATE_FACE), need=3) and self.at_mark(pose) == j.slot:
                        j.set_t, j.anchor = o.t, pose  # still liquid: set square, and the set starts again
                    else:
                        moved = (np.linalg.norm(np.asarray(pose[1]) - np.asarray(j.anchor[1])) * 1000, c.angle_deg(pose[0], j.anchor[0]))
                        self.events.append(f"{i}: cracked {o.t - j.set_t:.1f} s into the set, moved {moved[0]:.1f} mm, {moved[1]:.1f} deg")
                        j.dose, j.off_t, j.wet_t, j.set_t, j.anchor, j.slot = 0.0, None, None, None, None, None  # cracked while setting
                elif o.t - j.set_t >= T_SET:
                    j.welded, j.weld_pose = True, pose
                    self.events.append(f"{i}: welded")
                    self.locks.append(i)
        for k, pose in enumerate(o.neighbours):
            ref = c.NEIGHBOURS[k]
            if self.moved(pose, (ref[2], ref[1]), KNOCK_POS, KNOCK_ANG):
                self.knock = True
            if self.near_boxes(blade, pose, MARKED, CONTACT):
                self.neighbour_marked[k] = True
        if np.any(np.linalg.norm(o.tips[:, None, :] - blade[None, :, :], axis=2) <= PAD):
            self.pad = True
        self._crush(o)
        self.rest_t = self.rest_t + dt if self._clear(o) else 0.0

    def _crush(self, o: Obs) -> None:
        """A hand's two fingers on either side of a loose pattern's platform, closed past its 30 mm by CRUSH."""
        name, ctr, half, rot = c.BOXES[2]
        for arm in (0, 1):
            tips = o.tips[12 * arm:12 * arm + 12]
            for pose in o.loose:
                q = c.to_local(*pose, tips) - ctr
                a, b = q[:6], q[6:]
                for ax in (0,):
                    other = [k for k in (0, 1, 2) if k != ax]
                    inside = lambda P: (np.abs(P[:, 1]) <= half[1] + 0.004) & (np.abs(P[:, 2]) <= half[2])  # noqa: E731  level with the platform, not above it
                    if np.any(inside(a)) and np.any(inside(b)) and np.sign(a[:, ax].mean()) != np.sign(b[:, ax].mean()):
                        if 2 * o.opening[arm] < 2 * half[ax] - CRUSH:
                            self.crush = True

    def _clear(self, o: Obs) -> bool:
        """Every finger point CLEAR of the knife, every pattern and the sprue."""
        knife = c.to_world(*o.knife, c.KNIFE_PTS)
        if np.any(np.linalg.norm(o.tips[:, None, :] - knife[None, :, :], axis=2) < CLEAR):
            return False
        for pose in list(o.loose) + [(r, q) for _, q, r in c.NEIGHBOURS]:
            if self.near_boxes(o.tips, pose, c.BOXES, CLEAR):
                return False
        return not self.touches_sprue_within(o.tips, CLEAR)

    @staticmethod
    def touches_sprue_within(points, dist) -> bool:
        return bool(np.any(c.box_distance(points, SPRUE_C, SPRUE_H, np.eye(3))[0] <= dist))

    @staticmethod
    def at_mark(pose) -> int | None:
        """The mark a gate's end is at: on the sprue's -y face (within 4 mm off it), centred within MARK_TOL across and along the sprue."""
        p = np.asarray(pose[1])
        if abs(p[0] - c.CX) > MARK_TOL or not -0.004 <= p[1] - c.FACE_Y <= 0.001:
            return None
        for k, z in enumerate(c.SLOTS_Z):
            if abs(p[2] - z) <= MARK_TOL:
                return k
        return None

    # -- the score
    def on_mark(self, pose) -> int | None:
        R, p = pose
        if c.angle_deg(R, np.eye(3)) > SQUARE_TOL or abs(p[0] - c.CX) > MARK_TOL or abs(p[1] - c.FACE_Y) > MARK_TOL:
            return None
        for k, z in enumerate(c.SLOTS_Z):
            if abs(p[2] - z) <= MARK_TOL:
                return k
        return None

    def knife_home(self, o: Obs) -> bool:
        R, p = o.knife
        along = float(np.clip(np.asarray(R)[:, 0] @ c.KNIFE_REST_R[:, 0], -1.0, 1.0))  # its axis as it lay; a roll about its own round handle is still in the rest
        return float(np.linalg.norm(np.asarray(p) - c.KNIFE_REST)) <= KNIFE_HOME_POS and np.degrees(np.arccos(along)) <= KNIFE_HOME_ANG

    def parts(self, o: Obs) -> dict:
        out = {"patterns": [], "marks": 0, "knife_home": self.knife_home(o), "clear": self.rest_t >= REST_S}
        for i, pose in enumerate(o.loose):
            j = self.joints[i]
            held = self.near_boxes(o.tips, pose, c.BOXES, HELD)
            k = self.on_mark(pose) if j.welded and not held else None
            out["patterns"].append({"welded": j.welded, "slot": k, "melt_back": j.melt_back, "held": held})
        out["marks"] = sum(j.marked for j in self.joints) + sum(self.neighbour_marked)
        return out

    def score(self, o: Obs) -> float:
        if self.knock or self.crush or self.pad:
            return 0.0
        p = self.parts(o)
        s, slots = 0.0, set()
        for q in p["patterns"]:
            if q["slot"] is not None and q["slot"] not in slots:
                slots.add(q["slot"])
                s += 0.25 - (0.10 if q["melt_back"] else 0.0)
        if slots:
            s += 0.15 if p["knife_home"] else 0.0
            s += 0.10 if p["clear"] else 0.0
        s -= 0.10 * p["marks"]
        return round(max(0.0, s), 4)
