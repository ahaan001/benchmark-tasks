"""The embedding rules as bookkeeping, pure Python and NumPy: when the piece curls, when the wax sets, and the score. `write_pose` is the one
call into Isaac Lab (imported lazily).

Frames: the station frame with z up from the table. The mould's frame has its origin at the centre of its floor's underside; the piece's at its
centre, its inked cut face towards -z; the forceps' at their closed tips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FLOAT_AFTER = 0.3  # s without the tips on the strip before it curls in molten wax: a dab does not hold it
FLOAT_LIFT = 0.002  # m: its curled end off the floor
FLOAT_TILT = 15.0  # degrees, about the strip's own x axis (a curl, as a rigid strip shows it)
DRIFT_SPEED = 0.01  # m/s: the mould moving faster than this while the strip is curled shifts it
DRIFT_LAG = 0.004  # m the curled strip lags behind the mould's motion, once per curl
SET_AFTER = 8.0  # s on the cold block for the bottom layer to set
FLAT_DEPTH = 0.001  # m: the whole inked face within the first 1 mm of facing (spread of its depth, and its lowest point off the floor)
CENTRE = 0.003  # m from the mould's middle
CLEAR = 0.03  # m from every fingertip point to the mould
TILT_SPILL, LIFT_SPILL = 10.0, 0.01  # degrees and m: the mould tipped or lifted
PRESS_ABOVE = 0.0003  # m: the tips' point on the strip's top face: no higher than this above it (in contact)
PRESS_INTO = 0.001  # m: and no lower than this below it (pushed in, not resting on the floor beside it)


@dataclass
class Geometry:
    plate_z: float
    cold: tuple  # x0, x1, y0, y1
    well: tuple  # x, y
    well_h: float
    mould_in: tuple  # inside x, y
    mould_h: float
    floor: float
    tissue: tuple  # x, y, z
    start_mould: tuple


@dataclass
class Snapshot:
    mould_pos: np.ndarray
    mould_rot: np.ndarray
    mould_vel: np.ndarray
    tissue_pos: np.ndarray
    tissue_rot: np.ndarray
    forceps_pos: np.ndarray
    forceps_rot: np.ndarray
    tips: np.ndarray  # fingertip points of both hands, (n, 3)


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


class Rules:
    def __init__(self, g: Geometry, dt: float):
        self.g, self.dt = g, dt
        self.reset()

    def reset(self) -> None:
        self.unpressed = FLOAT_AFTER  # a strip just dropped in has not been pressed: it starts curled
        self.floating = True  # curled
        self.curl_set = False  # curled while on the cold block: the bottom sets round the curl
        self.drift_done = False
        self.cold_time = 0.0
        self.set_ = False
        self.flat = self.centred = False
        self.rel = None  # the strip's pose in the mould's frame, held after the set
        self.float_xy = None
        self.failed = False
        self.why = ""

    # -- geometry
    def in_mould(self, snap: Snapshot, p) -> np.ndarray:
        return (np.atleast_2d(p) - snap.mould_pos) @ snap.mould_rot

    def pressed(self, snap: Snapshot) -> bool:
        sx, sy, sz = self.g.tissue
        rel = snap.tissue_rot.T @ (snap.forceps_pos - snap.tissue_pos)
        on_top = sz / 2 - PRESS_INTO <= rel[2] <= sz / 2 + PRESS_ABOVE
        middle_or_curl = -sy / 6 <= rel[1] <= sy / 2  # the middle third or the curling (+y) end, not the end that lies flat anyway
        return bool(abs(rel[0]) <= sx / 2 and middle_or_curl and on_top)

    def over_cold(self, snap: Snapshot) -> bool:
        """Any part of the mould's floor over the cold block, the mould resting: the bottom starts to cool, and a curl then sets."""
        x0, x1, y0, y1 = self.g.cold
        mx, my, mz = snap.mould_pos
        hx, hy = self.g.mould_in[0] / 2, self.g.mould_in[1] / 2
        return mx + hx > x0 and mx - hx < x1 and my + hy > y0 and my - hy < y1 and abs(mz - self.g.plate_z) < 0.004

    def on_cold(self, snap: Snapshot) -> bool:
        x0, x1, y0, y1 = self.g.cold
        mx, my, mz = snap.mould_pos
        hx, hy = self.g.mould_in[0] / 2, self.g.mould_in[1] / 2
        return x0 + hx <= mx <= x1 - hx and y0 + hy <= my <= y1 - hy and abs(mz - self.g.plate_z) < 0.004

    def piece_in_mould(self, snap: Snapshot) -> tuple[np.ndarray, np.ndarray]:
        return self.in_mould(snap, snap.tissue_pos)[0], snap.mould_rot.T @ snap.tissue_rot

    def float_pose(self, snap: Snapshot, xy) -> tuple[np.ndarray, np.ndarray]:
        sz = self.g.tissue[2]
        rel_rot = snap.mould_rot.T @ snap.tissue_rot
        yaw = math.atan2(rel_rot[1, 0], rel_rot[0, 0])
        local = np.array([xy[0], xy[1], self.g.floor + sz / 2 + FLOAT_LIFT + 0.5 * self.g.tissue[1] * math.sin(math.radians(FLOAT_TILT))])
        return snap.mould_pos + snap.mould_rot @ local, snap.mould_rot @ rot_z(yaw) @ rot_x(FLOAT_TILT)

    def fail(self, why: str) -> None:
        if not self.failed:
            self.failed, self.why = True, why

    # -- every control step
    def step(self, snap: Snapshot):
        """Advance the rules one control step. Returns the pose (position, rotation) at which the piece must be held, or None to leave it to physics."""
        g = self.g
        local = self.in_mould(snap, snap.tips)
        ix, iy = g.mould_in
        inside = (np.abs(local[:, 0]) < ix / 2) & (np.abs(local[:, 1]) < iy / 2) & (local[:, 2] > g.floor) & (local[:, 2] < g.mould_h)
        if np.any(inside):
            self.fail("a gripper touched the wax")
        if tilt_deg(snap.mould_rot) > TILT_SPILL or snap.mould_pos[2] > g.plate_z + LIFT_SPILL:
            self.fail("the mould tipped or was lifted")
        tp, _ = self.piece_in_mould(snap)
        if abs(tp[0]) > ix / 2 or abs(tp[1]) > iy / 2 or tp[2] < -0.002:
            self.fail("the piece left the mould")
        if self.set_:
            return snap.mould_pos + snap.mould_rot @ self.rel[0], snap.mould_rot @ self.rel[1]
        on_cold = self.on_cold(snap)
        if self.pressed(snap) and not self.curl_set:
            self.unpressed, self.floating, self.float_xy, self.drift_done = 0.0, False, None, False
        else:
            self.unpressed += self.dt
            if self.unpressed >= FLOAT_AFTER:
                self.floating = True
        if self.floating and self.over_cold(snap):
            self.curl_set = True
        if self.floating:
            if self.float_xy is None or not self.drift_done:
                self.float_xy = tp[:2].copy()  # a curled strip can still be pushed about
            if not self.drift_done and float(np.linalg.norm(snap.mould_vel[:2])) > DRIFT_SPEED:
                v = snap.mould_rot.T @ snap.mould_vel
                self.float_xy = self.float_xy - DRIFT_LAG * v[:2] / max(float(np.linalg.norm(v[:2])), 1e-9)
                self.drift_done = True
        self.cold_time = self.cold_time + self.dt if on_cold else 0.0
        hold = self.float_pose(snap, self.float_xy) if self.floating else None
        if self.cold_time >= SET_AFTER:
            pos, rot = hold if hold is not None else (snap.tissue_pos, snap.tissue_rot)
            rel_pos, rel_rot = snap.mould_rot.T @ (pos - snap.mould_pos), snap.mould_rot.T @ rot
            self.rel = (rel_pos, rel_rot)
            depth = self.ink_heights(rel_pos, rel_rot) - g.floor
            self.flat = float(depth.max() - depth.min()) <= FLAT_DEPTH and float(depth.min()) <= FLAT_DEPTH
            self.centred = math.hypot(rel_pos[0], rel_pos[1]) <= CENTRE
            self.set_ = True
            return snap.mould_pos + snap.mould_rot @ rel_pos, snap.mould_rot @ rel_rot
        return hold

    @property
    def drifted(self) -> bool:
        return self.drift_done

    def ink_heights(self, pos: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Heights in the mould frame of the inked face's four corners, for the strip at `pos`, `r` in that frame."""
        sx, sy, sz = self.g.tissue
        corners = np.array([[a * sx / 2, b * sy / 2, -sz / 2] for a in (-1, 1) for b in (-1, 1)])
        return pos[2] + (corners @ r.T)[:, 2]

    def forceps_in_well(self, snap: Snapshot) -> bool:
        wx, wy = self.g.well
        p = snap.forceps_pos
        return math.hypot(p[0] - wx, p[1] - wy) < 0.008 and p[2] < self.g.plate_z + self.g.well_h

    def hands_clear(self, snap: Snapshot) -> bool:
        local = self.in_mould(snap, snap.tips)
        ix, iy = self.g.mould_in[0] / 2 + 0.001, self.g.mould_in[1] / 2 + 0.001
        dx = np.maximum(np.abs(local[:, 0]) - ix, 0.0)
        dy = np.maximum(np.abs(local[:, 1]) - iy, 0.0)
        dz = np.maximum(local[:, 2] - self.g.mould_h, 0.0)
        return bool(np.min(np.sqrt(dx * dx + dy * dy + dz * dz)) >= CLEAR)

    def moment(self, snap: Snapshot) -> float:
        return score(self.set_, self.flat, self.centred, self.forceps_in_well(snap) and self.hands_clear(snap), self.failed)

    def report(self, snap: Snapshot) -> dict:
        tp, tr = self.piece_in_mould(snap)
        return {"pressed": self.pressed(snap), "curled": self.floating, "curl set": self.curl_set, "drifted": self.drifted, "cold s": round(self.cold_time, 2), "set": self.set_,
                "flat": self.flat, "centred": self.centred, "piece mm": np.round(tp * 1000, 1).tolist(), "piece tilt": round(tilt_deg(tr), 1),
                "forceps in well": self.forceps_in_well(snap), "clear": self.hands_clear(snap), "failed": self.why or False}


def score(set_: bool, flat: bool, centred: bool, clean: bool, failed: bool) -> float:
    """The ladder (task.json, scoring): nothing counts unless the strip set flat."""
    if failed or not set_ or not flat:
        return 0.0
    return 0.5 + (0.25 if centred else 0.0) + (0.25 if clean else 0.0)


def quat_xyzw(r: np.ndarray) -> np.ndarray:
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def write_pose(environment, name: str, pos, rot) -> None:
    """Put a free part's root at a pose (station frame, z from the floor), at rest: the technique of the accepted tasks bayonet-mate (partkit
    `write_pose`) and casement-stay-last-hole (mechkit `_write_pose`)."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0]
    q = quat_xyzw(np.asarray(rot, float))
    pose = torch.tensor([[*(float(v) for v in pos), *(float(v) for v in q)]], dtype=origin.dtype, device=origin.device)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=origin.dtype, device=origin.device))
