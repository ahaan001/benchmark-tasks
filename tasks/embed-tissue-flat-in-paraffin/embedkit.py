"""The embedding rules as bookkeeping, pure Python and NumPy: how the strip lies, when the wax sets, and the score. `write_pose` (holding the set
strip in its block) is the one call into Isaac Lab, imported lazily.

The bow is physics, not a rule. The strip is two rigid halves on a sprung hinge across its middle (task.py): left alone it stands bowed on its two
ends, and it lies flat only while something pushes its middle down against the spring. The rules read how it lies; they never flatten it.
When `bow_set` or `set_` turns true, task.py locks the hinge where it is (`mechkit.lock`): the wax has set round the strip.

Frames: the station frame with z up from the table. The mould's frame has its origin at the centre of its floor's underside; the strip's at its
centre when flat, its inked cut face towards -z, the hinge along x through the middle of its top face; the forceps' at their closed tips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SET_AFTER = 8.0  # s on the cold block for the bottom layer to set
BOW_SETS_AFTER = 0.3  # s standing bowed with the mould over the cold block before the bottom sets round the bow
FLAT_DEPTH = 0.001  # m: the whole inked face within the first 1 mm of facing (spread of its depth, and its lowest point off the floor)
CENTRE = 0.003  # m from the mould's middle
CLEAR = 0.03  # m from every fingertip point to the mould
TILT_SPILL, LIFT_SPILL = 10.0, 0.01  # degrees and m: the mould tipped or lifted


@dataclass
class Geometry:
    plate_z: float
    cold: tuple  # x0, x1, y0, y1
    well: tuple  # x, y
    well_in: float  # inside, square
    well_h: float
    mould_in: tuple  # inside x, y
    mould_h: float
    floor: float
    tissue: tuple  # x, y, z
    fold: float  # rad: the hinge's travel from the rest bow (0) to flat
    start_mould: tuple


@dataclass
class Snapshot:
    mould_pos: np.ndarray
    mould_rot: np.ndarray
    mould_vel: np.ndarray
    tissue_pos: np.ndarray  # the strip's near half: the hinge is on its top face
    tissue_rot: np.ndarray
    fold: float  # rad: the hinge now, read from the simulator: 0 at the rest bow, `Geometry.fold` flat
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


def ink_corners(tissue: tuple, fold: float, q: float) -> np.ndarray:
    """The inked face's corners in the strip's frame, four a half, with the hinge at `q`."""
    sx, sy, sz = tissue
    hinge = np.array([0.0, 0.0, sz / 2])
    near = np.array([[a * sx / 2, y, -sz / 2] for a in (-1, 1) for y in (-sy / 2, 0.0)])
    far = np.array([[a * sx / 2, y, -sz / 2] for a in (-1, 1) for y in (0.0, sy / 2)])
    return np.vstack([near, hinge + (far - hinge) @ rot_x(math.degrees(q - fold)).T])


def crown_height(tissue: tuple, tilt: float) -> float:
    """How high the hinge stands above a level floor with each half tilted `tilt` rad, both ends down."""
    return tissue[1] / 2 * math.sin(tilt) + tissue[2] * math.cos(tilt)


def bow_pose(hinge_xy, floor_z: float, tilt: float, tissue: tuple) -> tuple[np.ndarray, np.ndarray]:
    """The strip's pose (position, rotation) standing on a level floor at `floor_z`, each half tilted `tilt` rad, its hinge over `hinge_xy`."""
    rot = rot_x(math.degrees(tilt))  # the near half's far end down
    hinge = np.array([hinge_xy[0], hinge_xy[1], floor_z + crown_height(tissue, tilt)])
    return hinge - rot @ np.array([0.0, 0.0, tissue[2] / 2]), rot


class Rules:
    def __init__(self, g: Geometry, dt: float):
        self.g, self.dt = g, dt
        self.reset()

    def reset(self) -> None:
        self.bowed_cold = 0.0  # s the strip has stood bowed with the mould over the cold block
        self.bow_set = False  # the bottom has set round the bow: pressing no longer helps
        self.cold_time = 0.0
        self.set_ = False
        self.flat = self.centred = False
        self.rel = None  # the strip's pose in the mould's frame, held after the set
        self.failed = False
        self.why = ""

    # -- geometry
    def in_mould(self, snap: Snapshot, p) -> np.ndarray:
        return (np.atleast_2d(p) - snap.mould_pos) @ snap.mould_rot

    def hinge(self, snap: Snapshot) -> np.ndarray:
        """The middle of the strip's fold line, on its top face: the crown of the bow."""
        return snap.tissue_pos + snap.tissue_rot @ np.array([0.0, 0.0, self.g.tissue[2] / 2])

    def ink_depth(self, snap: Snapshot) -> np.ndarray:
        """How far each corner of the inked face stands off the mould's floor."""
        corners = snap.tissue_pos + ink_corners(self.g.tissue, self.g.fold, snap.fold) @ snap.tissue_rot.T
        return self.in_mould(snap, corners)[:, 2] - self.g.floor

    def lies_flat(self, snap: Snapshot) -> bool:
        depth = self.ink_depth(snap)
        return bool(float(depth.max() - depth.min()) <= FLAT_DEPTH and float(depth.min()) <= FLAT_DEPTH)

    def tip_point(self, snap: Snapshot) -> np.ndarray:
        """The lowest corner of the closed tips' end face (1.5 mm by 2 mm): where a tilted forceps touches."""
        corners = np.array([[a * 0.00075, b * 0.001, 0.0] for a in (-1, 1) for b in (-1, 1)])
        pts = snap.forceps_pos + corners @ snap.forceps_rot.T
        return pts[int(np.argmin(pts[:, 2]))]

    def over_cold(self, snap: Snapshot) -> bool:
        """Any part of the mould's floor over the cold block, the mould resting: the bottom starts to cool, and a bow then sets."""
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
        return self.in_mould(snap, self.hinge(snap))[0], snap.mould_rot.T @ snap.tissue_rot

    def fail(self, why: str) -> None:
        if not self.failed:
            self.failed, self.why = True, why

    # -- every control step
    def step(self, snap: Snapshot):
        """Advance the rules one control step. Returns the pose (position, rotation) at which the set strip is held in its block, or None."""
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
        flat_now = self.lies_flat(snap)
        self.bowed_cold = self.bowed_cold + self.dt if not flat_now and self.over_cold(snap) else 0.0
        if self.bowed_cold >= BOW_SETS_AFTER:
            self.bow_set = True
        self.cold_time = self.cold_time + self.dt if self.on_cold(snap) else 0.0
        if self.cold_time >= SET_AFTER:
            rel_pos, rel_rot = snap.mould_rot.T @ (snap.tissue_pos - snap.mould_pos), snap.mould_rot.T @ snap.tissue_rot
            self.rel = (rel_pos, rel_rot)
            self.flat = flat_now and not self.bow_set  # a bowed strip sets bowed
            self.centred = math.hypot(tp[0], tp[1]) <= CENTRE
            self.set_ = True
            return snap.mould_pos + snap.mould_rot @ rel_pos, snap.mould_rot @ rel_rot
        return None

    def forceps_in_well(self, snap: Snapshot) -> bool:
        """The closed tips inside the well's square tube, below its top."""
        wx, wy = self.g.well
        p = snap.forceps_pos
        return abs(p[0] - wx) < self.g.well_in / 2 and abs(p[1] - wy) < self.g.well_in / 2 and p[2] < self.g.plate_z + self.g.well_h

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
        depth = self.ink_depth(snap)
        return {"tip from crown mm": np.round((self.tip_point(snap) - self.hinge(snap)) * 1000, 1).tolist(), "mould": np.round(snap.mould_pos * 1000, 1).tolist(),
                "mould yaw": round(math.degrees(math.atan2(snap.mould_rot[1, 0], snap.mould_rot[0, 0])), 1), "fold deg": round(math.degrees(snap.fold), 1), "ink off floor mm": [round(float(depth.min()) * 1000, 2), round(float(depth.max()) * 1000, 2)],
                "lies flat": self.lies_flat(snap), "bow set": self.bow_set, "cold s": round(self.cold_time, 2), "set": self.set_, "flat": self.flat, "centred": self.centred,
                "crown mm": np.round(tp * 1000, 1).tolist(), "near half tilt": round(tilt_deg(tr), 1), "forceps in well": self.forceps_in_well(snap), "clear": self.hands_clear(snap),
                "failed": self.why or False}


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


def quat_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def write_pose(environment, name: str, pos, rot, body: str | None = None) -> None:
    """Put a free part at a pose (station frame, z from the floor), at rest: the technique of the accepted tasks bayonet-mate (partkit
    `write_pose`) and casement-stay-last-hole (mechkit `_write_pose`). For a jointed part the pose is that of `body`; its joints stay as they are."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0]
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    names = list(getattr(asset, "body_names", ()))
    if body is not None and names and names[0] != body:  # the root link is another body: carry it with `body`
        poses = getattr(asset.data.body_link_pose_w, "torch", asset.data.body_link_pose_w)[0].detach().cpu().numpy().astype(float)
        root, held = poses[0], poses[names.index(body)]
        turn = rot @ quat_matrix(held[3:7]).T
        pos, rot = pos + turn @ (root[:3] - held[:3]), turn @ quat_matrix(root[3:7])
    q = quat_xyzw(rot)
    pose = torch.tensor([[*(float(v) for v in pos), *(float(v) for v in q)]], dtype=origin.dtype, device=origin.device)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=origin.dtype, device=origin.device))
