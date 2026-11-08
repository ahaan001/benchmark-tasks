"""On the station table, in front of the right arm: a light green silicone medication mat, 34 x 26 cm, with a
6 x 6 cm square of black tape on it. A cloudy (milky white) insulin vial, 2.5 cm across and 5.5 cm tall, stands in the
middle of the square, 30 cm in front of the arm bases and 16 cm to the right of their midpoint. An identical clear
vial stands 10 cm to its right. A blue box of pen needles and a dark grey glucose meter lie on the mat behind them,
and a red sharps container, 10 x 10 cm inside with 8 cm walls, is fixed to the table near the right arm. The left
arm has nothing to do.

The vial must be mixed by rolling: turned about its own axis, standing on the mat, ten full turns in all. The wrist
cannot turn that far in one go, so the robot grasps, turns, lets go and grasps again.

Score (task_score), from every step of the trial and the final state:
- 0 if the cloudy vial was ever lifted more than 2 cm off the mat, tilted more than 15 degrees, or taken out of
  the taped square (centre more than 3 cm from the square's centre along x or y), or if the clear vial was ever
  moved more than 1 cm or turned more than 20 degrees. These stay for the rest of the trial.
- 0 if at the end the cloudy vial is still held (any part of either gripper within 2 cm of it).
- Otherwise 0.1 for each full turn (360 degrees) of the cloudy vial about its axis, in either direction, summed
  over the trial while its base was within 5 mm of the mat, up to 1.0 at ten turns.

Run it:  inspect-robots run --embodiment yam_arena -E task=insulin_vial_roll.py -E object_jitter=0 \\
           --instruction "Mix the cloudy insulin vial by rolling it ten full turns on the mat. Do not lift or tip \\
it, and leave the clear vial where it is." --scorer task_score ...
"""

import functools
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Cylinder
from yam_arena.urdf import forward_kinematics, rpy_matrix

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # ten turns done, the vial let go and at rest in the square this long
TOP = TABLE_TOP[SCENE]

# ---- layout (m). x forward from the arm bases, y left; the right arm's base is at (0, -0.305) ----
MAT_CENTRE = (0.28, -0.18)
MAT_SIZE = (0.34, 0.26, 0.003)
MAT_TOP = TOP + MAT_SIZE[2]
SQUARE_CENTRE = (0.30, -0.16)  # the taped square, where the cloudy vial stands
SQUARE_INSIDE = 0.06
TAPE = (0.003, 0.0015)  # width, thickness of the tape strips
VIAL_RADIUS, VIAL_HEIGHT, VIAL_MASS = 0.0125, 0.055, 0.025
CLEAR_CENTRE = (0.30, -0.26)  # the clear vial, 10 cm to the right of the cloudy one
NEEDLES_CENTRE, NEEDLES_SIZE = (0.22, -0.08), (0.06, 0.04, 0.03)
METER_CENTRE, METER_SIZE = (0.36, -0.08), (0.08, 0.05, 0.015)
SHARPS_CENTRE = (0.20, -0.40)
SHARPS_INSIDE, SHARPS_WALL, SHARPS_HEIGHT = 0.10, 0.004, 0.08

# ---- scoring ----
TURNS_FULL = 10  # full turns for the whole score
PER_TURN = 0.1
LIFT_ZERO = 0.020  # the vial's base this far off the mat at any time: score 0
LIFT_NOCOUNT = 0.005  # rotation while the base is this far off the mat does not count
TILT_DEG = 15.0  # the vial this far from upright at any time: score 0
SQUARE_TOL = SQUARE_INSIDE / 2  # the vial's centre this far from the square's centre, along x or y: outside it
DISTRACTOR_MOVE = 0.010  # the clear vial's centre moved this far at any time: score 0
DISTRACTOR_TURN_DEG = 20.0  # or turned this far
RELEASED = 0.020  # the vial is let go when every part of both grippers is this far from it
MESH_SAMPLE = 0.002  # spacing of the points checked on the robot's collision meshes
NEAR = 0.05  # gripper links farther than this from the vial are not measured

GREEN = (0.68, 0.80, 0.68)
BLACK = (0.05, 0.05, 0.05)
MILKY = (0.93, 0.93, 0.88)
CLEAR = (0.62, 0.78, 0.92)
BLUE = (0.25, 0.45, 0.85)
DARK = (0.20, 0.20, 0.22)
RED = (0.80, 0.10, 0.10)


def _box(name, xy, size, under, color, mass=None) -> Box:
    """A box with its centre over xy and its underside at height `under`."""
    return Box(name, pos=(*xy, under + size[2] / 2), size=size, color=color, mass=mass)


def _square() -> list:
    (x, y), half, (w, h) = SQUARE_CENTRE, SQUARE_INSIDE / 2 + TAPE[0] / 2, TAPE
    long = SQUARE_INSIDE + 2 * TAPE[0]
    return [
        _box("tape_near", (x - half, y), (w, long, h), MAT_TOP, BLACK),
        _box("tape_far", (x + half, y), (w, long, h), MAT_TOP, BLACK),
        _box("tape_right", (x, y - half), (long, w, h), MAT_TOP, BLACK),
        _box("tape_left", (x, y + half), (long, w, h), MAT_TOP, BLACK),
    ]


def _sharps() -> list:
    (x, y), s, w, h = SHARPS_CENTRE, SHARPS_INSIDE, SHARPS_WALL, SHARPS_HEIGHT
    ends = [(f"sharps_wall_{n}", (x + k * (s + w) / 2, y), (w, s + 2 * w)) for n, k in (("near", -1), ("far", 1))]
    sides = [(f"sharps_wall_{n}", (x, y + k * (s + w) / 2), (s, w)) for n, k in (("right", -1), ("left", 1))]
    walls = [_box(n, xy, (*size, h), TOP, RED) for n, xy, size in ends + sides]
    return [_box("sharps_floor", (x, y), (s, s, w), TOP, RED), *walls]


def _vial(name, xy, color) -> Cylinder:
    return Cylinder(
        name, pos=(*xy, MAT_TOP + VIAL_HEIGHT / 2), radius=VIAL_RADIUS, height=VIAL_HEIGHT, mass=VIAL_MASS, color=color
    )


FIXED = [_box("mat", MAT_CENTRE, MAT_SIZE, TOP, GREEN), *_square(), *_sharps()]
CLOUDY = _vial("cloudy_vial", SQUARE_CENTRE, MILKY)
CLEAR_VIAL = _vial("clear_vial", CLEAR_CENTRE, CLEAR)
DRESSING = [
    _box("pen_needles", NEEDLES_CENTRE, NEEDLES_SIZE, MAT_TOP, BLUE, mass=0.03),
    _box("meter", METER_CENTRE, METER_SIZE, MAT_TOP, DARK, mass=0.06),
]
OBJECTS = [*FIXED, CLOUDY, CLEAR_VIAL, *DRESSING]


# ---- geometry ----
def _rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def wrap(angle) -> float:
    """The angle brought into (-pi, pi]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def yaw_of(state) -> float:
    """The vial's turn about the vertical (rad): where its own x axis points, seen from above."""
    r = _rotation(state.quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def tilt_deg(state) -> float:
    """The vial's axis from upright (degrees)."""
    return math.degrees(math.acos(min(max(_rotation(state.quat_xyzw)[2, 2], -1.0), 1.0)))


def base_rise(state) -> float:
    """How far the vial's lowest point is above the mat (m)."""
    return float(state.low[2] - MAT_TOP)


def in_square(state) -> bool:
    return bool(np.all(np.abs(state.pos[:2] - SQUARE_CENTRE) <= SQUARE_TOL))


def cylinder_gap(points, state) -> float:
    """Smallest distance (m) from world points to the vial's surface; 0 for a point inside it, inf for none."""
    if not len(points):
        return math.inf
    local = (points - state.pos) @ _rotation(state.quat_xyzw)
    radial = np.maximum(np.hypot(local[:, 0], local[:, 1]) - VIAL_RADIUS, 0.0)
    axial = np.maximum(np.abs(local[:, 2]) - VIAL_HEIGHT / 2, 0.0)
    return float(np.hypot(radial, axial).min())


# ---- the robot, as the simulator has it. The hooks are handed object states only, so the episode clock and the
# arms' joints are read (never changed) from the yam_arena embodiment that calls them ----
def _embodiment():
    frame = sys._getframe(1)
    while frame is not None:
        owner = frame.f_locals.get("self")
        if type(owner).__name__ == "YamArenaEmbodiment":
            return owner
        frame = frame.f_back
    raise RuntimeError("insulin_vial_roll: no YamArenaEmbodiment on the call stack")


def _sim_time() -> float:
    """Seconds since the episode's reset. It goes back to 0 at each reset."""
    return float(_embodiment()._t)


def _arm_joints() -> dict:
    """Per arm, its measured joints: [joint1..joint6 (rad), finger opening (m, FINGER_OPEN .. 0)]."""
    owner = _embodiment()
    out = {}
    for name, (arm_ids, finger_ids) in owner._ids.items():
        q = owner._base.scene[name].data.joint_pos.torch[0]
        out[name.removesuffix("_robot")] = np.r_[q[arm_ids].cpu().numpy(), q[finger_ids].mean().item()].astype(float)
    return out


clock, arm_joints = _sim_time, _arm_joints  # the offline tests replace them


def _triangles(stl: Path) -> np.ndarray:
    data = stl.read_bytes()  # binary STL: 80-byte header, count, then 50 bytes per triangle
    count = struct.unpack("<I", data[80:84])[0]
    record = np.dtype([("normal", "<3f4"), ("corners", "<9f4"), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=record, count=count, offset=84)["corners"].reshape(-1, 3, 3).astype(float)


def _sampled(triangles) -> np.ndarray:
    """Points no more than MESH_SAMPLE apart over the triangles."""
    splits = np.ceil(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max(1) / MESH_SAMPLE)
    splits = np.maximum(splits, 1).astype(int)
    out = []
    for n in np.unique(splits):
        weights = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]) / n
        out.append(np.einsum("wc,tcx->twx", weights, triangles[splits == n]).reshape(-1, 3))
    return np.vstack(out)


@functools.cache
def link_surfaces() -> dict:
    """Per moving link: points over its collision mesh (the URDF's) in the link's frame, and the centre and
    radius of a ball that holds them."""
    out = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        clouds = []
        for collision in link.findall("collision"):
            mesh, origin = collision.find("geometry/mesh"), collision.find("origin")
            if mesh is None:
                continue
            xyz, rpy = (np.array(origin.get(k).split(), dtype=float) for k in ("xyz", "rpy"))
            triangles = _triangles(Path(YAM_URDF).parent / mesh.get("filename"))
            clouds.append(_sampled(triangles) @ rpy_matrix(rpy).T + xyz)
        if clouds and link.get("name") != "base":
            points = np.unique(np.round(np.vstack(clouds) / (MESH_SAMPLE / 2)), axis=0) * MESH_SAMPLE / 2
            centre = (points.min(0) + points.max(0)) / 2
            out[link.get("name")] = (points, centre, float(np.linalg.norm(points - centre, axis=1).max()))
    return out


GRIPPER_LINKS = ("gripper", "tip_left", "tip_right")


def arm_points(joints, side, links=None, near=None) -> np.ndarray:
    """World points over one arm's links (all moving links, or the ones named) for its measured joints. With
    near = (point, radius), links that lie wholly outside that ball are left out."""
    q = dict(zip(ARM_JOINTS, joints[:6], strict=True))
    q["joint7"] = q["joint8"] = joints[6]
    poses = forward_kinematics(YAM_URDF, q)
    base = np.asarray(station.BASES[side])
    clouds = []
    for name, (points, centre, radius) in link_surfaces().items():
        if links and name not in links:
            continue
        rotation, origin = poses[name][:3, :3], poses[name][:3, 3] + base
        if near is not None and np.linalg.norm(rotation @ centre + origin - near[0]) > radius + near[1]:
            continue
        clouds.append(points @ rotation.T + origin)
    return np.vstack(clouds) if clouds else np.empty((0, 3))


def gripper_gap(state, arms) -> float:
    """Smallest distance (m) from either gripper (body and fingers) to the vial; inf when both are far."""
    ball = (state.pos, VIAL_HEIGHT / 2 + NEAR)
    hands = np.vstack([arm_points(joints, side, GRIPPER_LINKS, ball) for side, joints in arms.items()])
    return cylinder_gap(hands, state)


# ---- what the final state does not show ----
class Tracker:
    """Reads both vials and the arms after each control step and keeps, until the episode's next reset: the
    cloudy vial's rotation about its axis summed over the trial, and whether a rule was ever broken (the cloudy
    vial lifted, tilted or taken out of the square; the clear vial moved or turned). A reset shows as the episode
    clock going back; nothing the robot does to the objects clears a broken rule."""

    def __init__(self):
        self.restart()
        self._t = None

    def restart(self):
        self.total_rot = 0.0  # radians, both directions summed
        self.prev_yaw = None
        self.spoiled = False
        self.held = False  # at the last step, a gripper was within RELEASED of the cloudy vial
        self.rise = 0.0  # at the last step, the cloudy vial's base above the mat
        self.clear_yaw = None  # the clear vial's turn when the episode started

    def update(self, objects):
        """Read one step. success() and score() both ask in the same step, so a step is read once."""
        t = clock()
        if t == self._t:
            return
        if self._t is None or t < self._t:
            self.restart()
        self._t = t
        vial, clear = objects["cloudy_vial"], objects["clear_vial"]
        yaw = yaw_of(vial)
        self.rise = base_rise(vial)
        if self.prev_yaw is not None and self.rise <= LIFT_NOCOUNT:
            self.total_rot += abs(wrap(yaw - self.prev_yaw))
        self.prev_yaw = yaw
        if self.clear_yaw is None:
            self.clear_yaw = yaw_of(clear)
        self.spoiled |= bool(
            self.rise > LIFT_ZERO
            or tilt_deg(vial) > TILT_DEG
            or not in_square(vial)
            or np.linalg.norm(clear.pos - CLEAR_VIAL.pos) > DISTRACTOR_MOVE
            or abs(math.degrees(wrap(yaw_of(clear) - self.clear_yaw))) > DISTRACTOR_TURN_DEG
        )
        self.held = gripper_gap(vial, arm_joints()) < RELEASED

    @property
    def turns(self) -> int:
        return int(self.total_rot // (2 * math.pi))


TRACKER = Tracker()


def _placed(objects) -> bool:
    """The cloudy vial let go, upright and inside the square, with no rule broken."""
    vial = objects["cloudy_vial"]
    return not TRACKER.spoiled and not TRACKER.held and in_square(vial) and tilt_deg(vial) <= TILT_DEG


# ---- hooks read by yam_arena (task_score) ----
def success(objects):
    """Score 1.0: ten full turns, the vial let go, upright, at rest inside the square, no rule broken."""
    TRACKER.update(objects)
    return _placed(objects) and TRACKER.turns >= TURNS_FULL and objects["cloudy_vial"].resting()


def score(objects):
    """0 once a rule was broken, or while the vial is held or out of the square. Otherwise 0.1 per full turn."""
    TRACKER.update(objects)
    if not _placed(objects):
        return 0.0
    return round(min(1.0, PER_TURN * TRACKER.turns), 4)


# ---- demo ----
YAW_LO, YAW_HI = math.radians(-80.0), math.radians(130.0)  # the grip's turn at the start and end of a twist
SWING = YAW_HI - YAW_LO  # 210 degrees per grasp (check.py measures the joint margins at both ends)
SLACK = 1.10  # the demo turns this much more than ten turns, for slip between the pads and the vial
CYCLES = math.ceil(TURNS_FULL * 2 * math.pi * SLACK / SWING)
GRIP_IN = 0.0025  # finger target on the vial: each pad this far into it
APPROACH = 0.060  # the fingers come down and go back up this far apart
HOVER = TOP + 0.12  # TCP height on the way to the vial and away from it
GRASP_Z = MAT_TOP + VIAL_HEIGHT / 2 + PAD_TIP_PAST_TCP  # pad tips at the vial's mid-height
RETURN_Z = GRASP_Z + 0.04  # the fingers go back for the next grasp with their tips above the vial
LIFT_SPOIL = 0.03  # variant="lift": the vial is raised this much during each twist


def demo(variant="scripted"):
    """Right arm only. Come down on the cloudy vial with the grip turned to -80 degrees, pinch it, turn the grip
    to +130 degrees (210 degrees), open, rise so the fingertips clear the vial, turn back, come down and pinch
    again: 19 times, 3,990 degrees in all, then rise and move away. Every turn is split in two, so no waypoint
    turns the grip more than 105 degrees (the plan interpolates rotations the short way round).

    variant="lift" is for the offline check only: the same plan, but the vial is raised 3 cm during each twist,
    which breaks the lift rule at the first twist."""
    x, y = SQUARE_CENTRE
    wide, grip = -APPROACH / 2, -(VIAL_RADIUS - GRIP_IN)
    mid = (YAW_LO + YAW_HI) / 2
    twist_z = GRASP_Z + LIFT_SPOIL if variant == "lift" else GRASP_Z
    plan = [
        Waypoint(2.0, (x, y, HOVER), wide, "to the vial", YAW_LO),
        Waypoint(1.5, (x, y, GRASP_Z), wide, "descend", YAW_LO),
        Waypoint(0.4, (x, y, GRASP_Z), wide, "settle", YAW_LO),
    ]
    for n in range(1, CYCLES + 1):
        plan += [
            Waypoint(0.4, (x, y, GRASP_Z), grip, f"{n}: pinch", YAW_LO),
            Waypoint(0.2, (x, y, twist_z), grip, f"{n}: hold", YAW_LO),
            Waypoint(0.75, (x, y, twist_z), grip, f"{n}: twist", mid),
            Waypoint(0.75, (x, y, twist_z), grip, f"{n}: twist", YAW_HI),
            Waypoint(0.3, (x, y, twist_z), wide, f"{n}: let go", YAW_HI),
            Waypoint(0.4, (x, y, RETURN_Z), wide, f"{n}: rise", YAW_HI),
            Waypoint(0.5, (x, y, RETURN_Z), wide, f"{n}: turn back", mid),
            Waypoint(0.5, (x, y, RETURN_Z), wide, f"{n}: turn back", YAW_LO),
            Waypoint(0.4, (x, y, GRASP_Z), wide, f"{n}: descend", YAW_LO),
            Waypoint(0.2, (x, y, GRASP_Z), wide, f"{n}: settle", YAW_LO),
        ]
    plan += [
        Waypoint(1.0, (x, y, HOVER), wide, "rise", YAW_LO),
        Waypoint(1.0, (x - 0.10, y, HOVER), wide, "away", YAW_LO),
    ]
    return {"right": plan}
