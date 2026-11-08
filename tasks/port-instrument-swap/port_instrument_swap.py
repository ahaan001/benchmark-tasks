"""On the station table, in front of the right arm: a draped silicone abdomen with two ports. In the right-hand port,
a 2 cm square hole 3 cm deep between two loose red flanges (3 x 1.2 x 2 cm, 15 g), stands a grey instrument: a
1.2 cm square rod, 9 cm long, seated on the hole's floor and reaching 6 cm above the skin. The other port holds a
fixed scope. A blue instrument of the same size stands in a steel stand beside the abdomen, and a steel tray lies
on the other side. The hole takes one rod at a time, so the grey rod must be fully out before the blue one can seat,
and a rod lowered or raised more than 2 mm off the hole's axis touches a flange.

Score (task_score), from every step of the trial and the final state:
- 0 if a flange was ever pushed more than 1 cm or tilted more than 15 degrees, or touched three times. A touch is
  the robot or either rod within 2 mm of a flange, or a flange moved; it ends once the flanges are still and
  everything is 3.5 mm clear. All of this stays for the rest of the trial.
- Otherwise 0.3 if the grey rod rests inside the tray and is let go, plus 0.7 if the blue rod stands in the port on
  its floor (within 5 mm, upright within 5 degrees) and is let go, or 0.35 if it is in the port but short of the
  floor. The sum is halved once per touch.

Run it:  inspect-robots run --embodiment yam_arena -E task=port_instrument_swap.py -E object_jitter=0 \\
           --instruction "Take the grey instrument out of the port between the two red flanges and lay it in the \\
steel tray, then put the blue instrument from the stand down the same port until it stops, without disturbing \\
the port." --scorer task_score ...
"""

import functools
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box
from yam_arena.urdf import forward_kinematics, rpy_matrix

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # the rods must stay at rest, let go, this long
TOP = TABLE_TOP[SCENE]

# ---- layout (m). x forward from the arm bases, y left; the right arm's base is at (0, -0.305) ----
DRAPE_SIZE, DRAPE_XY = (0.46, 0.43, 0.002), (0.26, -0.175)
DRAPE_TOP = TOP + DRAPE_SIZE[2]
SLAB_SIZE, SLAB_XY = (0.30, 0.22, 0.035), (0.26, -0.20)  # the abdomen, on the drape
SKIN_TOP = DRAPE_TOP + SLAB_SIZE[2]
PORT = SLAB_XY  # the hole's axis
WELL_W = 0.020  # the hole, square
WELL_DEPTH = 0.030
WELL_FLOOR = SKIN_TOP - WELL_DEPTH
LINER_T = 0.003  # the hole's dark walls, inside a cut-out in the slab
CUT = WELL_W + 2 * LINER_T
POST_SIZE = (0.030, 0.012, 0.020)  # the flanges: long along x, standing on the skin either side of the hole
POST_Y = {"post_a": PORT[1] - WELL_W / 2 - POST_SIZE[1] / 2, "post_b": PORT[1] + WELL_W / 2 + POST_SIZE[1] / 2}
ROD_SIZE = (0.012, 0.012, 0.090)
ROD_SIDE, ROD_LEN = ROD_SIZE[0], ROD_SIZE[2]
STAND_XY, STAND_OUTER, STAND_WALL = (0.14, -0.36), 0.05, 0.015  # the instrument stand: one socket, same depth as the hole
STAND_FLOOR = SKIN_TOP - WELL_DEPTH
TRAY_XY, TRAY_INNER, TRAY_WALL, TRAY_H = (0.14, -0.02), (0.16, 0.10), 0.003, 0.03
TRAY_FLOOR = DRAPE_TOP + TRAY_WALL
SCOPE_XY, SCOPE_LEN = (0.36, -0.12), 0.10

# ---- scoring ----
GREY_VALUE, BLUE_VALUE, BLUE_SHORT = 0.3, 0.7, 0.5  # the blue rod in the port but short of the floor earns half
TOUCH = 0.002  # a touch starts when a rod or the robot comes this close to a flange
TOUCH_CLEAR = 0.0035  # and ends once everything is this far from both flanges again and they are still (a rod seated on the hole's axis is 4 mm from them)
TOUCH_FACTOR = 0.5  # each touch halves the score
MAX_TOUCHES = 3  # the third touch scores 0
POST_MOVED, POST_TILTED_DEG = 0.003, 3.0  # a flange this far from its mark or this far from upright was touched
POST_PUSHED, POST_PUSHED_DEG = 0.010, 15.0  # a flange pushed this far: score 0
POST_END, POST_END_DEG = 0.003, 5.0  # success needs both flanges this close to their marks
SEATED = 0.005  # the blue rod's base within this of the hole's floor
UPRIGHT_DEG = 5.0
RELEASED = 0.02  # a rod is let go when every part of both grippers is this far from it
ROD_SAMPLE = 0.0025  # spacing of the points checked on a rod's surface
MESH_SAMPLE = 0.004  # spacing of the points checked on the robot's collision meshes
NEAR = 0.05  # distances to a flange beyond this are not measured


def _box(name, xy, size, under, color, mass=None) -> Box:
    return Box(name, pos=(*xy, under + size[2] / 2), size=size, mass=mass, color=color)


SKIN, DRAPE_BLUE, RED = (0.87, 0.68, 0.58), (0.25, 0.40, 0.70), (0.85, 0.12, 0.12)
GREY, BLUE, STEEL, DARK = (0.45, 0.45, 0.48), (0.20, 0.35, 0.85), (0.70, 0.72, 0.74), (0.15, 0.15, 0.17)


def _ring(prefix, xy, inner, wall, under, height, color) -> list:
    """Four fixed walls round a square opening `inner` wide."""
    x, y = xy
    off = (inner + wall) / 2
    return [
        _box(f"{prefix}_wall_a", (x, y - off), (inner + 2 * wall, wall, height), under, color),
        _box(f"{prefix}_wall_b", (x, y + off), (inner + 2 * wall, wall, height), under, color),
        _box(f"{prefix}_wall_c", (x - off, y), (wall, inner, height), under, color),
        _box(f"{prefix}_wall_d", (x + off, y), (wall, inner, height), under, color),
    ]


def _slab() -> list:
    """The abdomen in four fixed pieces round the hole's cut-out, the hole's floor and its dark walls."""
    (x, y), (sx, sy, sz) = SLAB_XY, SLAB_SIZE
    side = (sy - CUT) / 2
    return [
        _box("slab_a", (x, y - CUT / 2 - side / 2), (sx, side, sz), DRAPE_TOP, SKIN),
        _box("slab_b", (x, y + CUT / 2 + side / 2), (sx, side, sz), DRAPE_TOP, SKIN),
        _box("slab_c", (x - CUT / 2 - (sx - CUT) / 4, y), ((sx - CUT) / 2, CUT, sz), DRAPE_TOP, SKIN),
        _box("slab_d", (x + CUT / 2 + (sx - CUT) / 4, y), ((sx - CUT) / 2, CUT, sz), DRAPE_TOP, SKIN),
        _box("well_floor", PORT, (CUT, CUT, WELL_FLOOR - DRAPE_TOP), DRAPE_TOP, DARK),
        *_ring("well", PORT, WELL_W, LINER_T, WELL_FLOOR, WELL_DEPTH, DARK),
    ]


def _stand() -> list:
    wall = (STAND_OUTER - WELL_W) / 2
    return [
        _box("stand_floor", STAND_XY, (STAND_OUTER, STAND_OUTER, STAND_FLOOR - DRAPE_TOP), DRAPE_TOP, STEEL),
        *_ring("stand", STAND_XY, WELL_W, wall, STAND_FLOOR, WELL_DEPTH, STEEL),
    ]


def _tray() -> list:
    return [
        _box("tray_floor", TRAY_XY, (TRAY_INNER[0] + 2 * TRAY_WALL, TRAY_INNER[1] + 2 * TRAY_WALL, TRAY_WALL), DRAPE_TOP, STEEL),
        _box("tray_wall_a", (TRAY_XY[0], TRAY_XY[1] - TRAY_INNER[1] / 2 - TRAY_WALL / 2), (TRAY_INNER[0] + 2 * TRAY_WALL, TRAY_WALL, TRAY_H), TRAY_FLOOR, STEEL),
        _box("tray_wall_b", (TRAY_XY[0], TRAY_XY[1] + TRAY_INNER[1] / 2 + TRAY_WALL / 2), (TRAY_INNER[0] + 2 * TRAY_WALL, TRAY_WALL, TRAY_H), TRAY_FLOOR, STEEL),
        _box("tray_wall_c", (TRAY_XY[0] - TRAY_INNER[0] / 2 - TRAY_WALL / 2, TRAY_XY[1]), (TRAY_WALL, TRAY_INNER[1], TRAY_H), TRAY_FLOOR, STEEL),
        _box("tray_wall_d", (TRAY_XY[0] + TRAY_INNER[0] / 2 + TRAY_WALL / 2, TRAY_XY[1]), (TRAY_WALL, TRAY_INNER[1], TRAY_H), TRAY_FLOOR, STEEL),
    ]


FIXED = [
    _box("drape", DRAPE_XY, DRAPE_SIZE, TOP, DRAPE_BLUE),
    *_slab(),
    *_stand(),
    *_tray(),
    *_ring("scope_port", SCOPE_XY, WELL_W, LINER_T, SKIN_TOP, POST_SIZE[2], DARK),
    _box("scope", SCOPE_XY, (ROD_SIDE, ROD_SIDE, SCOPE_LEN), SKIN_TOP, DARK),
]
POSTS = {name: _box(name, (PORT[0], y), POST_SIZE, SKIN_TOP, RED, mass=0.015) for name, y in POST_Y.items()}
RODS = {
    "grey_inst": _box("grey_inst", PORT, ROD_SIZE, WELL_FLOOR, GREY, mass=0.04),
    "blue_inst": _box("blue_inst", STAND_XY, ROD_SIZE, STAND_FLOOR, BLUE, mass=0.04),
}
OBJECTS = [*FIXED, *POSTS.values(), *RODS.values()]


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


def _surface(size, spacing) -> np.ndarray:
    """Points every `spacing` over a box's six faces, in the box's own frame."""
    half = np.array(size) / 2
    lines = [np.linspace(-h, h, round(2 * h / spacing) + 1) for h in half]
    faces = []
    for axis in range(3):
        a, b = (i for i in range(3) if i != axis)
        grid = np.stack(np.meshgrid(lines[a], lines[b], indexing="ij"), -1).reshape(-1, 2)
        for sign in (-1, 1):
            face = np.empty((len(grid), 3))
            face[:, axis], face[:, a], face[:, b] = sign * half[axis], grid[:, 0], grid[:, 1]
            faces.append(face)
    return np.vstack(faces)


ROD_SURFACE = _surface(ROD_SIZE, ROD_SAMPLE)


def rod_points(rod) -> np.ndarray:
    """World points over a rod's surface."""
    return rod.pos + ROD_SURFACE @ _rotation(rod.quat_xyzw).T


def box_gap(points, state, size) -> float:
    """Smallest distance (m) from world points to a box of that size at an object's pose; 0 for a point inside it,
    inf for no points."""
    if not len(points):
        return math.inf
    outside = np.maximum(np.abs((points - state.pos) @ _rotation(state.quat_xyzw)) - np.array(size) / 2, 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


def tilt_deg(state) -> float:
    """Degrees between an object's own vertical and the world's."""
    return math.degrees(math.acos(min(max(_rotation(state.quat_xyzw)[2, 2], -1.0), 1.0)))


# ---- the robot, as the simulator has it. The hooks are handed object states only, so the episode clock and the
# arms' joints are read (never changed) from the yam_arena embodiment that calls them ----
def _embodiment():
    frame = sys._getframe(1)
    while frame is not None:
        owner = frame.f_locals.get("self")
        if type(owner).__name__ == "YamArenaEmbodiment":
            return owner
        frame = frame.f_back
    raise RuntimeError("port_instrument_swap: no YamArenaEmbodiment on the call stack")


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
    """Points over each moving link's collision mesh (the URDF's), in the link's frame."""
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
            out[link.get("name")] = np.unique(np.round(np.vstack(clouds) / (MESH_SAMPLE / 2)), axis=0) * MESH_SAMPLE / 2
    return out


GRIPPER_LINKS = ("gripper", "tip_left", "tip_right")


def arm_points(joints, side, links=None) -> np.ndarray:
    """World points over one arm's links (all moving links, or the ones named) for its measured joints."""
    q = dict(zip(ARM_JOINTS, joints[:6], strict=True))
    q["joint7"] = q["joint8"] = joints[6]
    poses = forward_kinematics(YAM_URDF, q)
    clouds = [p @ poses[n][:3, :3].T + poses[n][:3, 3] for n, p in link_surfaces().items() if not links or n in links]
    return np.vstack(clouds) + np.asarray(station.BASES[side])


def robot_points(arms) -> np.ndarray:
    """World points over every moving link of both arms."""
    return np.vstack([arm_points(joints, side) for side, joints in arms.items()])


def hand_points(arms) -> np.ndarray:
    """World points over both grippers (bodies and fingers)."""
    return np.vstack([arm_points(joints, side, GRIPPER_LINKS) for side, joints in arms.items()])


# ---- task quantities ----
def post_gap(points, objects) -> float:
    """Smallest distance (m) from world points to either flange, or NEAR when nothing is that close."""
    close = points[(np.abs(points[:, 0] - PORT[0]) < POST_SIZE[0] / 2 + NEAR) & (np.abs(points[:, 1] - PORT[1]) < 0.05 + NEAR)]
    if not len(close):
        return NEAR
    return min(NEAR, *(box_gap(close, objects[name], POST_SIZE) for name in POSTS))


def post_moved(objects, name) -> float:
    """How far (m) a flange's centre is from its mark, seen from above."""
    return float(np.linalg.norm(objects[name].pos[:2] - POSTS[name].pos[:2]))


def post_pushed(objects, name) -> bool:
    """The flange is pushed out of place, tilted or down."""
    post = objects[name]
    return bool(post_moved(objects, name) > POST_PUSHED or tilt_deg(post) > POST_PUSHED_DEG or post.high[2] - SKIN_TOP < POST_SIZE[2] / 2)


def post_touched(objects, name) -> bool:
    """The flange is off its mark or off upright by more than a resting flange ever is."""
    return post_moved(objects, name) >= POST_MOVED or tilt_deg(objects[name]) >= POST_TILTED_DEG


def posts_in_place(objects) -> bool:
    """Both flanges on their marks, upright and still."""
    return all(post_moved(objects, n) <= POST_END and tilt_deg(objects[n]) <= POST_END_DEG and objects[n].resting() for n in POSTS)


def over_well(rod) -> bool:
    """The rod's centre is over the hole."""
    return bool(np.all(np.abs(rod.pos[:2] - np.array(PORT)) < WELL_W / 2))


def in_port(rod) -> bool:
    """Some of the rod is down the hole."""
    return over_well(rod) and rod.low[2] < SKIN_TOP


def seated(rod) -> bool:
    """Standing in the port on its floor, upright and still."""
    return in_port(rod) and rod.low[2] - WELL_FLOOR <= SEATED and tilt_deg(rod) < UPRIGHT_DEG and rod.resting()


def in_tray(rod) -> bool:
    """At rest inside the tray's walls, any way up."""
    lo = np.array(TRAY_XY) - np.array(TRAY_INNER) / 2
    hi = np.array(TRAY_XY) + np.array(TRAY_INNER) / 2
    return bool(np.all(rod.low[:2] >= lo) and np.all(rod.high[:2] <= hi) and rod.low[2] < TRAY_FLOOR + TRAY_H and rod.resting())


# ---- what the final state does not show ----
class Tracker:
    """Reads the flanges, the rods and the arms after each control step and keeps, until the episode's next reset:
    the number of touches of the flanges and whether a flange was pushed out of place. A reset shows as the episode
    clock going back; nothing the robot does to the objects clears any of it."""

    def __init__(self):
        self.restart()
        self._t = None

    def restart(self):
        self.touches = 0
        self.in_touch = False  # a touch has begun and the flanges are not yet still and 5 mm clear
        self.disturbed = dict.fromkeys(POSTS, False)  # the flange has been off its mark at some time
        self.pushed = False
        self.gaps = (NEAR, NEAR)  # at the last step: a rod to a flange, the robot to a flange
        self.held = dict.fromkeys(RODS, False)  # at the last step, a gripper was within RELEASED of the rod

    def update(self, objects):
        """Read one step. success() and score() both ask in the same step, so a step is read once."""
        t = clock()
        if t == self._t:
            return
        if self._t is None or t < self._t:
            self.restart()
        self._t = t
        arms = arm_joints()
        rods = np.vstack([rod_points(objects[name]) for name in RODS])
        self.gaps = (post_gap(rods, objects), post_gap(robot_points(arms), objects))
        self.pushed |= any(post_pushed(objects, name) for name in POSTS)
        still = all(objects[name].resting() for name in POSTS)
        newly_off = any(post_touched(objects, name) and not self.disturbed[name] for name in POSTS)
        for name in POSTS:
            self.disturbed[name] |= post_touched(objects, name)
        if min(self.gaps) < TOUCH or not still or newly_off:
            if not self.in_touch:
                self.in_touch = True
                self.touches += 1
        elif min(self.gaps) > TOUCH_CLEAR:
            self.in_touch = False
        hands = hand_points(arms)
        for name in RODS:
            self.held[name] = box_gap(hands, objects[name], ROD_SIZE) < RELEASED


TRACKER = Tracker()


def _failed() -> bool:
    return TRACKER.pushed or TRACKER.touches >= MAX_TOUCHES


def grey_share(objects) -> float:
    rod = objects["grey_inst"]
    return GREY_VALUE if in_tray(rod) and not TRACKER.held["grey_inst"] else 0.0


def blue_share(objects) -> float:
    rod = objects["blue_inst"]
    if TRACKER.held["blue_inst"] or not in_port(rod):
        return 0.0
    return BLUE_VALUE if seated(rod) else BLUE_VALUE * BLUE_SHORT


# ---- hooks read by yam_arena (task_score) ----
def success(objects):
    """Score 1.0: the grey rod at rest in the tray, the blue rod seated in the port, both let go, the flanges on
    their marks and still, with no touch."""
    TRACKER.update(objects)
    clean = TRACKER.touches == 0 and not _failed() and posts_in_place(objects)
    return clean and grey_share(objects) == GREY_VALUE and blue_share(objects) == BLUE_VALUE


def score(objects):
    """0 once a flange was pushed or touched three times. Otherwise the grey rod's value in the tray plus the blue
    rod's value in the port (half if short of the floor), halved once per touch."""
    TRACKER.update(objects)
    if _failed():
        return 0.0
    return round((grey_share(objects) + blue_share(objects)) * TOUCH_FACTOR**TRACKER.touches, 4)


# ---- demo ----
GRIP = -(ROD_SIDE / 2 - 0.004)  # finger target on a rod: each pad 4 mm into it (a full close twists the wrist)
HOLD_BELOW_TOP = 0.015  # the pads hold a rod this far below its top
TCP_ABOVE_BASE = ROD_LEN - HOLD_BELOW_TOP + PAD_TIP_PAST_TCP  # the TCP is this far above a held rod's base
GRASP_Z = WELL_FLOOR + TCP_ABOVE_BASE  # on a rod standing in the port or the stand
CARRY_Z = SKIN_TOP + POST_SIZE[2] + 0.010 + TCP_ABOVE_BASE  # a carried rod's base 1 cm above the flange tops
SEAT_Z = WELL_FLOOR + 0.002 + TCP_ABOVE_BASE  # the blue rod let go with its base 2 mm above the floor
DROP_Z = TRAY_FLOOR + 0.010 + TCP_ABOVE_BASE  # the grey rod let go with its base 1 cm above the tray's floor
STROKE_S = 3.0  # a rod's travel down or up the hole, slow enough for the arm to track it
ANGLED_RISE, ANGLED_DROP = 0.010, 0.003  # the angled variant: off the axis by this much on the way out and in


def _stroke(xy, z0, z1, finger, label, dy=0.0) -> list:
    """The TCP from z0 to z1 over the point xy, in two legs so the arm settles before the pads pass the flanges."""
    mid = (z0 + z1) / 2
    return [
        Waypoint(STROKE_S / 2, (xy[0], xy[1] + dy / 2, mid), finger, label),
        Waypoint(STROKE_S / 2, (xy[0], xy[1] + dy, z1), finger, label),
    ]


def demo(variant="scripted"):
    """Right arm, grip yaw 0: close on the grey rod 1.5 cm below its top, draw it straight up out of the hole until its
    base clears the flanges, carry it over the tray, lower it and let it go (it falls over in the tray). Close on the
    blue rod in the stand, lift it, carry it over the hole, lower it straight down until its base is 2 mm above the
    floor, let go, rise and move clear.

    variant="angled" is for the offline check only: the grey rod comes out along a path 1 cm off the hole's axis
    and the blue rod goes down 3 mm off it."""
    dy_out, dy_in = (ANGLED_RISE, ANGLED_DROP) if variant == "angled" else (0.0, 0.0)
    clear = (PORT[0], PORT[1] - 0.10)
    plan = [
        Waypoint(2.5, (*PORT, CARRY_Z), FINGER_OPEN, "over the grey rod"),
        *_stroke(PORT, CARRY_Z, GRASP_Z, FINGER_OPEN, "descend"),
        Waypoint(0.5, (*PORT, GRASP_Z), FINGER_OPEN, "settle"),
        Waypoint(1.0, (*PORT, GRASP_Z), GRIP, "close"),
        Waypoint(0.3, (*PORT, GRASP_Z), GRIP, "hold"),
        *_stroke(PORT, GRASP_Z, CARRY_Z, GRIP, "draw out", dy_out),
        Waypoint(3.0, (*TRAY_XY, CARRY_Z), GRIP, "to the tray"),
        Waypoint(1.5, (*TRAY_XY, DROP_Z), GRIP, "lower"),
        Waypoint(0.8, (*TRAY_XY, DROP_Z), FINGER_OPEN, "let go"),
        Waypoint(1.0, (*TRAY_XY, CARRY_Z), FINGER_OPEN, "rise"),
        Waypoint(3.0, (*STAND_XY, CARRY_Z), FINGER_OPEN, "over the blue rod"),
        Waypoint(1.5, (*STAND_XY, GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.5, (*STAND_XY, GRASP_Z), FINGER_OPEN, "settle"),
        Waypoint(1.0, (*STAND_XY, GRASP_Z), GRIP, "close"),
        Waypoint(0.3, (*STAND_XY, GRASP_Z), GRIP, "hold"),
        Waypoint(1.5, (*STAND_XY, CARRY_Z), GRIP, "lift"),
        Waypoint(3.0, (*PORT, CARRY_Z), GRIP, "over the port"),
        Waypoint(0.8, (*PORT, CARRY_Z), GRIP, "settle"),
        *_stroke((PORT[0], PORT[1] + dy_in), CARRY_Z, SEAT_Z, GRIP, "down the port"),
        Waypoint(0.5, (PORT[0], PORT[1] + dy_in, SEAT_Z), GRIP, "settle"),
        Waypoint(0.8, (PORT[0], PORT[1] + dy_in, SEAT_Z), FINGER_OPEN, "let go"),
        *_stroke((PORT[0], PORT[1] + dy_in), SEAT_Z, CARRY_Z, FINGER_OPEN, "rise"),
        Waypoint(2.0, (*clear, CARRY_Z), FINGER_OPEN, "move clear"),
    ]
    return {"right": plan}
