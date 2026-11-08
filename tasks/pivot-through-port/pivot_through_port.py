"""On the station table, in front of the right arm: a grey wall, 2 cm thick and 14 cm high, stands across the table
46 cm out, in two halves with a 16 cm gap between them that a lintel closes from 10.5 cm up. Two loose pink posts,
1.5 x 3.5 x 10 cm, stand in the gap and leave a 7 cm slot between them, 15 cm to the right of the midline: the
port. Behind the wall, 13 cm past its far face, three red pegs, 2 x 2 x 8 cm, stand on the slot's axis, 7 cm to the
left of it and 9 cm to the right of it. An aluminium rod, 35 cm long and 1.2 cm square, lies on two wooden blocks
beside the right end of the wall, pointing away from the robot. The pegs can only be reached with the rod through
the slot, and an off-axis peg only by swinging the rod about the slot: sliding it sideways pushes a post.

Score (task_score), from every step of the trial and the final state:
- 0 if a post was ever pushed more than 1 cm or tilted more than 15 degrees, if the rod ever reached behind the wall
  any other way than through the slot, if any part of the robot ever reached behind the wall below 16 cm,
  or if a post was touched three times. A touch is the rod or the robot within 2 mm of a post, or a post moved; it
  ends once the posts are still and everything is 5 mm clear. All of this stays for the rest of the trial.
- Otherwise each peg that fell (tilted over 60 degrees or moved over 3 cm) earns its value: 0.2 on the axis, 0.3
  for the one 7 cm to the left, 0.5 for the one 9 cm to the right. The sum is halved once per touch, and halved if
  the rod is not lying back on its blocks and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=pivot_through_port.py -E object_jitter=0 \\
           --instruction "Use the rod to knock over as many of the three red pegs behind the grey wall as you can, \\
then lay the rod back on its two wooden blocks." --scorer task_score ...
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
SUCCESS_HOLD_S = 1.0  # the rod must stay at rest on its blocks, let go, this long
TOP = TABLE_TOP[SCENE]

# ---- layout (m). x forward from the arm bases, y left; the right arm's base is at (0, -0.305) ----
Y0 = -0.15  # the slot's axis
WALL_X = (0.46, 0.48)  # near and far face
WALL_H = 0.14
WALL_T = WALL_X[1] - WALL_X[0]
WALL_LEN = 0.15  # each half, along y
SLOT_W = 0.07  # the simulated arm settles 1 to 2.5 cm off a commanded pose, with 1 cm of hysteresis, and drifts 2 cm sideways while moving: the slot leaves room for that
POST_SIZE = (0.015, 0.035, 0.10)  # thinner than the wall: the rod's lateral offset across a post's depth grows with the angle
POST_GAP = 0.01  # between a post and the wall half beside it
LINTEL_Z0 = POST_SIZE[2] + 0.005  # underside of the lintel
WALL_Y = (Y0 - SLOT_W / 2 - POST_SIZE[1] - POST_GAP - WALL_LEN, Y0 + SLOT_W / 2 + POST_SIZE[1] + POST_GAP + WALL_LEN)
SLOT_X = (WALL_X[0] + WALL_X[1]) / 2  # the pivot point: the slot's centre
PEG_SIZE = (0.02, 0.02, 0.08)
PEG_X = WALL_X[1] + 0.13
PEG_SPECS = {"peg_axis": (0.0, 0.2), "peg_near": (0.07, 0.3), "peg_far": (-0.09, 0.5)}  # y offset from Y0, value
ROD_SIZE = (0.35, 0.012, 0.012)  # long axis along x at the start
ROD_SIDE = ROD_SIZE[1]
ROD_Y = -0.41
ROD_X = (0.11, 0.46)  # tail and tip at the start
REST_SIZE = (0.03, 0.03, 0.03)
REST_X = (0.14, 0.43)
REST_TOP = TOP + REST_SIZE[2]
GRIP_FROM_TIP = 0.28  # the demo holds the rod this far behind its tip: further back, the gripper works within 10 cm of its base's axis, where the IK flips

# ---- scoring ----
VALUES = {name: value for name, (_, value) in PEG_SPECS.items()}
TOUCH = 0.002  # a touch starts when the rod or the robot comes this close to a post
TOUCH_CLEAR = 0.005  # and ends once everything is this far from both posts again and they are still
TOUCH_FACTOR = 0.5  # each touch halves the score
MAX_TOUCHES = 3  # the third touch scores 0
POST_MOVED, POST_TILTED_DEG = 0.003, 3.0  # a post this far from its mark or this far from upright was touched
POST_PUSHED, POST_PUSHED_DEG = 0.010, 15.0  # a post pushed this far: score 0
PEG_FALLEN_DEG, PEG_FALLEN_MOVE = 60.0, 0.03  # a peg is down once it tilts or moves this much
CAVITY_X = WALL_X[1]  # a robot point past the wall's far face, lower than CAVITY_Z, is reaching into the cavity: score 0
CAVITY_Z = WALL_H + 0.02
SLOT_MARGIN = 0.01  # the slot's opening, widened this much, is the only way the rod may cross the wall
RELEASED = 0.02  # the rod is let go when every part of both grippers is this far from it
ON_RESTS = 0.005  # lying on the blocks: underside within this of their tops
REST_OFF_AXIS = 0.02  # and the rod's centre within this of the blocks' line
PARK_FACTOR = 0.5  # share of the score with the rod not back on its blocks
ROD_SAMPLE = 0.0025  # spacing of the points checked on the rod's surface
MESH_SAMPLE = 0.004  # spacing of the points checked on the robot's collision meshes
NEAR = 0.05  # distances to a post beyond this are not measured


def _box(name, xy, size, under, color, mass=None, yaw=0.0) -> Box:
    return Box(name, pos=(*xy, under + size[2] / 2), size=size, mass=mass, color=color, yaw=yaw)


GREY, PINK, RED, ALUMINIUM, WOOD = (0.55, 0.55, 0.57), (0.88, 0.40, 0.45), (0.80, 0.05, 0.08), (0.75, 0.76, 0.78), (0.60, 0.45, 0.30)
_post_y = {"post_right": Y0 - SLOT_W / 2 - POST_SIZE[1] / 2, "post_left": Y0 + SLOT_W / 2 + POST_SIZE[1] / 2}
FIXED = [
    _box("wall_right", (SLOT_X, WALL_Y[0] + WALL_LEN / 2), (WALL_T, WALL_LEN, WALL_H), TOP, GREY),
    _box("wall_left", (SLOT_X, WALL_Y[1] - WALL_LEN / 2), (WALL_T, WALL_LEN, WALL_H), TOP, GREY),
    _box("lintel", (SLOT_X, Y0), (WALL_T, WALL_Y[1] - WALL_Y[0] - 2 * WALL_LEN, WALL_H - LINTEL_Z0), TOP + LINTEL_Z0, GREY),
    _box("rest_a", (REST_X[0], ROD_Y), REST_SIZE, TOP, WOOD),
    _box("rest_b", (REST_X[1], ROD_Y), REST_SIZE, TOP, WOOD),
]
POSTS = {name: _box(name, (SLOT_X, y), POST_SIZE, TOP, PINK, mass=0.04) for name, y in _post_y.items()}
PEGS = {name: _box(name, (PEG_X, Y0 + dy), PEG_SIZE, TOP, RED, mass=0.02) for name, (dy, _) in PEG_SPECS.items()}
ROD = _box("rod", (sum(ROD_X) / 2, ROD_Y), ROD_SIZE, REST_TOP, ALUMINIUM, mass=0.045)
OBJECTS = [*FIXED, *POSTS.values(), *PEGS.values(), ROD]


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
    """World points over the rod's surface."""
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
    raise RuntimeError("pivot_through_port: no YamArenaEmbodiment on the call stack")


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
    """Smallest distance (m) from world points to either post, or NEAR when nothing is that close."""
    close = points[np.abs(points[:, 0] - SLOT_X) < WALL_T / 2 + NEAR]
    if not len(close):
        return NEAR
    return min(NEAR, *(box_gap(close, objects[name], POST_SIZE) for name in POSTS))


def post_moved(objects, name) -> float:
    """How far (m) a post's centre is from its mark, seen from above."""
    return float(np.linalg.norm(objects[name].pos[:2] - POSTS[name].pos[:2]))


def post_pushed(objects, name) -> bool:
    """The post is pushed out of place, tilted or down."""
    post = objects[name]
    return bool(
        post_moved(objects, name) > POST_PUSHED
        or tilt_deg(post) > POST_PUSHED_DEG
        or post.high[2] - TOP < POST_SIZE[2] / 2
    )


def post_touched(objects, name) -> bool:
    """The post is off its mark or off upright by more than a resting post ever is."""
    return post_moved(objects, name) >= POST_MOVED or tilt_deg(objects[name]) >= POST_TILTED_DEG


def in_cavity(points) -> bool:
    """Some robot point is behind the wall (past its far face) and lower than the wall top plus 2 cm."""
    return bool(np.any((points[:, 0] > CAVITY_X) & (points[:, 2] < TOP + CAVITY_Z)))


def round_wall(rod) -> bool:
    """The rod reaches behind the wall (past its far face, within the wall's span) with no part of it in the slot's
    opening: it went over the wall or round an end."""
    points = rod_points(rod)
    behind = (points[:, 0] > WALL_X[1]) & (points[:, 1] > WALL_Y[0]) & (points[:, 1] < WALL_Y[1])
    if not behind.any():
        return False
    in_slab = (points[:, 0] >= WALL_X[0]) & (points[:, 0] <= WALL_X[1])
    in_slot = in_slab & (np.abs(points[:, 1] - Y0) < SLOT_W / 2 + SLOT_MARGIN) & (points[:, 2] < TOP + LINTEL_Z0)
    return not in_slot.any()


def peg_fallen(objects, name) -> bool:
    peg = objects[name]
    return tilt_deg(peg) > PEG_FALLEN_DEG or float(np.linalg.norm(peg.pos[:2] - PEGS[name].pos[:2])) > PEG_FALLEN_MOVE


def on_rests(rod) -> bool:
    """At rest on both blocks: lowest point on their tops, centre on their line, both blocks under it."""
    return bool(
        abs(rod.low[2] - REST_TOP) <= ON_RESTS
        and abs(rod.pos[1] - ROD_Y) < REST_OFF_AXIS
        and rod.low[0] < REST_X[0]
        and rod.high[0] > REST_X[1]
        and rod.resting()
    )


# ---- what the final state does not show ----
class Tracker:
    """Reads the posts, the pegs, the rod and the arms after each control step and keeps, until the episode's next
    reset: the number of touches of the posts, whether a post was pushed out of place, whether the rod reached
    behind the wall any way but through the slot, whether the robot reached into the cavity, and which pegs fell.
    A reset shows as the episode clock going back; nothing the robot does to the objects clears any of it."""

    def __init__(self):
        self.restart()
        self._t = None

    def restart(self):
        self.touches = 0
        self.in_touch = False  # a touch has begun and the posts are not yet still and 5 mm clear
        self.disturbed = dict.fromkeys(POSTS, False)  # the post has been off its mark at some time
        self.pushed = False
        self.cavity = False
        self.round_wall = False
        self.fallen = dict.fromkeys(VALUES, False)
        self.gaps = (NEAR, NEAR)  # at the last step: rod to a post, robot to a post
        self.held = False  # at the last step, a gripper was within RELEASED of the rod

    def update(self, objects):
        """Read one step. success() and score() both ask in the same step, so a step is read once."""
        t = clock()
        if t == self._t:
            return
        if self._t is None or t < self._t:
            self.restart()
        self._t = t
        arms = arm_joints()
        robot = robot_points(arms)
        rod = objects["rod"]
        self.gaps = (post_gap(rod_points(rod), objects), post_gap(robot, objects))
        self.cavity |= in_cavity(robot)
        self.round_wall |= round_wall(rod)
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
        for name in VALUES:
            self.fallen[name] |= peg_fallen(objects, name)
        self.held = box_gap(hand_points(arms), rod, ROD_SIZE) < RELEASED


TRACKER = Tracker()


def parked(objects) -> bool:
    """The rod lies back on its blocks and is let go."""
    TRACKER.update(objects)
    return on_rests(objects["rod"]) and not TRACKER.held


def _failed() -> bool:
    return TRACKER.pushed or TRACKER.cavity or TRACKER.round_wall or TRACKER.touches >= MAX_TOUCHES


# ---- hooks read by yam_arena (task_score) ----
def success(objects):
    """Score 1.0: all three pegs down, the rod at rest on its blocks and let go, the posts still, with no touch."""
    TRACKER.update(objects)
    clean = TRACKER.touches == 0 and not _failed() and all(objects[name].resting() for name in POSTS)
    return clean and all(TRACKER.fallen.values()) and parked(objects)


def score(objects):
    """0 once a post was pushed, the robot reached into the cavity, the rod got behind the wall any other way than
    through the slot, or a post was touched three times. Otherwise the fallen pegs' values, halved once per touch
    and halved if the rod is not back on its blocks."""
    TRACKER.update(objects)
    if _failed():
        return 0.0
    earned = sum(value for name, value in VALUES.items() if TRACKER.fallen[name])
    return round(earned * TOUCH_FACTOR**TRACKER.touches * (1.0 if parked(objects) else PARK_FACTOR), 4)


# ---- demo ----
PADS_ABOVE_UNDERSIDE = -0.003  # pad tips this far below the rod's underside: the pads hold its whole side (8 mm of it slipped)
GRIP = -(ROD_SIDE / 2 - 0.004)  # finger target on the rod: each pad 4 mm into it (a full close twists the wrist)
TCP_ABOVE_ROD_CENTRE = PAD_TIP_PAST_TCP + PADS_ABOVE_UNDERSIDE - ROD_SIDE / 2
PORT_ROD_BOTTOM = 0.04  # the rod passes the slot with its underside this far above the table
PORT_Z = TOP + PORT_ROD_BOTTOM + ROD_SIDE / 2 + TCP_ABOVE_ROD_CENTRE
GRASP_Z = REST_TOP + ROD_SIDE / 2 + TCP_ABOVE_ROD_CENTRE
TRAVEL_Z = TOP + 0.07  # carry height of the TCP: over the arm's own base, higher runs joint 4 out of range
CARRY_YAW = math.radians(-45)  # the rod is carried turned toward the right, so the gripper passes 22 cm from the
CARRY_X = 0.22  # base's axis (passing 10 cm from it, the IK flipped and threw the rod) with the tip short of the wall
CARRY_S = 7.0
BACK_OFF = 0.08  # the tip stays this far before the wall's near face while the rod is aimed
AIM_RADIUS = GRIP_FROM_TIP + BACK_OFF  # the gripper's distance from the slot's centre with the tip backed off
PUSH_PAST = 0.02  # a peg is pushed until the tip is this far past its centre
AIM_DEG_PER_S = 15.0
ENTER_S, BACK_S = 5.0, 4.0  # along the ray: the arm trails the target along the rod's axis, which costs no room
ORDER = ("peg_near", "peg_far", "peg_axis")  # the axis last, so the rod leaves the slot square to the wall
NAIVE_TIP_PAST = 0.10  # naive variant: the tip this far past the wall, then the gripper slides sideways


def _peg_angles() -> dict:
    """Angle of each peg from the slot's centre (rad) and its distance from it."""
    out = {}
    for name, (dy, _) in PEG_SPECS.items():
        dx = PEG_X - SLOT_X
        out[name] = (math.atan2(dy, dx), math.hypot(dx, dy))
    return out


PEG_ANGLES = _peg_angles()


def _on_ray(angle, radius, z) -> tuple[float, float, float]:
    """The TCP with the rod's axis through the slot's centre at `angle`, the gripper `radius` short of it."""
    return (SLOT_X - radius * math.cos(angle), Y0 - radius * math.sin(angle), z)


def _aim(a0, a1) -> list:
    """Turn the rod from angle a0 to a1 (rad) about the slot's centre with its tip backed off from the wall:
    waypoints every 5 degrees along the arc, then a pause."""
    n = max(1, round(abs(math.degrees(a1 - a0)) / 5.0))
    duration = abs(math.degrees(a1 - a0)) / AIM_DEG_PER_S / n
    arc = [Waypoint(duration, _on_ray(a, AIM_RADIUS, PORT_Z), GRIP, "aim", a) for a in np.linspace(a0, a1, n + 1)[1:]]
    return [*arc, Waypoint(0.7, _on_ray(a1, AIM_RADIUS, PORT_Z), GRIP, "settle", a1)]


def _push(name) -> list:
    """Advance along the rod's axis through the slot until the peg is pushed over, then back out of the slot."""
    angle, reach = PEG_ANGLES[name]
    contact = GRIP_FROM_TIP - (reach + PUSH_PAST)
    tag = name.removeprefix("peg_")
    return [
        Waypoint(ENTER_S, _on_ray(angle, contact, PORT_Z), GRIP, f"{tag}: push", angle),
        Waypoint(0.5, _on_ray(angle, contact, PORT_Z), GRIP, f"{tag}: settle", angle),
        Waypoint(BACK_S, _on_ray(angle, AIM_RADIUS, PORT_Z), GRIP, f"{tag}: back", angle),
        Waypoint(0.5, _on_ray(angle, AIM_RADIUS, PORT_Z), GRIP, f"{tag}: settle", angle),
    ]


def demo(variant="scripted"):
    """Right arm: grip the rod 28 cm behind its tip, lift it, turn it 45 degrees toward the right and carry it in
    front of the slot, lower it to port height, bring it square to the wall with the tip 8 cm short of it. For each
    peg (the near one, the far one, then the one on the axis): turn the rod about the slot's centre until its axis
    points through the slot at the peg, push it straight along that axis through the slot until the peg falls, draw
    it straight back out. Carry the rod back the same way, lower it onto its blocks and let go.

    variant="naive" is for the offline check only: instead of aiming, it slides the gripper sideways by each
    off-axis peg's offset with the tip 10 cm past the wall."""
    grip_x = ROD_X[1] - GRIP_FROM_TIP
    home = _on_ray(0.0, AIM_RADIUS, PORT_Z)
    plan = [
        Waypoint(2.0, (grip_x, ROD_Y, TRAVEL_Z), FINGER_OPEN, "to the rod"),
        Waypoint(1.5, (grip_x, ROD_Y, GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.5, (grip_x, ROD_Y, GRASP_Z), FINGER_OPEN, "settle"),
        Waypoint(1.0, (grip_x, ROD_Y, GRASP_Z), GRIP, "close"),
        Waypoint(0.5, (grip_x, ROD_Y, GRASP_Z), GRIP, "hold"),
        Waypoint(1.5, (grip_x, ROD_Y, TRAVEL_Z), GRIP, "lift"),
        Waypoint(2.5, (grip_x, ROD_Y, TRAVEL_Z), GRIP, "turn", CARRY_YAW),
        Waypoint(2.0, (CARRY_X, ROD_Y, TRAVEL_Z), GRIP, "out", CARRY_YAW),
        Waypoint(CARRY_S, (CARRY_X, Y0, TRAVEL_Z), GRIP, "to the slot", CARRY_YAW),
        Waypoint(0.5, (CARRY_X, Y0, TRAVEL_Z), GRIP, "settle", CARRY_YAW),
        Waypoint(1.5, (CARRY_X, Y0, PORT_Z), GRIP, "lower", CARRY_YAW),
        Waypoint(3.0, (home[0], Y0, PORT_Z), GRIP, "back off", CARRY_YAW),
        Waypoint(3.0, home, GRIP, "square up"),
        Waypoint(0.7, home, GRIP, "settle"),
    ]
    if variant == "naive":
        slide_x = WALL_X[1] + NAIVE_TIP_PAST - GRIP_FROM_TIP
        plan += [Waypoint(4.0, (slide_x, Y0, PORT_Z), GRIP, "axis: push")]
        for name in ("peg_near", "peg_far"):
            y = Y0 + PEG_SPECS[name][0]
            plan += [
                Waypoint(2.5, (slide_x, y, PORT_Z), GRIP, f"{name[4:]}: slide"),
                Waypoint(2.5, (slide_x, Y0, PORT_Z), GRIP, f"{name[4:]}: slide back"),
            ]
        plan += [Waypoint(4.0, home, GRIP, "withdraw")]
    else:
        angle = 0.0
        for name in ORDER:
            plan += _aim(angle, PEG_ANGLES[name][0])
            plan += _push(name)
            angle = PEG_ANGLES[name][0]
        plan += _aim(angle, 0.0)
    plan += [
        Waypoint(3.0, home, GRIP, "turn", CARRY_YAW),
        Waypoint(3.0, (CARRY_X, Y0, PORT_Z), GRIP, "forward", CARRY_YAW),
        Waypoint(1.5, (CARRY_X, Y0, TRAVEL_Z), GRIP, "rise", CARRY_YAW),
        Waypoint(CARRY_S, (CARRY_X, ROD_Y, TRAVEL_Z), GRIP, "to the blocks", CARRY_YAW),
        Waypoint(0.5, (CARRY_X, ROD_Y, TRAVEL_Z), GRIP, "settle", CARRY_YAW),
        Waypoint(2.5, (CARRY_X, ROD_Y, TRAVEL_Z), GRIP, "square up"),
        Waypoint(2.0, (grip_x, ROD_Y, TRAVEL_Z), GRIP, "over the blocks"),
        Waypoint(1.5, (grip_x, ROD_Y, GRASP_Z + ON_RESTS), GRIP, "lower"),
        Waypoint(0.5, (grip_x, ROD_Y, GRASP_Z + ON_RESTS), GRIP, "settle"),
        Waypoint(0.8, (grip_x, ROD_Y, GRASP_Z + ON_RESTS), FINGER_OPEN, "release"),
        Waypoint(1.5, (grip_x, ROD_Y, TRAVEL_Z), FINGER_OPEN, "rise"),
    ]
    return {"right": plan}
