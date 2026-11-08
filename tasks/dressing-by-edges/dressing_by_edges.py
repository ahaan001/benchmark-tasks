"""On the station table, in front of the right arm: a green drape, 44 x 30 cm, with a skin-coloured silicone pad,
26 x 26 x 1 cm, on it, centred 30 cm in front of the arm bases and 14 cm to the right of their midpoint. A red
wound insert, 4 x 3 cm, lies on the middle of the pad. To the right of the drape, on the bare table, an opened
sterile dressing pack: a pale blue tray 5 x 5 cm and 2.5 cm tall, 18 cm out and 36 cm right, with a white
dressing pad, 7.5 x 7.5 cm and 4 mm thick, lying on it so 1.25 cm of the pad overhangs the tray on every side.
Beside the pack a roll of tape (5 cm across), further out a purple glove box, and near the arm's base a saline
bottle; none of them is part of the task. The right arm reaches everything.

The dressing is picked up by its edge faces (the pads close on two opposite edges with 9.5 mm of room each side),
carried flat and laid over the wound. Its central 5 x 5 cm, top and underside, is the key part: the surface that
lies on the wound, never touched.

Score (task_score), from every step of the trial and the final state:
- a key-part touch (any part of the robot within 2 mm of the central 5 x 5 cm) halves the score; a second scores 0.
- 0 unless the dressing ends flat on the skin pad (tilted under 5 degrees), let go and at rest.
- then 1.0 if the whole wound outline is under the dressing, 0.5 if only the wound's centre is, 0 otherwise.
- halved if the dressing was slid more than 2 cm across the skin after it first touched it.

Run it:  inspect-robots run --embodiment yam_arena -E task=dressing_by_edges.py -E object_jitter=0 \\
           --instruction "Take the white dressing from its opened pack and lay it flat over the red wound so the \\
wound is completely covered, touching the dressing only by its edges, then let go." --scorer task_score ...
"""

import functools
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box, Cylinder
from yam_arena.urdf import forward_kinematics, rpy_matrix

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # the dressing flat over the whole wound, let go and at rest, this long
TOP = TABLE_TOP[SCENE]

# ---- layout (m). x forward from the arm bases, y left; the right arm's base is at (0, -0.305) ----
DRAPE_CENTRE = (0.28, -0.14)
DRAPE_SIZE = (0.44, 0.30, 0.002)
DRAPE_TOP = TOP + DRAPE_SIZE[2]
SKIN_CENTRE = (0.30, -0.14)
SKIN_SIZE = (0.26, 0.26, 0.010)
SKIN_TOP = DRAPE_TOP + SKIN_SIZE[2]
WOUND_CENTRE = SKIN_CENTRE
WOUND_SIZE = (0.04, 0.03, 0.002)
WOUND_TOP = SKIN_TOP + WOUND_SIZE[2]
PACK_CENTRE = (0.18, -0.36)
PACK_SIZE = (0.05, 0.05, 0.025)  # the opened pack's inner tray; the dressing overhangs it on every side
PACK_TOP = TOP + PACK_SIZE[2]
DRESSING_SIZE = (0.075, 0.075, 0.004)
DRESSING_MASS = 0.006
KEY_HALF = 0.025  # the central 5 x 5 cm of the dressing, its whole thickness, is the key part
KEY_SIZE = (2 * KEY_HALF, 2 * KEY_HALF, DRESSING_SIZE[2])
TAPE_CENTRE = (0.26, -0.36)
TAPE_RADIUS, TAPE_HEIGHT = 0.025, 0.012
SALINE_CENTRE = (0.10, -0.40)
SALINE_RADIUS, SALINE_HEIGHT = 0.020, 0.09
GLOVES_CENTRE = (0.62, -0.36)
GLOVES_SIZE = (0.12, 0.07, 0.06)

# ---- scoring ----
KEY_TOUCH = 0.002  # a robot point this close to the key part is a touch
KEY_CLEAR = 0.005  # the touch is over once every point is this far away again
PENALTY = 0.5  # per key-part touch, and for sliding the dressing across the skin
MAX_TOUCHES = 2  # this many touches: 0
ON_SKIN = 0.003  # the dressing's underside within this of the skin's top (it may rest on the 2 mm wound insert)
TILT_DEG = 5.0  # flat: tilted less than this
DRAG = 0.02  # slid this far across the skin after first touching it: halved
PARTIAL = 0.5  # the wound's centre covered but not its whole outline
RELEASED = 0.02  # let go: every part of both grippers this far from the dressing
MESH_SAMPLE = 0.002  # spacing of the points checked on the robot's collision meshes

GREEN = (0.25, 0.55, 0.45)
SKIN = (0.87, 0.68, 0.58)
RED = (0.85, 0.15, 0.15)
PALE_BLUE = (0.75, 0.82, 0.90)
WHITE = (0.97, 0.97, 0.97)
TAPE_WHITE = (0.90, 0.90, 0.88)
CLEAR_BLUE = (0.70, 0.80, 0.90)
PURPLE = (0.45, 0.30, 0.60)


def _box(name, xy, size, under, color, mass=None) -> Box:
    """A box with its centre over xy and its underside at height `under`."""
    return Box(name, pos=(*xy, under + size[2] / 2), size=size, color=color, mass=mass)


def _cylinder(name, xy, radius, height, under, color, mass=None) -> Cylinder:
    return Cylinder(name, pos=(*xy, under + height / 2), radius=radius, height=height, color=color, mass=mass)


FIXED = [
    _box("drape", DRAPE_CENTRE, DRAPE_SIZE, TOP, GREEN),
    _box("skin", SKIN_CENTRE, SKIN_SIZE, DRAPE_TOP, SKIN),
    _box("wound", WOUND_CENTRE, WOUND_SIZE, SKIN_TOP, RED),
    _box("pack", PACK_CENTRE, PACK_SIZE, TOP, PALE_BLUE),
    _cylinder("saline", SALINE_CENTRE, SALINE_RADIUS, SALINE_HEIGHT, TOP, CLEAR_BLUE),
    _box("gloves", GLOVES_CENTRE, GLOVES_SIZE, TOP, PURPLE),
]
DRESSING = _box("dressing", PACK_CENTRE, DRESSING_SIZE, PACK_TOP, WHITE, mass=DRESSING_MASS)
TAPE = _cylinder("tape", TAPE_CENTRE, TAPE_RADIUS, TAPE_HEIGHT, TOP, TAPE_WHITE, mass=0.02)
OBJECTS = [*FIXED, DRESSING, TAPE]


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


def box_gap(points, state, size) -> float:
    """Smallest distance (m) from world points to a box of that size centred at an object's pose; 0 for a point
    inside it, inf for no points."""
    if not len(points):
        return math.inf
    outside = np.maximum(np.abs((points - state.pos) @ _rotation(state.quat_xyzw)) - np.array(size) / 2, 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


def _inside(xy, centre, size) -> bool:
    return bool(np.all(np.abs(np.asarray(xy) - centre) < np.array(size) / 2))


# ---- the robot, as the simulator has it. The hooks are handed object states only, so the episode clock and the
# arms' joints are read (never changed) from the yam_arena embodiment that calls them ----
def _embodiment():
    frame = sys._getframe(1)
    while frame is not None:
        owner = frame.f_locals.get("self")
        if type(owner).__name__ == "YamArenaEmbodiment":
            return owner
        frame = frame.f_back
    raise RuntimeError("dressing_by_edges: no YamArenaEmbodiment on the call stack")


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


def hand_points(arms) -> np.ndarray:
    """World points over both grippers (bodies and fingers)."""
    return np.vstack([arm_points(joints, side, GRIPPER_LINKS) for side, joints in arms.items()])


# ---- task quantities ----
def tilt_deg(dressing) -> float:
    """Degrees between the dressing's faces and the horizontal."""
    return math.degrees(math.acos(min(abs(_rotation(dressing.quat_xyzw)[2, 2]), 1.0)))


def on_skin(dressing) -> bool:
    """Lying on the skin pad: centre over it, underside at its top (or on the wound insert, 2 mm up)."""
    return _inside(dressing.pos[:2], SKIN_CENTRE, SKIN_SIZE[:2]) and abs(dressing.low[2] - SKIN_TOP) <= ON_SKIN


def covers(dressing, xy) -> bool:
    """Whether a point of the skin, seen from above, is under the dressing."""
    local = (np.array([*xy, dressing.pos[2]]) - dressing.pos) @ _rotation(dressing.quat_xyzw)
    return bool(np.all(np.abs(local[:2]) <= np.array(DRESSING_SIZE[:2]) / 2))


WOUND_CORNERS = [
    (WOUND_CENTRE[0] + sx * WOUND_SIZE[0] / 2, WOUND_CENTRE[1] + sy * WOUND_SIZE[1] / 2)
    for sx in (-1, 1)
    for sy in (-1, 1)
]


def coverage(dressing) -> float:
    """1 with the whole wound outline under the dressing, PARTIAL with only the wound's centre under it, else 0."""
    if all(covers(dressing, corner) for corner in WOUND_CORNERS):
        return 1.0
    return PARTIAL if covers(dressing, WOUND_CENTRE) else 0.0


# ---- what the final state does not show ----
class Tracker:
    """Reads the dressing and the arms after each control step and keeps, until the episode's next reset: how
    many times the robot touched the key part, and whether the dressing was slid across the skin. A reset shows as
    the episode clock going back; nothing the robot does to the dressing clears either."""

    def __init__(self):
        self.restart()
        self._t = None

    def restart(self):
        self.touches = 0
        self.touching = False
        self.key_gap = math.inf  # at the last step: the robot's distance to the key part
        self.held = False  # at the last step: a gripper was within RELEASED of the dressing
        self.contact = None  # where the dressing's centre was when its current rest on the skin began
        self.slid = 0.0  # how far it has moved across the skin since then
        self.dragged = False

    def update(self, objects):
        """Read one step. success() and score() both ask in the same step, so a step is read once."""
        t = clock()
        if t == self._t:
            return
        if self._t is None or t < self._t:
            self.restart()
        self._t = t
        hands = hand_points(arm_joints())
        dressing = objects["dressing"]
        self.key_gap = box_gap(hands, dressing, KEY_SIZE)
        if self.key_gap < KEY_TOUCH and not self.touching:
            self.touches += 1
            self.touching = True
        elif self.key_gap > KEY_CLEAR:
            self.touching = False
        self.held = box_gap(hands, dressing, DRESSING_SIZE) < RELEASED
        if on_skin(dressing):
            if self.contact is None:
                self.contact = dressing.pos[:2].copy()
            self.slid = float(np.linalg.norm(dressing.pos[:2] - self.contact))
            if self.slid > DRAG:
                self.dragged = True
        else:
            self.contact, self.slid = None, 0.0


TRACKER = Tracker()


def laid(objects) -> bool:
    """The dressing lies flat on the skin pad, let go and at rest."""
    TRACKER.update(objects)
    dressing = objects["dressing"]
    return on_skin(dressing) and tilt_deg(dressing) < TILT_DEG and not TRACKER.held and dressing.resting()


# ---- hooks read by yam_arena (task_score) ----
def success(objects):
    """Score 1.0: the dressing flat on the skin over the whole wound, let go and at rest, the key part never
    touched, the dressing never slid across the skin."""
    TRACKER.update(objects)
    return laid(objects) and coverage(objects["dressing"]) == 1.0 and TRACKER.touches == 0 and not TRACKER.dragged


def score(objects):
    """0 after a second key-part touch, or unless the dressing lies flat on the skin, let go and at rest. Otherwise
    its coverage of the wound, halved per key-part touch and halved if it was slid across the skin."""
    TRACKER.update(objects)
    if TRACKER.touches >= MAX_TOUCHES or not laid(objects):
        return 0.0
    factor = PENALTY**TRACKER.touches * (PENALTY if TRACKER.dragged else 1.0)
    return round(coverage(objects["dressing"]) * factor, 4)


# ---- demo ----
GRIP_YAW = 0.0  # the fingers close along y, on the dressing's two edge faces beside the pack
TRAVEL = TOP + 0.10  # carry height of the TCP
TIPS_BELOW_UNDERSIDE = 0.001  # at the grasp the pad tips sit this far below the dressing's underside, beside the pack
GRIP_IN = 0.003  # finger target on the dressing: each pad this far into it (a full close twists the wrist)
DROP = 0.0025  # the dressing is let go with its underside this far above the skin (0.5 mm above the wound insert)
NUDGE = 0.03  # naive variant: set down this far off along x, then pushed into place
PUSH_CLEAR = 0.003  # naive variant: the closed fingers start this far from the dressing's edge face
BLADE_HALF = 0.0085  # half the closed fingers' width along x (17 mm at the tips)
GRASP_Z = PACK_TOP - TIPS_BELOW_UNDERSIDE + PAD_TIP_PAST_TCP
PLACE_Z = SKIN_TOP + DROP - TIPS_BELOW_UNDERSIDE + PAD_TIP_PAST_TCP
PUSH_Z = SKIN_TOP + 0.0015 + PAD_TIP_PAST_TCP  # naive variant: pad tips 1.5 mm above the skin
HOLD = -(DRESSING_SIZE[1] / 2 - GRIP_IN)


def _lay(x, y) -> list:
    """Pick the dressing from the pack by its edge faces, carry it flat and set it down centred over (x, y)."""
    px, py = PACK_CENTRE
    return [
        Waypoint(2.0, (px, py, TRAVEL), FINGER_OPEN, "dressing: over the pack", GRIP_YAW),
        Waypoint(1.5, (px, py, GRASP_Z), FINGER_OPEN, "dressing: descend", GRIP_YAW),
        Waypoint(0.4, (px, py, GRASP_Z), FINGER_OPEN, "dressing: settle", GRIP_YAW),
        Waypoint(0.8, (px, py, GRASP_Z), HOLD, "dressing: close", GRIP_YAW),
        Waypoint(0.5, (px, py, GRASP_Z), HOLD, "dressing: hold", GRIP_YAW),
        Waypoint(1.2, (px, py, TRAVEL), HOLD, "dressing: lift", GRIP_YAW),
        Waypoint(2.5, (x, y, TRAVEL), HOLD, "dressing: carry", GRIP_YAW),
        Waypoint(1.5, (x, y, PLACE_Z), HOLD, "dressing: lower", GRIP_YAW),
        Waypoint(0.5, (x, y, PLACE_Z), HOLD, "dressing: settle", GRIP_YAW),
        Waypoint(0.6, (x, y, PLACE_Z), FINGER_OPEN, "dressing: release", GRIP_YAW),
        Waypoint(1.0, (x, y, TRAVEL), FINGER_OPEN, "dressing: rise", GRIP_YAW),
    ]


def _push(x_from, x_to, y) -> list:
    """Naive variant: the closed fingers come down beyond the dressing's far edge and push it along -x."""
    start = x_from + DRESSING_SIZE[0] / 2 + PUSH_CLEAR + BLADE_HALF
    end = x_to + DRESSING_SIZE[0] / 2 + BLADE_HALF
    return [
        Waypoint(1.5, (start, y, TRAVEL), FINGER_CLOSED, "push: beyond the far edge", GRIP_YAW),
        Waypoint(1.5, (start, y, PUSH_Z), FINGER_CLOSED, "push: descend", GRIP_YAW),
        Waypoint(0.4, (start, y, PUSH_Z), FINGER_CLOSED, "push: settle", GRIP_YAW),
        Waypoint(3.0, (end, y, PUSH_Z), FINGER_CLOSED, "push: slide it into place", GRIP_YAW),
        Waypoint(0.5, (end, y, PUSH_Z), FINGER_CLOSED, "push: settle", GRIP_YAW),
        Waypoint(1.0, (end, y, TRAVEL), FINGER_CLOSED, "push: rise", GRIP_YAW),
    ]


def demo(variant="scripted"):
    """Right arm: pinch the dressing's two edge faces where they overhang the pack tray, with the pad tips 1 mm
    below its underside and the fingers 9.5 mm clear of it on the way down, lift it straight up to 10 cm, carry
    it flat over the wound, lower it to 2.5 mm above the skin, open, rise and move clear.

    variant="nudge" is for the offline check only: the dressing is set down 3 cm off along x and then pushed into
    place with the closed fingers, the natural correction, which slides it across the skin."""
    x, y = WOUND_CENTRE
    if variant == "nudge":
        plan = _lay(x + NUDGE, y) + _push(x + NUDGE, x, y)
    else:
        plan = _lay(x, y)
    plan.append(Waypoint(1.5, (x, y - 0.10, TRAVEL), plan[-1].finger, "clear", GRIP_YAW))
    return {"right": plan}
