"""On the station table, in front of the right arm: a skin-coloured silicone pad, 46 x 20 cm, with a wound channel
along its middle running away from the robot: two red strips 6 cm apart, 2.5 cm high, are the channel's edges. A
pink tissue band, 25 x 8 x 2 cm, lies across the strips over the channel, from 18 to 43 cm in front of the arm
bases, and passes under a yellow ligament bridge 26 to 28 cm out, so 15 cm of band shows beyond the bridge. Two
dark grey metal fragments, 2 cm cubes, lie on the channel floor under that far part of the band, 40 and 33.5 cm
out. A steel tray, 14 x 10 cm inside with 3 cm walls, is fixed to the table to the right of the pad. Either arm may
be used; the right arm reaches everything, the left arm only the band.

Sliding the band toward the robot uncovers the fragments: the one 40 cm out after 7 cm, the one 33.5 cm out after
13.5 cm. After 17 cm the band's far end comes out from under the bridge.

Score (task_score), from every step of the trial and the final state:
- 0 if the band ever came out from under the bridge. Halved if its far end ever came within 1.5 cm of the bridge's
  near face. Both stay for the rest of the trial.
- Otherwise each fragment that was lifted off the pad (not slid more than 3 cm first), is let go and is at rest
  earns its value on the tray's floor, half its value anywhere else off the channel: 0.4 for the one 40 cm out,
  0.6 for the one 33.5 cm out.
- Halved if at the end the band is not lying back on the strips, let go, with its near end within 2 cm of its mark.

Run it:  inspect-robots run --embodiment yam_arena -E task=retract_tissue_band.py -E object_jitter=0 \\
           --instruction "Put the two metal fragments that lie under the pink tissue band into the steel tray, then \\
lay the band back where it was." --scorer task_score ...
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
SUCCESS_HOLD_S = 1.0  # both fragments on the tray's floor and the band back on the strips, all let go, this long
TOP = TABLE_TOP[SCENE]

# ---- layout (m). x forward from the arm bases, y left; the right arm's base is at (0, -0.305) ----
Y0 = -0.14  # the channel's axis
PAD_X = (0.04, 0.50)  # the pad's near and far edge
PAD_SIZE = (PAD_X[1] - PAD_X[0], 0.20, 0.010)
PAD_CENTRE = (sum(PAD_X) / 2, Y0)
PAD_TOP = TOP + PAD_SIZE[2]
CHANNEL = 0.06  # between the strips' inner faces
STRIP_SIZE = (PAD_SIZE[0], 0.015, 0.025)  # each edge strip, the whole length of the pad
STRIP_OUTER = CHANNEL / 2 + STRIP_SIZE[1]  # the strips' outer faces, from the axis
STRIP_TOP = PAD_TOP + STRIP_SIZE[2]
BAND_SIZE = (0.25, 0.08, 0.02)
BAND_X = (0.18, 0.43)  # the band's near and far end at the start
BAND_TOP = STRIP_TOP + BAND_SIZE[2]
SLACK = 0.17  # band beyond the bridge's near face at the start
BRIDGE_X = (BAND_X[1] - SLACK, BAND_X[1] - SLACK + 0.02)  # the bridge's near and far face
BRIDGE_CENTRE = (sum(BRIDGE_X) / 2, Y0)
BRIDGE_SPAN = 0.14  # across the channel, post to post
BRIDGE_CLEAR = 0.003  # between the band's top and the bridge
BRIDGE_BAR = 0.015  # height of the bar
POST_SIZE = (BRIDGE_X[1] - BRIDGE_X[0], 0.015, BAND_TOP + BRIDGE_CLEAR - PAD_TOP)
FRAGMENT_SIZE = (0.02, 0.02, 0.02)
EXPOSE = 0.04  # a fragment is clear of the band once the band's far end is this far before the fragment's centre
# name: the slide toward the robot that uncovers it, its value
FRAGMENT_SPECS = {"fragment_shallow": (0.07, 0.4), "fragment_deep": (0.135, 0.6)}
FRAGMENT_X = {name: BAND_X[1] - slide + EXPOSE for name, (slide, _) in FRAGMENT_SPECS.items()}
TRAY_CENTRE = (0.30, -0.32)
TRAY_INSIDE = (0.14, 0.10)
TRAY_WALL = 0.003  # thickness of the walls and of the floor
TRAY_HEIGHT = 0.03  # top of the walls above the table
TRAY_FLOOR_TOP = TOP + TRAY_WALL

# ---- scoring ----
VALUES = {name: value for name, (_, value) in FRAGMENT_SPECS.items()}
WARN = 0.015  # the band's far end this close to the bridge's near face: the score is halved
WARN_FACTOR = 0.5
OPEN_FACTOR = 0.5  # the band not laid back at the end: the score is halved
BAND_BACK = 0.02  # laid back: the band's near end within this of its mark
BAND_OFF_AXIS = 0.01  # and its centre within this of the channel's axis
BAND_ASKEW_DEG = 10.0  # and turned no more than this from the channel
ON_STRIPS = 0.005  # and its underside within this of the strips' tops
LIFTED = 0.015  # a fragment is lifted once its underside is this far above the pad
DRAG = 0.03  # slid this far from its start before it was lifted: that fragment earns nothing
RELEASED = 0.02  # an object is let go when every part of both grippers is this far from it
IN_TRAY = 0.005  # lying on the tray's floor: underside within this of it
OFF_CHANNEL_CREDIT = 0.5  # share of the value for a fragment at rest off the channel but not on the tray's floor
OFF_TABLE = 0.02  # a fragment whose underside is this far below the table top has left the table
MESH_SAMPLE = 0.002  # spacing of the points checked on the robot's collision meshes

SKIN = (0.87, 0.68, 0.58)
RED = (0.80, 0.22, 0.22)
PINK = (0.93, 0.45, 0.62)
YELLOW = (0.95, 0.88, 0.50)
WOOD = (0.82, 0.70, 0.42)
STEEL = (0.70, 0.72, 0.74)
DARK = (0.30, 0.30, 0.32)


def _box(name, xy, size, under, color, mass=None) -> Box:
    """A box with its centre over xy and its underside at height `under`."""
    return Box(name, pos=(*xy, under + size[2] / 2), size=size, color=color, mass=mass)


def _tray() -> list:
    (x, y), (sx, sy), w = TRAY_CENTRE, TRAY_INSIDE, TRAY_WALL
    ends = [(f"tray_wall_{n}", (x + s * (sx + w) / 2, y), (w, sy + 2 * w)) for n, s in (("near", -1), ("far", 1))]
    sides = [(f"tray_wall_{n}", (x, y + s * (sy + w) / 2), (sx, w)) for n, s in (("right", -1), ("left", 1))]
    walls = [_box(n, xy, (*size, TRAY_HEIGHT), TOP, STEEL) for n, xy, size in ends + sides]
    return [_box("tray_floor", (x, y), (sx, sy, w), TOP, STEEL), *walls]


SIDES = (("right", -1), ("left", 1))
FIXED = [
    _box("pad", PAD_CENTRE, PAD_SIZE, TOP, SKIN),
    *(
        _box(f"strip_{n}", (PAD_CENTRE[0], Y0 + s * (STRIP_OUTER - STRIP_SIZE[1] / 2)), STRIP_SIZE, PAD_TOP, RED)
        for n, s in SIDES
    ),
    *(
        _box(f"post_{n}", (BRIDGE_CENTRE[0], Y0 + s * (BRIDGE_SPAN - POST_SIZE[1]) / 2), POST_SIZE, PAD_TOP, WOOD)
        for n, s in SIDES
    ),
    _box("bridge", BRIDGE_CENTRE, (POST_SIZE[0], BRIDGE_SPAN, BRIDGE_BAR), PAD_TOP + POST_SIZE[2], YELLOW),
    *_tray(),
]
BAND = _box("band", (sum(BAND_X) / 2, Y0), BAND_SIZE, STRIP_TOP, PINK, mass=0.1)
FRAGMENTS = {name: _box(name, (x, Y0), FRAGMENT_SIZE, PAD_TOP, DARK, mass=0.02) for name, x in FRAGMENT_X.items()}
OBJECTS = [*FIXED, BAND, *FRAGMENTS.values()]


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
    """Smallest distance (m) from world points to a box of that size at an object's pose; 0 for a point inside it,
    inf for no points."""
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
    raise RuntimeError("retract_tissue_band: no YamArenaEmbodiment on the call stack")


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
def band_ends(band) -> tuple[float, float]:
    """x of the centres of the band's near and far end faces."""
    half = abs(_rotation(band.quat_xyzw)[0, 0]) * BAND_SIZE[0] / 2
    return float(band.pos[0] - half), float(band.pos[0] + half)


def slack_left(band) -> float:
    """How much band is left beyond the bridge's near face (m); 0 or less: it has come out."""
    return band_ends(band)[1] - BRIDGE_X[0]


def band_out(band) -> bool:
    """No part of the band is under the bridge any more."""
    near, far = band_ends(band)
    return far <= BRIDGE_X[0] or near >= BRIDGE_X[1]


def band_askew(band) -> float:
    """Degrees between the band's long side and the channel."""
    return math.degrees(math.acos(min(abs(_rotation(band.quat_xyzw)[0, 0]), 1.0)))


def band_back(band) -> bool:
    """Lying back on the strips where it started: near end on its mark, centred on the channel, square to it,
    underside on the strips, at rest."""
    return bool(
        abs(band_ends(band)[0] - BAND_X[0]) < BAND_BACK
        and abs(band.pos[1] - Y0) < BAND_OFF_AXIS
        and band_askew(band) < BAND_ASKEW_DEG
        and abs(band.low[2] - STRIP_TOP) <= ON_STRIPS
        and band.resting()
    )


def in_tray(fragment) -> bool:
    """Lying on the tray's floor: centre inside the walls, underside on the floor."""
    return _inside(fragment.pos[:2], TRAY_CENTRE, TRAY_INSIDE) and fragment.low[2] - TRAY_FLOOR_TOP <= IN_TRAY


def off_channel(fragment) -> bool:
    """On the table or on something standing on it, with its centre outside the strips' outer faces or past the
    pad's ends: beside the channel on the pad, on the table, or in the tray but not on its floor."""
    x, y = fragment.pos[:2]
    in_channel = abs(y - Y0) <= STRIP_OUTER and PAD_X[0] <= x <= PAD_X[1]
    return not in_channel and fragment.low[2] > TOP - OFF_TABLE


# ---- what the final state does not show ----
class Tracker:
    """Reads the band, the fragments and the arms after each control step and keeps, until the episode's next
    reset: whether the band came out from under the bridge, whether its far end came within WARN of doing so, and
    per fragment whether it was lifted and whether it was slid too far before that. A reset shows as the episode
    clock going back; nothing the robot does to the objects clears what the band did."""

    def __init__(self):
        self.restart()
        self._t = None

    def restart(self):
        self.out = False
        self.warned = False
        self.slack = SLACK  # at the last step: band left beyond the bridge's near face
        self.lifted = dict.fromkeys(VALUES, False)
        self.dragged = dict.fromkeys(VALUES, False)
        self.held = dict.fromkeys(VALUES, False)  # at the last step, a gripper was within RELEASED of it
        self.band_held = False

    def update(self, objects):
        """Read one step. success() and score() both ask in the same step, so a step is read once."""
        t = clock()
        if t == self._t:
            return
        if self._t is None or t < self._t:
            self.restart()
        self._t = t
        hands = hand_points(arm_joints())
        band = objects["band"]
        self.slack = slack_left(band)
        if band_out(band):
            self.out = True
        elif self.slack < WARN:
            self.warned = True
        self.band_held = box_gap(hands, band, BAND_SIZE) < RELEASED
        for name, start in FRAGMENTS.items():
            fragment = objects[name]
            if fragment.low[2] - PAD_TOP >= LIFTED:
                self.lifted[name] = True
            elif not self.lifted[name] and np.linalg.norm(fragment.pos[:2] - start.pos[:2]) > DRAG:
                self.dragged[name] = True
            self.held[name] = box_gap(hands, fragment, FRAGMENT_SIZE) < RELEASED


TRACKER = Tracker()


def share(objects, name) -> float:
    """The part of a fragment's value it earns as it lies now: 1 on the tray's floor, OFF_CHANNEL_CREDIT anywhere
    else off the channel, 0 if it was not lifted out, was slid first, is still held or is not at rest."""
    TRACKER.update(objects)
    fragment = objects[name]
    if not TRACKER.lifted[name] or TRACKER.dragged[name] or TRACKER.held[name] or not fragment.resting():
        return 0.0
    if in_tray(fragment):
        return 1.0
    return OFF_CHANNEL_CREDIT if off_channel(fragment) else 0.0


def laid_back(objects) -> bool:
    """The band lies back where it started and is let go."""
    TRACKER.update(objects)
    return band_back(objects["band"]) and not TRACKER.band_held


# ---- hooks read by yam_arena (task_score) ----
def success(objects):
    """Score 1.0: both fragments at rest on the tray's floor and the band laid back, all let go, with the band
    never out from under the bridge nor within 1.5 cm of it."""
    TRACKER.update(objects)
    safe = not TRACKER.out and not TRACKER.warned
    return safe and laid_back(objects) and all(share(objects, name) == 1.0 for name in VALUES)


def score(objects):
    """0 once the band came out from under the bridge. Otherwise the fragments' values, each times its share,
    halved if the band ever came within WARN of coming out, and halved if it is not laid back at the end."""
    TRACKER.update(objects)
    if TRACKER.out:
        return 0.0
    earned = sum(value * share(objects, name) for name, value in VALUES.items())
    factor = (WARN_FACTOR if TRACKER.warned else 1.0) * (1.0 if laid_back(objects) else OPEN_FACTOR)
    return round(earned * factor, 4)


# ---- demo ----
GRIP_FROM_END = 0.04  # the band is held this far in from its near end
GRIP_X = BAND_X[0] + GRIP_FROM_END
SLIDE = FRAGMENT_SPECS["fragment_deep"][0]  # uncovers both fragments and leaves 3.5 cm of band beyond the bridge
NAIVE_SLIDE = 0.18  # naive variant: the band comes out from under the bridge
TRAVEL = TOP + 0.12  # carry height of the TCP
BAND_YAW = 0.0  # fingers close across the band
BAND_PADS_ABOVE = 0.006  # pad tips this far above the band's underside, so 6 mm above the strips
FRAGMENT_YAW = math.pi / 2  # fingers close along the channel: room to the strips on both sides
TRAY_YAW = 0.0  # over the tray the wrist cannot hold the grip it uses in the channel
APPROACH = 0.014  # the fingers come down this much wider than the object
GRIP_IN = 0.003  # finger target on an object: each pad this far into it (a full close twists the wrist)
PADS_ABOVE_UNDERSIDE = 0.004  # pad tips this far above a fragment's underside
LET_GO = 0.02  # the fingers open this much wider than the fragment to release it between the tray's walls
DROP = 0.005  # a fragment is let go with its underside this far above the tray's floor
ORDER = tuple(FRAGMENTS)  # the fragment uncovered first, first
TRAY_SPOTS = {name: (TRAY_CENTRE[0] + dx, TRAY_CENTRE[1]) for name, dx in zip(ORDER, (0.03, -0.03), strict=True)}


def _slide(x_from, x_to, where) -> list:
    z = STRIP_TOP + BAND_PADS_ABOVE + PAD_TIP_PAST_TCP
    hold = -(BAND_SIZE[1] / 2 - GRIP_IN)
    return [
        Waypoint(2.0, (x_from, Y0, TRAVEL), FINGER_OPEN, "band: over its near end", BAND_YAW),
        Waypoint(1.5, (x_from, Y0, z), FINGER_OPEN, "band: descend", BAND_YAW),
        Waypoint(0.4, (x_from, Y0, z), FINGER_OPEN, "band: settle", BAND_YAW),
        Waypoint(0.8, (x_from, Y0, z), hold, "band: close", BAND_YAW),
        Waypoint(0.5, (x_from, Y0, z), hold, "band: hold", BAND_YAW),
        Waypoint(6.0, (x_to, Y0, z), hold, f"band: slide {where}", BAND_YAW),
        Waypoint(0.5, (x_to, Y0, z), hold, "band: settle", BAND_YAW),
        Waypoint(0.6, (x_to, Y0, z), FINGER_OPEN, "band: release", BAND_YAW),
        Waypoint(1.0, (x_to, Y0, TRAVEL), FINGER_OPEN, "band: rise", BAND_YAW),
    ]


def _pick(name) -> list:
    x, y, _ = FRAGMENTS[name].pos
    width = FRAGMENT_SIZE[0]
    wide, grip, let_go = -(width + APPROACH) / 2, -(width / 2 - GRIP_IN), -(width + LET_GO) / 2
    tcp_up = PADS_ABOVE_UNDERSIDE + PAD_TIP_PAST_TCP  # the TCP above the fragment's underside
    grasp_z, place_z = PAD_TOP + tcp_up, TRAY_FLOOR_TOP + DROP + tcp_up
    tx, ty = TRAY_SPOTS[name]
    tag = name.split("_")[1][0]
    return [
        Waypoint(2.0, (x, y, TRAVEL), wide, f"{tag}: to the fragment", FRAGMENT_YAW),
        Waypoint(1.5, (x, y, grasp_z), wide, f"{tag}: descend", FRAGMENT_YAW),
        Waypoint(0.4, (x, y, grasp_z), wide, f"{tag}: settle", FRAGMENT_YAW),
        Waypoint(0.8, (x, y, grasp_z), grip, f"{tag}: close", FRAGMENT_YAW),
        Waypoint(0.5, (x, y, grasp_z), grip, f"{tag}: hold", FRAGMENT_YAW),
        Waypoint(1.2, (x, y, TRAVEL), grip, f"{tag}: lift", FRAGMENT_YAW),
        Waypoint(2.2, (tx, ty, TRAVEL), grip, f"{tag}: carry", TRAY_YAW),
        Waypoint(1.5, (tx, ty, place_z), grip, f"{tag}: lower", TRAY_YAW),
        Waypoint(0.4, (tx, ty, place_z), grip, f"{tag}: settle", TRAY_YAW),
        Waypoint(0.6, (tx, ty, place_z), let_go, f"{tag}: release", TRAY_YAW),
        Waypoint(1.0, (tx, ty, TRAVEL), let_go, f"{tag}: rise", TRAY_YAW),
    ]


def demo(variant="scripted"):
    """Right arm: pinch the band across its width 4 cm in from its near end and slide it 13.5 cm toward the robot,
    which uncovers both fragments and leaves 3.5 cm of band beyond the bridge. Take each fragment with the fingers
    closing along the channel, lift it straight up to 12 cm, carry it to its own place in the tray while turning
    the grip a quarter turn, lower it to 5 mm above the floor and let go. Then pinch the band's near end again and
    slide it back to its mark.

    variant="naive" is for the offline check only: the first slide is 18 cm, past the end of the slack, so the
    band comes out from under the bridge."""
    slide = NAIVE_SLIDE if variant == "naive" else SLIDE
    plan = _slide(GRIP_X, GRIP_X - slide, "toward the robot")
    for name in ORDER:
        plan += _pick(name)
    plan += _slide(GRIP_X - slide, GRIP_X, "back to its mark")
    return {"right": plan}
