"""On the station table: a grey wall panel, 26.5 cm long, 16 cm high and 6 mm thick, stands between the arms, turned
45 degrees: it runs from near the right arm to further away on the left, and its near face looks toward the left
arm. Three square windows are cut in it, all with their lower edge 4 cm above the table. From the end nearest the
robot: 8 cm (large), 3.5 cm (small), 5 cm (medium). A brown rod, 26 cm long with a 2 x 2 cm section, lies on the
table on the left arm's side, square to the wall, pointing at the strip of wall between the large and the small
window. A blue stand, 14 x 8 x 3 cm, is fixed to the table behind the wall, on the right arm's side. Neither
gripper fits through a window, so the rod has to change hands through the wall.

Score (task_score), from every step of the trial and the final state:
- 0 if the rod ever came within 2 mm of the wall (a touch), or its centre ever moved down faster than 0.5 m/s (a
  drop).
- 0 if the rod's centre is not behind the wall at the end, or last went behind it any other way than through a
  window (over the top, round an end).
- Otherwise the value of the window it went through, if the rod rests on the stand: large 0.4, medium 0.7,
  small 1.0. A quarter of that value if it is anywhere else behind the wall.

Run it with -E object_jitter=0: the scoring starts over whenever the rod lies at its start pose.

Run it:  inspect-robots run --embodiment yam_arena -E task=rod_through_wall.py -E object_jitter=0 \\
           --instruction "Pass the rod through the wall to the blue stand on the other side, using the smallest \\
window you can, without touching the wall or dropping the rod." --scorer task_score ...
"""

import math

import numpy as np

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
# success must hold this long: longer than release and retreat, so the rod must stay on the stand after the
# gripper leaves (the scorer cannot see the gripper)
SUCCESS_HOLD_S = 3.0
TOP = TABLE_TOP[SCENE]

# ---- wall (m). Wall frame: u along the wall from its centre (away from the robot), v out of its near face
# (toward the left arm), heights above the table ----
T = 0.006  # thickness (MDF or foam board)
WALL_CENTRE = np.array([0.22, -0.085])
WALL_YAW = math.radians(45)
WALL_H = 0.16
SILL = 0.04  # lower edge of every window
PILLAR = 0.025  # wall left standing between the windows and at both ends
WINDOW_ORDER = (("large", 0.08, 0.4), ("small", 0.035, 1.0), ("medium", 0.05, 0.7))  # name, side, value; along u
WALL_LENGTH = sum(side for _, side, _ in WINDOW_ORDER) + (len(WINDOW_ORDER) + 1) * PILLAR
WALL_COLOR = (0.55, 0.55, 0.57)

# ---- rod and stand, in the wall frame ----
ROD_SIZE = (0.02, 0.26, 0.02)  # long axis along v
ROD_SIDE, ROD_LENGTH = ROD_SIZE[0], ROD_SIZE[1]
ROD_GAP = 0.05  # from the rod's far end to the wall's mid-plane at the start
STAND_SIZE = (0.14, 0.08, 0.03)  # along u, along v, height
STAND_TOP = TOP + STAND_SIZE[2]

# ---- thresholds ----
CONTACT = 0.002  # the rod touched the wall when any point of its surface came this close to it
SAMPLE = 0.0025  # spacing of the points checked on the rod's surface
FALL_SPEED = 0.5  # m/s: a rod whose centre moves down faster than this was dropped (a free fall of about 1.3 cm)
ON_STAND_TOLERANCE = 0.005  # resting on the stand: lowest point within this of the stand's top
FAR_SIDE_CREDIT = 0.25  # share of the window's value for a rod behind the wall but not on the stand
AT_START, AT_START_DEG = 0.01, 5.0  # the rod lies at its start pose: the tracker starts again

_AXES = np.array(
    [[math.cos(WALL_YAW), math.sin(WALL_YAW)], [-math.sin(WALL_YAW), math.cos(WALL_YAW)]]
)  # rows: u and v in the world


def to_wall(xy) -> np.ndarray:
    """World xy (one point or rows of points) in the wall frame (u, v)."""
    return (np.asarray(xy) - WALL_CENTRE) @ _AXES.T


def to_world(u, v) -> tuple[float, float]:
    x, y = WALL_CENTRE + np.array([u, v]) @ _AXES
    return float(x), float(y)


def _windows() -> dict:
    out, u = {}, -WALL_LENGTH / 2
    for name, side, value in WINDOW_ORDER:
        out[name] = dict(u=(u + PILLAR, u + PILLAR + side), side=side, value=value)
        u += PILLAR + side
    return out


WINDOWS = _windows()
_TALLEST = max(w["side"] for w in WINDOWS.values())


def _wall_parts() -> list:
    """(name, u range, height range) of the boxes that make up the wall around its windows."""
    ends = (-WALL_LENGTH / 2, WALL_LENGTH / 2)
    edges = [ends[0], *(e for w in WINDOWS.values() for e in w["u"]), ends[1]]
    parts = [("wall_sill", ends, (0.0, SILL)), ("wall_lintel", ends, (SILL + _TALLEST, WALL_H))]
    parts += [
        (f"wall_pillar_{i}", (edges[2 * i], edges[2 * i + 1]), (SILL, SILL + _TALLEST))
        for i in range(len(WINDOWS) + 1)
    ]
    parts += [
        (f"wall_over_{name}", w["u"], (SILL + w["side"], SILL + _TALLEST))
        for name, w in WINDOWS.items()
        if w["side"] < _TALLEST
    ]
    return parts


WALL_PARTS = _wall_parts()
_LOW = np.array([(u[0], -T / 2, z[0]) for _, u, z in WALL_PARTS])  # each part's corners in (u, v, height)
_HIGH = np.array([(u[1], T / 2, z[1]) for _, u, z in WALL_PARTS])

SMALL_U = sum(WINDOWS["small"]["u"]) / 2  # centre line of the small window
# (u, v) of the rod's centre: it points at the pillar between the large and the small window
ROD_START = ((WINDOWS["large"]["u"][1] + WINDOWS["small"]["u"][0]) / 2, ROD_GAP + ROD_LENGTH / 2)
STAND_UV = (SMALL_U, -0.16)  # (u, v) of the stand's centre

WALL = [
    Box(
        name,
        pos=(*to_world(sum(u) / 2, 0.0), TOP + sum(z) / 2),
        size=(u[1] - u[0], T, z[1] - z[0]),
        color=WALL_COLOR,
        yaw=WALL_YAW,
    )
    for name, u, z in WALL_PARTS
]
ROD = Box(
    "rod",
    pos=(*to_world(*ROD_START), TOP + ROD_SIDE / 2),
    size=ROD_SIZE,
    mass=0.05,
    color=(0.35, 0.22, 0.12),
    yaw=WALL_YAW,
)
STAND = Box(
    "stand",
    pos=(*to_world(*STAND_UV), TOP + STAND_SIZE[2] / 2),
    size=STAND_SIZE,
    color=(0.20, 0.45, 0.85),
    yaw=WALL_YAW,
)
OBJECTS = [*WALL, STAND, ROD]


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


def _surface() -> np.ndarray:
    """Points every SAMPLE over the rod's six faces, in the rod's own frame."""
    half = np.array(ROD_SIZE) / 2
    lines = [np.linspace(-h, h, round(2 * h / SAMPLE) + 1) for h in half]
    faces = []
    for axis in range(3):
        a, b = (i for i in range(3) if i != axis)
        grid = np.stack(np.meshgrid(lines[a], lines[b], indexing="ij"), -1).reshape(-1, 2)
        for sign in (-1, 1):
            face = np.empty((len(grid), 3))
            face[:, axis], face[:, a], face[:, b] = sign * half[axis], grid[:, 0], grid[:, 1]
            faces.append(face)
    return np.vstack(faces)


ROD_SURFACE = _surface()
ROD_CORNERS = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.array(ROD_SIZE) / 2


def rod_points(objects, local=ROD_SURFACE) -> np.ndarray:
    """Points of the rod (given in its own frame) as rows of (u, v, height above the table)."""
    rod = objects["rod"]
    world = rod.pos + local @ _rotation(rod.quat_xyzw).T
    return np.c_[to_wall(world[:, :2]), world[:, 2] - TOP]


def rod_corners(objects) -> np.ndarray:
    return rod_points(objects, ROD_CORNERS)


# ---- task quantities ----
def wall_clearance(objects) -> float:
    """Smallest distance (m) from the rod's surface to the wall; 0 where the rod is inside it."""
    points = rod_points(objects)
    outside = np.maximum(np.maximum(_LOW[:, None] - points, points - _HIGH[:, None]), 0.0)
    return float(np.linalg.norm(outside, axis=2).min())


def window_of(objects) -> str | None:
    """The window that holds the rod's whole cross-section where it passes the wall, or None: the rod is beside
    the wall, above it, or not at the wall."""
    points = rod_points(objects)
    cut = points[np.abs(points[:, 1]) <= T / 2]
    if not len(cut):
        return None
    (u0, h0), (u1, h1) = cut[:, [0, 2]].min(0), cut[:, [0, 2]].max(0)
    for name, w in WINDOWS.items():
        if w["u"][0] < u0 and u1 < w["u"][1] and SILL < h0 and h1 < SILL + w["side"]:
            return name
    return None


def behind_wall(objects) -> bool:
    """The rod's centre is past the wall's mid-plane, on the stand's side."""
    return bool(to_wall(objects["rod"].pos[:2])[1] < 0.0)


def on_stand(objects) -> bool:
    """At rest on the stand: centre over the stand's top, lowest point on it."""
    rod = objects["rod"]
    du, dv = np.abs(to_wall(rod.pos[:2]) - STAND_UV)
    return bool(
        du < STAND_SIZE[0] / 2
        and dv < STAND_SIZE[1] / 2
        and abs(rod.low[2] - STAND_TOP) <= ON_STAND_TOLERANCE
        and rod.resting()
    )


def at_start(objects) -> bool:
    rod = objects["rod"]
    along = abs(_rotation(rod.quat_xyzw)[:2, 1] @ _AXES[1])  # the rod's long axis against its start direction
    turned = math.degrees(math.acos(min(along, 1.0)))
    return bool(np.linalg.norm(rod.pos - np.array(ROD.pos)) < AT_START and turned < AT_START_DEG)


# ---- what the final state does not show ----
class Tracker:
    """Reads the rod's pose after each control step and keeps: a touch of the wall, a drop, and the window of the
    rod's latest pass to the far side. The hooks get no reset signal, so it starts again whenever the rod lies at
    its start pose (which is true after each reset)."""

    def __init__(self):
        self.restart()
        self._last = None

    def restart(self):
        self.touched = False
        self.fell = False
        self.window = None  # of the latest pass to the far side; None: no pass yet, or not through a window
        self.near = True  # the rod's centre was on the near side at the last step

    def update(self, objects):
        """Read one step. success(), score() and distance() get the same dict each step, so a step is read once."""
        if objects is self._last:
            return
        self._last = objects
        if at_start(objects):
            self.restart()
            return
        self.touched |= wall_clearance(objects) < CONTACT
        self.fell |= bool(objects["rod"].lin_vel[2] < -FALL_SPEED)
        near = not behind_wall(objects)
        if self.near and not near:
            self.window = window_of(objects)
        self.near = near


TRACKER = Tracker()


def _score(objects) -> float:
    TRACKER.update(objects)
    if TRACKER.touched or TRACKER.fell or TRACKER.window is None or not behind_wall(objects):
        return 0.0
    value = WINDOWS[TRACKER.window]["value"]
    return value if on_stand(objects) else FAR_SIDE_CREDIT * value


# ---- hooks read by yam_arena (task_score) ----
def success(objects):
    """Score 1.0: the rod rests on the stand after a pass through the small window, with no touch and no drop."""
    return _score(objects) == 1.0


def score(objects):
    """0 after a touch or a drop, or if the rod's centre is not behind the wall after a pass through a window.
    Otherwise the window's value with the rod at rest on the stand, a quarter of it anywhere else."""
    return _score(objects)


def distance(objects):
    """Horizontal distance (m) from the rod's centre to the stand's centre."""
    return float(np.linalg.norm(to_wall(objects["rod"].pos[:2]) - STAND_UV))


# ---- demo ----
GRIP_YAW = WALL_YAW - math.pi / 2  # fingers close along the wall, across the rod
GRIP_FROM_END = 0.04  # left grip, from the rod's near end
HANDOFF_V = 0.085  # both TCPs stay this far from the wall's mid-plane, so the wrists clear each other above it


def demo():
    """Left arm: grip the rod 4 cm from its near end, raise it to the middle of the small window, push it through
    until the gripper is 8.5 cm from the wall, hold it while the right arm takes the far end, let go and back off.
    Right arm: wait above the far side, grip the rod 8.5 cm behind the wall, draw it out through the window until
    its centre is over the stand, lower it to 3 mm above the stand and let go."""
    u0, v0 = ROD_START
    tcp_to_tip = ROD_LENGTH - GRIP_FROM_END  # left TCP to the rod's far end
    grasp_z = TOP + PAD_TIP_PAST_TCP + ROD_SIDE / 2  # pad tips at the rod's mid-height
    window_z = grasp_z + SILL + (WINDOWS["small"]["side"] - ROD_SIDE) / 2  # rod in the middle of the small window
    place_z = grasp_z + STAND_SIZE[2] + 0.003  # rod 3 mm above the stand
    travel_z = TOP + 0.14
    pick_v = v0 + ROD_LENGTH / 2 - GRIP_FROM_END
    centre_past_take = 2 * HANDOFF_V - tcp_to_tip + ROD_LENGTH / 2  # from the right TCP to the rod's centre
    place_v = STAND_UV[1] - centre_past_take

    def at(u, v, z):
        return (*to_world(u, v), z)

    left = [
        Waypoint(1.5, at(u0, pick_v, travel_z), FINGER_OPEN, "to the rod", GRIP_YAW),
        Waypoint(1.5, at(u0, pick_v, grasp_z), FINGER_OPEN, "descend", GRIP_YAW),
        Waypoint(0.5, at(u0, pick_v, grasp_z), FINGER_OPEN, "settle", GRIP_YAW),
        Waypoint(1.0, at(u0, pick_v, grasp_z), FINGER_CLOSED, "close", GRIP_YAW),
        Waypoint(0.5, at(u0, pick_v, grasp_z), FINGER_CLOSED, "hold", GRIP_YAW),
        Waypoint(2.0, at(SMALL_U, pick_v, window_z), FINGER_CLOSED, "line up", GRIP_YAW),
        Waypoint(0.5, at(SMALL_U, pick_v, window_z), FINGER_CLOSED, "settle", GRIP_YAW),
        Waypoint(4.0, at(SMALL_U, HANDOFF_V, window_z), FINGER_CLOSED, "push through", GRIP_YAW),
    ]
    pushed = sum(w.duration for w in left)
    right = [
        Waypoint(2.0, at(SMALL_U, -HANDOFF_V, travel_z), FINGER_OPEN, "above the far side", GRIP_YAW),
        Waypoint(pushed - 2.0, at(SMALL_U, -HANDOFF_V, travel_z), FINGER_OPEN, "wait for the rod", GRIP_YAW),
        Waypoint(1.5, at(SMALL_U, -HANDOFF_V, window_z), FINGER_OPEN, "descend", GRIP_YAW),
        Waypoint(0.5, at(SMALL_U, -HANDOFF_V, window_z), FINGER_OPEN, "settle", GRIP_YAW),
        Waypoint(1.0, at(SMALL_U, -HANDOFF_V, window_z), FINGER_CLOSED, "close", GRIP_YAW),
        Waypoint(0.5, at(SMALL_U, -HANDOFF_V, window_z), FINGER_CLOSED, "hold", GRIP_YAW),
    ]
    taken = sum(w.duration for w in right)
    left += [
        Waypoint(taken - pushed, at(SMALL_U, HANDOFF_V, window_z), FINGER_CLOSED, "hold for the right arm", GRIP_YAW),
        Waypoint(1.0, at(SMALL_U, HANDOFF_V, window_z), FINGER_OPEN, "release", GRIP_YAW),
        Waypoint(1.5, at(SMALL_U, pick_v, window_z), FINGER_OPEN, "back off", GRIP_YAW),
        Waypoint(1.5, at(SMALL_U, pick_v, travel_z), FINGER_OPEN, "rise", GRIP_YAW),
    ]
    right += [
        Waypoint(1.5, at(SMALL_U, -HANDOFF_V, window_z), FINGER_CLOSED, "wait for the left arm", GRIP_YAW),
        Waypoint(4.0, at(SMALL_U, place_v, window_z), FINGER_CLOSED, "draw out", GRIP_YAW),
        Waypoint(0.5, at(SMALL_U, place_v, window_z), FINGER_CLOSED, "settle", GRIP_YAW),
        Waypoint(1.5, at(SMALL_U, place_v, place_z), FINGER_CLOSED, "lower", GRIP_YAW),
        Waypoint(0.5, at(SMALL_U, place_v, place_z), FINGER_CLOSED, "settle", GRIP_YAW),
        Waypoint(0.8, at(SMALL_U, place_v, place_z), FINGER_OPEN, "release", GRIP_YAW),
        Waypoint(1.2, at(SMALL_U, place_v, travel_z), FINGER_OPEN, "retreat", GRIP_YAW),
        Waypoint(1.5, at(SMALL_U, place_v, travel_z), FINGER_OPEN, "wait", GRIP_YAW),
    ]
    return {"left": left, "right": right}
