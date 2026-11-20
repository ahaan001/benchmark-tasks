"""A cafe counter. Far centre, a beech board with two plain croissants. Beside it, 10 cm away, a second beech board
with two almond croissants (white icing sugar, sliced almonds) and a pair of black-handled tongs lying with open jaws
round the nearer one, handle toward the robot. Near centre, an empty white plate. Near the robot on the other side, a
purple allergen board with a pair of purple-handled tongs lying on it. Tongs lie on their side: a gripper holds a pair
by squeezing its two arms together at mid-handle, and squeezing harder closes the jaws.

Two mirrored layouts: A (the screenshot) has the almond board on the robot's left and the purple board on its right;
B (environment variable TASK_LAYOUT=B) swaps left and right. Use them on alternate trials.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as geo  # noqa: E402
import parts  # noqa: E402
from yam_arena.demo import FINGER_OPEN, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
LAYOUT = os.environ.get("TASK_LAYOUT", "A").strip().upper() or "A"
if LAYOUT not in ("A", "B"):
    raise ValueError(f"TASK_LAYOUT must be A or B, got {LAYOUT!r}")
SIDE = 1.0 if LAYOUT == "A" else -1.0  # layout B is layout A with left and right swapped

BEECH, PURPLE, STEEL, BLACK = (0.80, 0.64, 0.42), (0.46, 0.22, 0.62), (0.76, 0.77, 0.79), (0.07, 0.07, 0.08)
CRUST, SUGAR, ALMOND, CHINA = (0.80, 0.52, 0.20), (0.97, 0.95, 0.91), (0.87, 0.73, 0.52), (0.95, 0.95, 0.93)


def _xy(x: float, y: float) -> tuple[float, float]:
    """A layout-A position in the layout being run."""
    return (x, SIDE * y)


# --- boards (fixed) ---------------------------------------------------------------------------------------------------
BOARD = (0.300, 0.150, 0.016)  # beech chopping board, long side front to back
PLAIN_BOARD_XY = _xy(0.500, -0.020)
ALMOND_BOARD_XY = _xy(0.500, 0.230)
PURPLE_BOARD = (0.305, 0.229, 0.0095)  # purple polyethylene allergen board, 12 x 9 x 3/8 in
PURPLE_BOARD_XY = _xy(0.2225, -0.280)


def _board(name, xy, size, color, friction):
    return parts.Part(name, pos=(xy[0], xy[1], TOP), friction=friction,
                      shapes=[parts.box(size, at=(0.0, 0.0, size[2] / 2), color=color)])


BOARD_PLAIN = _board("board_plain", PLAIN_BOARD_XY, BOARD, BEECH, (0.6, 0.5))
BOARD_ALMOND = _board("board_almond", ALMOND_BOARD_XY, BOARD, BEECH, (0.6, 0.5))
BOARD_PURPLE = _board("board_purple", PURPLE_BOARD_XY, PURPLE_BOARD, PURPLE, (0.5, 0.4))

# --- croissants (loose) -----------------------------------------------------------------------------------------------
CROISSANT = geo.Croissant(length=0.140, width=0.064, height=0.045)
CROISSANT_MASS = 0.065
ROW = (0.425, 0.575)  # the two croissants on a board, front to back
LIFT = 0.001  # loose things start 1 mm above what they rest on


def _croissant(name, xy, z, almond):
    pts, faces, normals = CROISSANT.mesh()
    shapes = [parts.hull(p, geo.hull_faces(p)) for p in CROISSANT.pieces()]
    shapes.append(parts.mesh(pts, faces, normals, color=CRUST))
    if almond:
        sugar, flakes = CROISSANT.topping()
        shapes.append(parts.mesh(*sugar, color=SUGAR))
        shapes += [parts.mesh(*f, color=ALMOND) for f in flakes]
    return parts.Part(name, pos=(xy[0], xy[1], z), mass=CROISSANT_MASS, friction=(0.9, 0.8), shapes=shapes,
                      yaw=0.0 if SIDE > 0 else math.pi)


ON_BOARD = TOP + BOARD[2] + LIFT
PLAIN = ("plain_1", "plain_2")
ALMONDS = ("almond_1", "almond_2")
CROISSANTS = [_croissant(n, (x, PLAIN_BOARD_XY[1]), ON_BOARD, False) for n, x in zip(PLAIN, ROW)]
CROISSANTS += [_croissant(n, (x, ALMOND_BOARD_XY[1]), ON_BOARD, True) for n, x in zip(ALMONDS, ROW)]

# --- plate (loose) ----------------------------------------------------------------------------------------------------
PLATE_XY = _xy(0.220, -0.020)
PLATE_R, PLATE_H, PLATE_WELL, PLATE_T = 0.108, 0.019, 0.070, 0.0035  # 8.5 in glass lunch plate
PLATE_MASS = 0.244
RIM_SEGMENTS = 16


def _plate():
    th = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([(PLATE_WELL + 0.004) * np.cos(th), (PLATE_WELL + 0.004) * np.sin(th)], axis=1)
    floor = np.vstack([np.c_[ring, np.zeros(24)], np.c_[ring, np.full(24, PLATE_T)]])
    shapes = [parts.hull(floor, geo.hull_faces(floor))]
    run, rise = PLATE_R - PLATE_WELL, PLATE_H - PLATE_T
    slope, length = math.atan2(rise, run), math.hypot(run, rise)
    width = 2 * PLATE_R * math.tan(math.pi / RIM_SEGMENTS) + 0.002
    for i in range(RIM_SEGMENTS):
        a = 2 * math.pi * i / RIM_SEGMENTS
        r = (PLATE_WELL + PLATE_R) / 2
        rot = parts._rot((0, 0, 1), a) @ parts._rot((0, 1, 0), -slope)
        up = rot @ np.array([0.0, 0.0, 1.0])
        centre = np.array([r * math.cos(a), r * math.sin(a), PLATE_T + rise / 2]) - up * (PLATE_T / 2)
        shapes.append(parts.box((length, width, PLATE_T), at=centre, visible=False,
                                rot=tuple(parts._matrix_to_quat(rot))))
    t = PLATE_T
    outline = [(0.0, 0.0), (PLATE_WELL, 0.0), (PLATE_R, PLATE_H - t), (PLATE_R, PLATE_H), (PLATE_WELL - 0.004, t),
               (0.0, t)]
    shapes.append(parts.mesh(*geo.lathe(outline), color=CHINA))
    return parts.Part("plate", pos=(PLATE_XY[0], PLATE_XY[1], TOP + LIFT), mass=PLATE_MASS, friction=(0.5, 0.4),
                      shapes=shapes)


PLATE = _plate()

# --- tongs (loose, jointed) -------------------------------------------------------------------------------------------
# Part frame: origin under the bent end, x along the tongs to the jaws, z up; the tongs lie on their side, so the arms
# swing about the vertical through the bend and a gripper pointing down squeezes them across y. The shapes are written
# at the free opening (joint 0); a one-piece tong has no stop there, the bend is the spring.
TONG_LEN = 0.305  # 12 in
TONG_OPEN = 0.108  # across the open jaws, outside (4.25 in)
STRIP_H, STRIP_T = 0.020, 0.004  # each arm: a strip on edge (0.9 mm steel; 4 mm here so contacts hold)
STRIP_X = (0.008, 0.232)
JAW_H, JAW_T = 0.036, 0.004  # scalloped jaw, 1.43 in tall
JAW_X, JAW_MID, JAW_WING, JAW_CUP = 0.2675, 0.030, 0.024, math.radians(14.0)
SLEEVE_X, SLEEVE_T, SLEEVE_H = (0.020, 0.172), 0.0065, 0.022  # coated handle
SLEEVE_GRIP = (2.0, 1.8)  # the coating's friction, static and dynamic
GRIP_X = 0.128  # where a gripper squeezes the handle, from the bend: the tongs balance there
HOLD_ANGLE = 0.022  # rad each arm is squeezed in to carry the tongs
FLAT_LEN, FLAT_T = 0.040, 0.004  # a hidden flat on each handle at GRIP_X, square to a pad at HOLD_ANGLE
PIVOT = (0.006, 0.004)  # each arm's pivot: x, and y either side of the centre line
BEND_R = 0.005
TONG_SPRING = 6.0  # N m/rad each arm (an estimate: about 7 N at mid-handle closes them)
TONG_DAMPING = 0.10
TONG_SPREAD = 0.06  # rad an arm can be pushed open past its free position
ARM_MASS, BEND_MASS = 0.062, 0.012
JAW_DROP = (JAW_H - STRIP_H) / 2  # a jaw's lower edge below its strip's
JAW_HEEL = JAW_X - JAW_MID / 2 - JAW_WING * math.cos(JAW_CUP)  # where a jaw's lower edge begins: it rests there


def _tilt(rise: float) -> float:
    """How far the jaw end is raised (rad) when the tongs lie with the bend on a surface and the jaws' heels
    ``rise`` above it."""
    reach = JAW_HEEL - PIVOT[0]
    lo, hi = 0.0, 0.5
    for _ in range(50):
        t = (lo + hi) / 2
        lo, hi = (t, hi) if reach * math.sin(t) - JAW_DROP * math.cos(t) < rise else (lo, t)
    return (lo + hi) / 2


def _arm_frame(side: int, angle: float, tilt: float) -> np.ndarray:
    """An arm's rotation: turned ``angle`` out from the centre line on its side, its jaw end raised by ``tilt``."""
    return parts._rot((0, 0, 1), side * angle) @ parts._rot((0, 1, 0), -tilt)


def _place(side, angle, tilt, size, x, y_in, z) -> tuple:
    """(size, centre, rotation) of a box on an arm: its middle ``x`` along the arm from the bend, its inner face
    ``y_in`` off the arm's own line toward the outside, its middle ``z`` above the strip's lower edge."""
    r = _arm_frame(side, angle, tilt)
    pivot = np.array([PIVOT[0], side * PIVOT[1], 0.0])
    local = np.array([x - PIVOT[0], side * (y_in + size[1] / 2), z])
    return np.asarray(size, float), pivot + r @ local, r


def _jaw_boxes(side, angle, tilt):
    """The jaw's three plates: the two wings turned in by JAW_CUP so the jaw cups what it holds."""
    out = []
    reach = JAW_MID / 2 + JAW_WING / 2 * math.cos(JAW_CUP)
    for dx, cup in ((-reach, JAW_CUP), (0.0, 0.0), (reach, -JAW_CUP)):
        size, centre, r = _place(side, angle, tilt, (JAW_WING if cup else JAW_MID, JAW_T, JAW_H), JAW_X + dx,
                                 -abs(math.sin(cup)) * JAW_WING / 2, STRIP_H / 2)
        out.append((size, centre, r @ parts._rot((0, 0, 1), side * cup)))
    return out


def _free_angle() -> float:
    """The half-angle between the arms at rest that makes the open jaws TONG_OPEN across, outside."""
    signs = np.array([[sx, sy, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    lo, hi = 0.05, 0.4
    for _ in range(40):
        a = (lo + hi) / 2
        widest = max((centre + (signs * size / 2) @ r.T)[:, 1].max() for size, centre, r in _jaw_boxes(1, a, 0.0))
        lo, hi = (a, hi) if widest < TONG_OPEN / 2 else (lo, a)
    return (lo + hi) / 2


TONG_FREE = _free_angle()
TONG_SHUT = TONG_FREE - 0.012  # joint travel from free to jaws together


def _flat(side, angle, tilt):
    """(size, centre, rotation) of the hidden flat on an arm's handle. The coating beds onto a gripper's pad; a rigid
    handle would meet it along one edge. The flat stands in for that: its outer face runs along the tongs when the arm
    is HOLD_ANGLE in from free, flush with the coating at its jaw end."""
    r = _arm_frame(side, angle, tilt)
    pivot = np.array([PIVOT[0], side * PIVOT[1], 0.0])
    turn = parts._rot((0, 0, 1), -side * (TONG_FREE - HOLD_ANGLE))
    front = np.array([GRIP_X + FLAT_LEN / 2 - PIVOT[0], side * (SLEEVE_T + STRIP_T) / 2, STRIP_H / 2])
    centre = front + turn @ np.array([-FLAT_LEN / 2, -side * FLAT_T / 2, 0.0])
    return np.array([FLAT_LEN, FLAT_T, SLEEVE_H]), pivot + r @ centre, r @ turn


def _tongs(name: str, xy, z: float, rise: float, handle):
    """A pair of tongs with the bent end at ``xy`` on a surface at ``z`` and the jaws' heels ``rise`` above it
    (0: lying on one surface; a board's thickness: jaws on the board, bent end on the table)."""
    tilt = _tilt(rise)
    joints = []
    for side, joint in ((1, "arm_l"), (-1, "arm_r")):
        shapes = []
        size, centre, r = _place(side, TONG_FREE, tilt, (STRIP_X[1] - STRIP_X[0], STRIP_T, STRIP_H),
                                 (STRIP_X[0] + STRIP_X[1]) / 2, 0.0, STRIP_H / 2)
        shapes.append(parts.box(size, at=centre, color=STEEL, rot=tuple(parts._matrix_to_quat(r))))
        for size, centre, r in _jaw_boxes(side, TONG_FREE, tilt):
            shapes.append(parts.box(size, at=centre, color=STEEL, rot=tuple(parts._matrix_to_quat(r))))
        size, centre, r = _place(side, TONG_FREE, tilt, (SLEEVE_X[1] - SLEEVE_X[0], SLEEVE_T, SLEEVE_H),
                                 (SLEEVE_X[0] + SLEEVE_X[1]) / 2, -(SLEEVE_T - STRIP_T) / 2, STRIP_H / 2)
        shapes.append(parts.box(size, at=centre, color=handle, rot=tuple(parts._matrix_to_quat(r)),
                                friction=SLEEVE_GRIP))
        size, centre, r = _flat(side, TONG_FREE, tilt)
        shapes.append(parts.box(size, at=centre, color=handle, rot=tuple(parts._matrix_to_quat(r)),
                                friction=SLEEVE_GRIP, visible=False))
        joints.append(parts.Hinge(joint, shapes=shapes, pivot=(PIVOT[0], side * PIVOT[1], STRIP_H / 2),
                                  axis=(0.0, 0.0, float(side)), lower=-TONG_SHUT, upper=TONG_SPREAD,
                                  mass=ARM_MASS, spring=TONG_SPRING, damping=TONG_DAMPING))
    bend = [parts.cylinder(BEND_R, STRIP_H, at=(BEND_R, 0.0, STRIP_H / 2), color=STEEL, axis="z")]
    return parts.Part(name, pos=(xy[0], xy[1], z + LIFT), mass=BEND_MASS, friction=(0.6, 0.5), shapes=bend,
                      joints=joints)


BLACK_BEND_XY = (ROW[0] - JAW_X, ALMOND_BOARD_XY[1])  # jaws round the nearer almond croissant
PURPLE_BEND_XY = (0.075, PURPLE_BOARD_XY[1])
BLACK_TONGS = _tongs("black_tongs", BLACK_BEND_XY, TOP, BOARD[2], BLACK)
PURPLE_TONGS = _tongs("purple_tongs", PURPLE_BEND_XY, TOP + PURPLE_BOARD[2], 0.0, PURPLE)

OBJECTS = [BOARD_PLAIN, BOARD_ALMOND, BOARD_PURPLE, PLATE, *CROISSANTS, BLACK_TONGS, PURPLE_TONGS]
SPECS = {o.name: o for o in OBJECTS}

# --- scoring ----------------------------------------------------------------------------------------------------------
TOUCH = 0.002  # touching: surfaces within 2 mm
CLEAR = 0.030  # let go: every part of both grippers at least 3 cm away
CARRY_STEPS = 5  # held in the purple jaws, touching nothing else, for 0.25 s: carried
OFF_TABLE = 0.003  # a croissant's lowest point this far above the table top: not lying on the table
REST_STEPS, REST_SHIFT, REST_TURN = 20, 0.002, math.radians(2.0)  # nothing moved over the last second
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"

NUT = (*ALMONDS, "board_almond", "black_tongs")  # what carries nut from the start
LOOSE = ("plate", *PLAIN, *ALMONDS, "black_tongs", "purple_tongs")
ARMS = ("left_arm", "right_arm")
ARM_LINKS = ("link3", "link4", "link5", "gripper", "tip_left", "tip_right")
HAND_LINKS = ("gripper", "tip_left", "tip_right")

_SAMPLES: dict = {}


def _local(name: str):
    """Each colliding shape of a part, once: (link, shape, surface points and planes in the shape's frame)."""
    if name not in _SAMPLES:
        out = []
        for link, _, shapes in SPECS[name].links:
            for s in shapes:
                if not s.collide:
                    continue
                if s.kind == "box":
                    out.append((link, s, geo.box_points(s.size), None))
                    continue
                if s.kind == "cylinder":
                    th = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
                    ring = np.stack([s.size[0] * np.cos(th), s.size[0] * np.sin(th)], axis=1)
                    pts = np.vstack([np.c_[ring, np.full(12, -s.size[1] / 2)], np.c_[ring, np.full(12, s.size[1] / 2)]])
                else:
                    pts = np.asarray(s.points, float)
                planes = geo.hull_planes(pts)
                out.append((link, s, geo.hull_points(pts, geo.hull_faces(pts, planes)), planes))
        _SAMPLES[name] = out
    return _SAMPLES[name]


def pieces(name: str, state) -> list[geo.Piece]:
    """A part's colliders in the world, from its state. A jaw's plates are tagged "<arm>:jaw", its middle plate
    "<arm>:jaw:mid"."""
    spec = SPECS[name]
    poses = spec.state_poses(state)
    origins = {link: origin for link, origin, _ in spec.links}
    out = []
    for link, s, pts, planes in _local(name):
        p, r = poses[link]
        at, rot = p + r @ (np.asarray(s.at) - origins[link]), r @ s.rotation
        tag = link
        if s.kind == "box" and abs(s.size[2] - JAW_H) < 1e-9:
            tag = f"{link}:jaw:mid" if abs(s.size[0] - JAW_MID) < 1e-9 else f"{link}:jaw"
        if planes is None:
            out.append(geo.world_box(at, rot, s.size, pts, tag))
        else:
            out.append(geo.world_hull(at, rot, pts, planes, tag))
    return out


def arm_pieces(link_poses: dict[str, np.ndarray], links=ARM_LINKS) -> list[geo.Piece]:
    """An arm's collision hulls in the world, from its link poses (4x4)."""
    from yam_arena import analysis

    out = []
    for link, hull in analysis.world_hulls({k: v for k, v in link_poses.items() if k in links}):
        v = hull.vertices
        out.append(geo.Piece("hull", v, v.min(axis=0), v.max(axis=0), planes=hull.planes, tag=link))
    return out


def _over_purple_board(piece: geo.Piece) -> bool:
    """The middle of a jaw's middle plate lies over the purple board."""
    half = np.asarray(PURPLE_BOARD[:2]) / 2
    return bool(np.all(np.abs(piece.centre[:2] - np.asarray(PURPLE_BOARD_XY)) <= half))


class Tracker:
    """What has touched what since the episode began. Nut contact spreads by touch and never clears."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.carriers = set(NUT)
        self.handled: set[str] = set()  # plain croissants an arm or gripper has touched
        self.carried: set[str] = set()  # plain croissants lifted clear in the purple jaws
        self.in_jaws = dict.fromkeys(PLAIN, 0)
        self.history: list[dict] = []
        self.steps = 0
        self.view: dict = {}

    def update(self, states: dict, arms: dict[str, list[geo.Piece]] | None = None) -> None:
        """One control step: ``states`` of every object, ``arms`` the collision pieces of each arm (None offline)."""
        self.steps += 1
        body = {n: pieces(n, states[n]) for n in SPECS}
        hands = {}
        for side in ARMS:
            got = (arms or {}).get(side)
            if got:
                body[side] = got
                hands[side] = [p for p in got if p.tag in HAND_LINKS]
        names = list(body)
        box = {n: geo.extent(body[n]) for n in names}
        touch = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if np.any(box[a][0] - box[b][1] > TOUCH) or np.any(box[b][0] - box[a][1] > TOUCH):
                    continue
                if geo.gap(body[a], body[b], reach=TOUCH) <= TOUCH:
                    touch.add((a, b))
                    touch.add((b, a))
        grew = True
        while grew:
            grew = False
            for a, b in touch:
                if a in self.carriers and b not in self.carriers:
                    self.carriers.add(b)
                    grew = True
        jaws = {side: [p for p in body["purple_tongs"] if p.tag.startswith(f"{side}:jaw")]
                for side in ("arm_l", "arm_r")}
        for c in PLAIN:
            if any((c, arm) in touch for arm in ARMS):
                self.handled.add(c)
            others = [n for n in names if n not in (c, "purple_tongs") and (c, n) in touch]
            held = all(geo.gap(body[c], jaws[side], reach=TOUCH) <= TOUCH for side in jaws)
            aloft = box[c][0][2] > TOP + OFF_TABLE
            self.in_jaws[c] = self.in_jaws[c] + 1 if held and aloft and not others else 0
            if self.in_jaws[c] >= CARRY_STEPS:
                self.carried.add(c)
        on_plate, served = set(), set()
        for c in PLAIN:
            if (c, "plate") not in touch:
                continue
            on_plate.add(c)
            alone = not any((c, n) in touch for n in names if n not in (c, "plate", *PLAIN))
            if alone and box[c][0][2] > TOP + OFF_TABLE:
                served.add(c)
        clear = {n: all(geo.gap(h, body[n], reach=CLEAR) >= CLEAR for h in hands.values())
                 for n in ("purple_tongs", "plate", *PLAIN)}
        mids = [p for p in body["purple_tongs"] if p.tag.endswith(":jaw:mid")]
        home = ("purple_tongs", "board_purple") in touch and all(_over_purple_board(p) for p in mids)
        pose = {n: (np.asarray(states[n].pos, float), np.asarray(states[n].quat_xyzw, float)) for n in LOOSE}
        self.history = (self.history + [pose])[-(REST_STEPS + 1):]
        self.view = dict(touch=touch, on_plate=on_plate, served=served, clear=clear, still=self._still(), home=home)

    def _still(self) -> bool:
        if len(self.history) < REST_STEPS + 1:
            return False
        first = self.history[0]
        for now in self.history[1:]:
            for n in LOOSE:
                if np.linalg.norm(now[n][0] - first[n][0]) > REST_SHIFT:
                    return False
                if 2 * math.acos(min(1.0, abs(float(now[n][1] @ first[n][1])))) > REST_TURN:
                    return False
        return True

    def good(self) -> list[str]:
        """Plain croissants carried in the purple tongs, free of nut contact and untouched by an arm."""
        return [c for c in PLAIN if c in self.carried and c not in self.carriers and c not in self.handled]

    def score(self) -> float:
        if "plate" in self.carriers:
            return 0.0
        good, v = self.good(), self.view
        if not good or any(c not in good for c in v.get("on_plate", ())):
            return 0.0
        served = [c for c in good if c in v.get("served", ())]
        if not served:
            return 0.25
        put_back = (v["home"] and "purple_tongs" not in self.carriers and v["clear"]["purple_tongs"]
                    and v["clear"]["plate"] and all(v["clear"][c] for c in served) and v["still"])
        return 1.0 if put_back else 0.75


TRACK = Tracker()
_seen = {"states": None, "t": -1.0}
TRACE = os.environ.get("TASK_TRACE", "")


def _track(objects, arms=None) -> None:
    if _seen["states"] is objects:
        return
    _seen["states"] = objects
    TRACK.update(objects, arms)


@parts.on_reset
def _reset(sim) -> None:
    TRACK.reset()
    _seen.update(states=None, t=-1.0)


@parts.on_step
def _step(sim) -> None:
    if sim.t < _seen["t"]:  # the episode clock went back: a new episode
        TRACK.reset()
    _seen["t"] = sim.t
    states = sim.states()
    links = None
    try:
        links = sim.robot_links(ARM_LINKS)
        arms = {"left_arm": arm_pieces(links["left"]), "right_arm": arm_pieces(links["right"])}
    except Exception as err:  # noqa: BLE001  no robot reading: score as if a gripper held everything
        print(f"[allergy-order] arm poses not read: {err!r}", file=sys.stderr, flush=True)
        arms = None
    _seen["states"] = states
    TRACK.update(states, arms)
    if arms is None:
        TRACK.view["clear"] = dict.fromkeys(TRACK.view["clear"], False)
    if TRACE and TRACK.steps % int(TRACE) == 0:
        _trace(sim, states, links)


def _trace(sim, states, links) -> None:
    """One line of what the scorer sees, for reading a run's log."""
    def mm(v):
        return np.round((np.asarray(v, float) - [0, 0, TOP]) * 1000, 1).tolist()

    tcp = {}
    for side in ("left", "right"):
        if links:
            g = links[side]["gripper"]
            tcp[side] = mm(g[:3, 3] + g[:3, :3] @ np.array([0.0, 0.0, 0.134]))
    pairs = sorted({tuple(sorted(t)) for t in TRACK.view["touch"]})
    tongs = {n: (mm(states[n].pos), np.round(states[n].quat_xyzw, 3).tolist(),
                 [round(states[n].joints.get(j, 0.0), 3) for j in ("arm_l", "arm_r")])
             for n in ("purple_tongs", "black_tongs")}
    print(f"[trace] step {TRACK.steps} t {sim.t:.2f} score {TRACK.score():.2f} tcp {tcp} tongs {tongs} "
          f"plain {[mm(states[c].pos) for c in PLAIN]} plate {mm(states['plate'].pos)} "
          f"new carriers {sorted(TRACK.carriers - set(NUT))} carried {sorted(TRACK.carried)} "
          f"handled {sorted(TRACK.handled)} home {TRACK.view['home']} still {TRACK.view['still']} "
          f"clear {TRACK.view['clear']} touch {pairs}", file=sys.stderr, flush=True)


def score(objects):
    """0 if the plate ever met nut contact, if a croissant on the plate was not carried there clean in the purple
    tongs, or if none was carried so; 0.25 for one carried clean; 0.75 once it rests on the plate; 1.0 with the
    purple tongs back on their board, clean, let go, and nothing moving for a second."""
    if "isaaclab" not in sys.modules:
        _track(objects)
    return float(TRACK.score())


def success(objects):
    return bool(score(objects) >= 1.0)


def distance(objects):
    return float(min(np.linalg.norm(np.asarray(objects[c].pos[:2]) - PLATE_XY) for c in PLAIN))


# --- demo -------------------------------------------------------------------------------------------------------------
PAD_TIP = 0.013  # pad tips below the TCP
PADS_OVER = 0.001  # pad tips this far above the handle's lower edge (the board is just below it)
PINCH = 0.004  # fingers commanded this far past where the jaws meet the croissant
TRAVEL = 0.095


def half_span(closed: float) -> float:
    """Half the distance across the two handles where a gripper's pads meet them (the hidden flats' widest point),
    each arm ``closed`` rad in from free."""
    size, centre, r = _flat(1, TONG_FREE - closed, 0.0)
    signs = np.array([[sx, 1.0, 0.0] for sx in (-1, 1)])
    return float((centre + (signs * size / 2) @ r.T)[:, 1].max())


def jaw_gap(closed: float) -> float:
    """The gap between the jaws' middle plates, each arm ``closed`` rad in from free."""
    return 2 * (PIVOT[1] + (JAW_X - PIVOT[0]) * math.sin(TONG_FREE - closed))


def closed_for(gap: float) -> float:
    """How far in (rad) each arm is with the jaws' middle plates ``gap`` apart."""
    return TONG_FREE - math.asin((gap / 2 - PIVOT[1]) / (JAW_X - PIVOT[0]))


def lying(rise: float) -> tuple[float, float]:
    """For tongs lying with their jaws' heels ``rise`` above the bend's surface: (the handle's lower edge at GRIP_X,
    the lower edge of the jaws' middle plates at their near end), both above that surface."""
    tilt = _tilt(rise)
    low = (JAW_X - JAW_MID / 2 - PIVOT[0]) * math.sin(tilt) - JAW_DROP * math.cos(tilt)
    return (GRIP_X - PIVOT[0]) * math.sin(tilt) - (SLEEVE_H - STRIP_H) / 2, low


def serve_plan(which: str = "purple_tongs") -> list[Waypoint]:
    """Waypoints that take a pair of tongs by squeezing its handle, carry it over the nearer plain croissant, lower
    the open jaws either side of it, squeeze, lift it across to the plate, let it down, and lay the tongs back where
    they were."""
    purple = which == "purple_tongs"
    bend_xy = PURPLE_BEND_XY if purple else BLACK_BEND_XY
    surface = PURPLE_BOARD[2] if purple else 0.0  # what the bent end lies on, above the table
    handle, jaws = lying(0.0 if purple else BOARD[2])
    grip_z = surface + LIFT + handle + PADS_OVER + PAD_TIP  # TCP above the table when taking the tongs
    jaw_below = grip_z - (surface + LIFT + jaws)  # the jaws' lower edge below the TCP while the tongs are held
    ahead = JAW_X - GRIP_X  # the jaws' middle ahead of the TCP
    take = (bend_xy[0] + GRIP_X, bend_xy[1])
    pick = (ROW[0] - ahead, PLAIN_BOARD_XY[1])
    drop = (PLATE_XY[0] - ahead, PLATE_XY[1])
    hold = -(half_span(HOLD_ANGLE) - 0.0002)
    grasp = -(half_span(closed_for(CROISSANT.width)) - PINCH)
    pick_z = BOARD[2] + 0.004 + jaw_below  # jaws 4 mm above the board
    drop_z = PLATE_T + 0.015 + jaw_below  # croissant 11 mm above the plate's well
    back_z = grip_z + 0.002

    def at(xy, z):
        return (float(xy[0]), float(xy[1]), TOP + z)

    return [
        Waypoint(1.5, at(take, TRAVEL), FINGER_OPEN, "over the tongs"),
        Waypoint(1.5, at(take, grip_z), FINGER_OPEN, "down beside the handle"),
        Waypoint(1.0, at(take, grip_z), hold, "squeeze the handle"),
        Waypoint(0.5, at(take, grip_z), hold, "settle"),
        Waypoint(1.5, at(take, TRAVEL), hold, "lift the tongs"),
        Waypoint(2.5, at(pick, TRAVEL), hold, "over the plain croissant"),
        Waypoint(0.5, at(pick, TRAVEL), hold, "settle"),
        Waypoint(2.0, at(pick, pick_z), hold, "jaws down either side of it"),
        Waypoint(0.5, at(pick, pick_z), hold, "settle"),
        Waypoint(1.2, at(pick, pick_z), grasp, "squeeze the jaws onto it"),
        Waypoint(0.5, at(pick, pick_z), grasp, "settle"),
        Waypoint(1.5, at(pick, pick_z + 0.050), grasp, "lift it"),
        Waypoint(2.5, at(drop, pick_z + 0.050), grasp, "across to the plate"),
        Waypoint(1.5, at(drop, drop_z), grasp, "down over the plate"),
        Waypoint(0.8, at(drop, drop_z), hold, "let it go"),
        Waypoint(0.5, at(drop, drop_z), hold, "settle"),
        Waypoint(1.2, at(drop, TRAVEL), hold, "up"),
        Waypoint(2.5, at(take, TRAVEL), hold, "back over where the tongs lay"),
        Waypoint(1.5, at(take, back_z), hold, "lower the tongs"),
        Waypoint(0.8, at(take, back_z), FINGER_OPEN, "let go"),
        Waypoint(1.2, at(take, 0.150), FINGER_OPEN, "up and clear"),
        Waypoint(3.0, at(take, 0.150), FINGER_OPEN, "wait"),
    ]


PURPLE_ARM = "right" if SIDE > 0 else "left"  # the arm on the purple board's side
BLACK_ARM = "left" if SIDE > 0 else "right"


def demo():
    """The arm on the purple board's side serves the nearer plain croissant with the purple tongs and lays them back
    on the purple board. It plays open loop from the published start layout. With TASK_DEMO=black the other arm
    does the same with the black-handled tongs instead, which scores 0."""
    if os.environ.get("TASK_DEMO", "").strip().lower() == "black":
        return {BLACK_ARM: serve_plan("black_tongs")}
    return {PURPLE_ARM: serve_plan("purple_tongs")}
