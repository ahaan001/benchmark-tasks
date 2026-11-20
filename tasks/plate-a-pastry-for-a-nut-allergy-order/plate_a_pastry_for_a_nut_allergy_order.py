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
CRUST_PATCH = 0.010  # radius of the patch the crust gives over under a jaw: it resists twisting between the jaws
ROW = (0.425, 0.575)  # the two croissants on a board, front to back
LIFT = 0.001  # loose things start 1 mm above what they rest on


def _croissant(name, xy, z, almond):
    pts, faces, normals = CROISSANT.mesh()
    shapes = [parts.hull(p, geo.hull_faces(p), patch=CRUST_PATCH) for p in CROISSANT.pieces()]
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
# Part frame: origin on the surface under the bent end, x along the tongs to the jaws, z up; the tongs lie on their
# side, so the arms swing about the vertical through the bend and a gripper pointing down squeezes them across y.
# A one-piece tong has no joint and no stop: the bend is the spring. Here the arm on the +y side, with the bend, is
# the part's body and the other arm turns on a sprung hinge at the bend. The shapes are written at the free opening
# (joint 0); the joint reads twice what each arm is squeezed in.
TONG_LEN = 0.305  # 12 in
TONG_OPEN = 0.108  # across the open jaws, outside (4.25 in)
TONG_MASS = 0.132  # 0.29 lb
STRIP_H, STRIP_T = 0.020, 0.004  # each arm: a strip on edge (20 gauge steel; 4 mm here so contacts hold)
STRIP_X = (0.009, 0.232)
JAW_H, JAW_T = 0.036, 0.004  # scalloped jaw, 1.43 in tall
JAW_X, JAW_MID, JAW_WING, JAW_CUP = 0.2675, 0.030, 0.024, math.radians(14.0)
JAW_TOE = math.radians(6.0)  # each jaw is set in toward the other: they lie parallel when shut on a croissant
SLEEVE_X, SLEEVE_T, SLEEVE_H = (0.020, 0.172), 0.0065, 0.022  # coated handle
SLEEVE_GRIP = (2.0, 1.8)  # the coating's friction, static and dynamic
SLEEVE_PATCH = 0.008  # radius of the patch a rubber pad beds onto the coating over: it resists twisting in the grip
ARM_CENTRE = 0.160  # each arm's mass centre along it from the bend: strip, jaw plates and coating weighed out
PIVOT = (0.006, 0.004)  # the hinge's x on the centre line; each arm's inner face this far to its side there
BEND_R = 0.005
TONG_SPRING = 3.0  # N m/rad between the arms (an estimate: about 6 N on each handle at mid-length closes them)
TONG_DAMPING = 0.05
TONG_SPREAD = 0.06  # rad an arm can be pushed open past its free position
JAW_DROP = (JAW_H - STRIP_H) / 2  # a jaw's lower edge below its strip's
SLEEVE_DROP = (SLEEVE_H - STRIP_H) / 2  # the coating's lower edge below the strip's
LEVEL = (0.0, 0.0)  # lying: (how far the jaw end is raised, how far the bend is above the surface under it)


def _arm_frame(side: int, angle: float, tilt: float) -> np.ndarray:
    """An arm's rotation: turned ``angle`` out from the centre line on its side, its jaw end raised by ``tilt``."""
    return parts._rot((0, 0, 1), side * angle) @ parts._rot((0, 1, 0), -tilt)


def _place(side, angle, lie, size, x, y_in, z) -> tuple:
    """(size, centre, rotation) of a box on an arm: its middle ``x`` along the arm from the bend, its inner face
    ``y_in`` off the arm's own line toward the outside, its middle ``z`` above the strip's lower edge."""
    tilt, up = lie
    r = _arm_frame(side, angle, tilt)
    hinge = np.array([PIVOT[0], 0.0, up])
    local = np.array([x - PIVOT[0], side * (PIVOT[1] + y_in + size[1] / 2), z])
    return np.asarray(size, float), hinge + r @ local, r


def _jaw_boxes(side, angle, lie):
    """The jaw's three plates: the two wings turned in by JAW_CUP so the jaw cups what it holds, the whole jaw set
    in by JAW_TOE about its middle."""
    tilt, up = lie
    r = _arm_frame(side, angle, tilt)
    hinge = np.array([PIVOT[0], 0.0, up])
    middle = np.array([JAW_X - PIVOT[0], side * (PIVOT[1] + JAW_T / 2), STRIP_H / 2])
    toe = parts._rot((0, 0, 1), -side * JAW_TOE)
    reach = JAW_MID / 2 + JAW_WING / 2 * math.cos(JAW_CUP)
    out = []
    for dx, cup in ((-reach, JAW_CUP), (0.0, 0.0), (reach, -JAW_CUP)):
        offset = np.array([dx, -side * abs(math.sin(cup)) * JAW_WING / 2, 0.0])
        size = np.array([JAW_WING if cup else JAW_MID, JAW_T, JAW_H])
        out.append((size, hinge + r @ (middle + toe @ offset), r @ toe @ parts._rot((0, 0, 1), side * cup)))
    return out


def _corners(size, centre, r) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return centre + (signs * size / 2) @ r.T


# where a jaw's lower edge begins, along the arm from the bend: lying, the tongs rest there
JAW_HEEL = float(min(_corners(*b)[:, 0].min() for b in _jaw_boxes(1, 0.0, LEVEL)))


def _lie(rise: float) -> tuple[float, float]:
    """How the tongs lie with their jaws' heels ``rise`` above the surface under the bent end: (how far the jaw end
    is raised, rad; how far the bend's own lower edge is above that surface). Lying flat, the coating's near end
    carries the bent end, not the bend itself."""
    def lift(t):
        return max(0.0, SLEEVE_DROP * math.cos(t) - (SLEEVE_X[0] - PIVOT[0]) * math.sin(t))

    lo, hi = 0.0, 0.5
    for _ in range(50):
        t = (lo + hi) / 2
        heel = lift(t) + (JAW_HEEL - PIVOT[0]) * math.sin(t) - JAW_DROP * math.cos(t)
        lo, hi = (t, hi) if heel < rise else (lo, t)
    return (lo + hi) / 2, lift((lo + hi) / 2)


def _sleeve(side, angle, lie):
    return _place(side, angle, lie, (SLEEVE_X[1] - SLEEVE_X[0], SLEEVE_T, SLEEVE_H), (SLEEVE_X[0] + SLEEVE_X[1]) / 2,
                  -(SLEEVE_T - STRIP_T) / 2, STRIP_H / 2)


def _free_angle() -> float:
    """The half-angle between the arms at rest that makes the open jaws TONG_OPEN across, outside."""
    lo, hi = 0.05, 0.4
    for _ in range(40):
        a = (lo + hi) / 2
        widest = max(_corners(*b)[:, 1].max() for b in _jaw_boxes(1, a, LEVEL))
        lo, hi = (a, hi) if widest < TONG_OPEN / 2 else (lo, a)
    return (lo + hi) / 2


def _shut_angle() -> float:
    """How far in each arm is when the two jaws are 1 mm apart at their nearest."""
    lo, hi = 0.0, TONG_FREE
    for _ in range(40):
        c = (lo + hi) / 2
        nearest = min(_corners(*b)[:, 1].min() for b in _jaw_boxes(1, TONG_FREE - c, LEVEL))
        lo, hi = (c, hi) if nearest > 0.0005 else (lo, c)
    return lo


TONG_FREE = _free_angle()
TONG_SHUT = _shut_angle()


def _arm(side: int, lie, handle) -> list:
    """One arm's shapes at the free opening: strip, jaw plates and coated handle."""
    def box(placed, color, **more):
        size, centre, r = placed
        return parts.box(size, at=centre, color=color, rot=tuple(parts._matrix_to_quat(r)), **more)

    shapes = [box(_place(side, TONG_FREE, lie, (STRIP_X[1] - STRIP_X[0], STRIP_T, STRIP_H),
                         (STRIP_X[0] + STRIP_X[1]) / 2, 0.0, STRIP_H / 2), STEEL)]
    shapes += [box(placed, STEEL) for placed in _jaw_boxes(side, TONG_FREE, lie)]
    shapes.append(box(_sleeve(side, TONG_FREE, lie), handle, friction=SLEEVE_GRIP, patch=SLEEVE_PATCH))
    return shapes


def _arm_centre(side: int, lie) -> tuple:
    return tuple(float(v) for v in _place(side, TONG_FREE, lie, (0.0, STRIP_T, 0.0), ARM_CENTRE, 0.0, STRIP_H / 2)[1])


def _tongs(name: str, xy, z: float, rise: float, handle):
    """A pair of tongs with the bent end at ``xy`` on a surface at ``z`` and the jaws' heels ``rise`` above it
    (0: lying on one surface; a board's thickness: jaws on the board, bent end on the table)."""
    lie = _lie(rise)
    bend = parts.cylinder(BEND_R, STRIP_H, at=(BEND_R, 0.0, lie[1] + STRIP_H / 2), color=STEEL, axis="z")
    hinge = parts.Hinge("arm_r", shapes=_arm(-1, lie, handle), pivot=(PIVOT[0], 0.0, lie[1] + STRIP_H / 2),
                        axis=(0.0, 0.0, -1.0), lower=-2 * TONG_SHUT, upper=2 * TONG_SPREAD, mass=TONG_MASS / 2,
                        spring=TONG_SPRING, damping=TONG_DAMPING, centre=_arm_centre(-1, lie))
    return parts.Part(name, pos=(xy[0], xy[1], z + LIFT), mass=TONG_MASS / 2, friction=(0.6, 0.5),
                      shapes=[*_arm(1, lie, handle), bend], joints=[hinge], centre=_arm_centre(1, lie))


def tong_state(name: str, closed: float = 0.0, pos=None, quat_xyzw=None):
    """The state of a pair of tongs with each arm ``closed`` rad in from free, its frame (origin under the bent end,
    x along its centre line) at ``pos``/``quat_xyzw`` (default: where it starts)."""
    spec = SPECS[name]
    p = np.asarray(spec.pos if pos is None else pos, float)
    r = np.eye(3) if quat_xyzw is None else parts._quat_to_matrix(quat_xyzw)
    hinge = np.array([PIVOT[0], 0.0, 0.0])
    turn = parts._rot((0, 0, 1), -closed)  # the body is the +y arm: closing turns it clockwise
    return spec.state(joints={"arm_r": -2 * closed}, pos=p + r @ (hinge - turn @ hinge),
                      quat_xyzw=parts._matrix_to_quat(r @ turn))


def tong_frame(state) -> tuple[np.ndarray, np.ndarray, float]:
    """The reverse of :func:`tong_state`: (position, rotation) of the tongs' own frame and how far in each arm is."""
    closed = -state.joints["arm_r"] / 2
    hinge = np.array([PIVOT[0], 0.0, 0.0])
    turn = parts._rot((0, 0, 1), -closed)
    r = parts._quat_to_matrix(state.quat_xyzw) @ turn.T
    return np.asarray(state.pos, float) - r @ (hinge - turn @ hinge), r, closed


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
            arm = "arm_l" if link == "base" else link  # the tongs' body is their +y arm
            tag = f"{arm}:jaw:mid" if abs(s.size[0] - JAW_MID) < 1e-9 else f"{arm}:jaw"
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

    def lay(state):
        p, r, closed = tong_frame(state)
        return (mm(p), [round(math.degrees(math.asin(max(-1.0, min(1.0, r[2, 0])))), 1),  # jaws up (+), down (-)
                        round(math.degrees(math.atan2(r[2, 1], r[2, 2])), 1),  # rolled about its own length
                        round(math.degrees(math.atan2(r[1, 0], r[0, 0])), 1)],  # turned about the vertical
                round(closed, 4))

    tcp, fingers, press = {}, {}, {}
    if links:
        joints = sim.robot_joints()
        try:
            targets = sim.robot_targets()
        except Exception:  # noqa: BLE001
            targets = None
        for side in ("left", "right"):
            g = links[side]["gripper"]
            tcp[side] = mm(g[:3, 3] + g[:3, :3] @ np.array([0.0, 0.0, 0.134]))
            fingers[side] = [round(joints[side].get(j, 0.0) * 1000, 2) for j in ("joint7", "joint8")]
            if targets:  # N each finger's drive pushes inward with
                press[side] = [round((targets[side][j] - joints[side][j]) * FINGER_K, 2) for j in ("joint7", "joint8")]
    pairs = sorted({tuple(sorted(t)) for t in TRACK.view["touch"]})
    tongs = {n: lay(states[n]) for n in ("purple_tongs", "black_tongs")}
    food = {c: mm(states[c].pos) for c in ("plain_1", "almond_1")}
    print(f"[trace] step {TRACK.steps} t {sim.t:.2f} score {TRACK.score():.2f} tcp {tcp} fingers {fingers} "
          f"press {press} tongs {tongs} food {food} plate {mm(states['plate'].pos)} "
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
FINGER_K = 5000.0  # N/m: the gripper's finger drive (yam_arena.constants.GRIPPER)
PAD_TIP = 0.0128  # a pad's tip below the TCP
PAD_SHOULDER = 0.013  # a pad's flat face is widest this far ahead of the TCP: it meets a handle there
PAD_CLEAR = 0.0015  # pad tips stay this far above what the tongs lie on
PADS_UNDER = 0.003  # ... and go this far below the handle's lower edge where there is room
GRIP_X = 0.145  # the TCP from the bent end: the pads then meet the handles where the tongs balance
HOLD_ANGLE = 0.040  # rad each arm is squeezed in to carry the tongs with their jaws open
GRASP_ANGLE = 0.061  # rad each arm is in when both jaws meet a croissant lying between them
BELLY = 0.0028  # a croissant is a crescent: the jaws meet it together with their centre line this far toward its belly
CARRY_FORCE = 12.0  # N each pad presses its handle with while the jaws hold a croissant
TRAVEL = 0.095


def half_span(closed: float) -> float:
    """Half the distance across the two handles where a gripper's pads meet them, each arm ``closed`` rad in from
    free."""
    size, centre, r = _sleeve(1, TONG_FREE - closed, LEVEL)
    n = r[:, 1]
    face = centre + n * size[1] / 2
    return float(face[1] - n[0] * (GRIP_X + PAD_SHOULDER - face[0]) / n[1])


def jaw_gap(closed: float) -> float:
    """The gap between the middles of the jaws' middle plates, each arm ``closed`` rad in from free."""
    a = TONG_FREE - closed
    return 2 * (PIVOT[1] * math.cos(a) + (JAW_X - PIVOT[0]) * math.sin(a))


def closed_for(gap: float) -> float:
    """How far in (rad) each arm is with the jaws' middle plates ``gap`` apart."""
    lo, hi = -TONG_SPREAD, TONG_SHUT
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if jaw_gap(mid) > gap else (lo, mid)
    return (lo + hi) / 2


def spring_force(closed: float) -> float:
    """N the tongs' spring pushes back on each pad with, each arm ``closed`` rad in from free."""
    return TONG_SPRING * 2 * closed / (GRIP_X + PAD_SHOULDER - PIVOT[0])


def _fingers(closed: float, press: float | None = None) -> float:
    """The finger target that holds each arm ``closed`` rad in from free against ``press`` N on each pad (default:
    the tongs' own spring)."""
    press = spring_force(closed) if press is None else press
    return -(half_span(closed) - press / FINGER_K)


def lying(rise: float) -> tuple[float, float]:
    """For tongs lying with their jaws' heels ``rise`` above the surface under the bent end: (the handle's lower edge
    where the pads meet it, the jaws' heels), both above that surface."""
    tilt, up = _lie(rise)
    return up + (GRIP_X + PAD_SHOULDER - PIVOT[0]) * math.sin(tilt) - SLEEVE_DROP * math.cos(tilt), rise


def _taking(which: str):
    """Where and how a gripper takes a pair of tongs: (TCP xy, TCP height above the table, how far the jaws' heels
    hang below the TCP while they are held)."""
    purple = which == "purple_tongs"
    bend_xy = PURPLE_BEND_XY if purple else BLACK_BEND_XY
    surface = PURPLE_BOARD[2] if purple else 0.0  # what the bent end lies on, above the table
    handle, heels = lying(0.0 if purple else BOARD[2])
    grip_z = surface + max(handle - PADS_UNDER, PAD_CLEAR) + PAD_TIP
    return (bend_xy[0] + GRIP_X, bend_xy[1]), grip_z, grip_z - (surface + heels)


def _at(xy, z):
    return (float(xy[0]), float(xy[1]), TOP + z)


JAWS_OVER = 0.008  # the jaws' lower edge above the board while they close on a croissant
DROP_OVER = 0.012  # the croissant's underside above the plate's well when the jaws let it go, or more:
RIM_OVER = 0.010  # the handle passes at least this far above the plate's rim


def serve_plan(which: str = "purple_tongs") -> list[Waypoint]:
    """Waypoints that take a pair of tongs by squeezing its handle, carry it with open jaws over the nearer plain
    croissant, lower the jaws either side of it, squeeze, lift it across to the plate, let it down, and lay the
    tongs back where they were."""
    take, grip_z, heels = _taking(which)
    ahead = JAW_X - GRIP_X  # the jaws' middle ahead of the TCP
    pick = (ROW[0] - ahead, PLAIN_BOARD_XY[1] + SIDE * BELLY)
    drop = (PLATE_XY[0] - ahead, PLATE_XY[1] + SIDE * BELLY)
    hold, grasp = _fingers(HOLD_ANGLE), _fingers(GRASP_ANGLE, CARRY_FORCE)
    pick_z = BOARD[2] + JAWS_OVER + heels
    drop_z = max(PLATE_T + DROP_OVER + JAWS_OVER + heels, PLATE_H + RIM_OVER + PAD_TIP - PADS_UNDER)
    back_z = grip_z + 0.002
    return [
        Waypoint(1.5, _at(take, TRAVEL), FINGER_OPEN, "over the tongs"),
        Waypoint(1.5, _at(take, grip_z), FINGER_OPEN, "down beside the handle"),
        Waypoint(1.0, _at(take, grip_z), hold, "squeeze the handle"),
        Waypoint(0.5, _at(take, grip_z), hold, "settle"),
        Waypoint(1.5, _at(take, TRAVEL), hold, "lift the tongs"),
        Waypoint(2.5, _at(pick, TRAVEL), hold, "over the plain croissant"),
        Waypoint(0.4, _at(pick, TRAVEL), hold, "settle"),
        Waypoint(2.5, _at(pick, pick_z), hold, "jaws down either side of it"),
        Waypoint(0.5, _at(pick, pick_z), hold, "settle"),
        Waypoint(1.2, _at(pick, pick_z), grasp, "squeeze the jaws onto it"),
        Waypoint(0.5, _at(pick, pick_z), grasp, "settle"),
        Waypoint(1.5, _at(pick, pick_z + 0.050), grasp, "lift it"),
        Waypoint(2.5, _at(drop, pick_z + 0.050), grasp, "across to the plate"),
        Waypoint(1.5, _at(drop, drop_z), grasp, "down over the plate"),
        Waypoint(0.8, _at(drop, drop_z), hold, "let it go"),
        Waypoint(0.5, _at(drop, drop_z), hold, "settle"),
        Waypoint(1.0, _at(drop, drop_z + 0.025), hold, "up off it"),
        Waypoint(1.0, _at(drop, TRAVEL), hold, "up"),
        Waypoint(2.5, _at(take, TRAVEL), hold, "back over where the tongs lay"),
        Waypoint(1.5, _at(take, back_z), hold, "lower the tongs"),
        Waypoint(0.8, _at(take, back_z), FINGER_OPEN, "let go"),
        Waypoint(1.2, _at(take, 0.150), FINGER_OPEN, "up and clear"),
        Waypoint(3.0, _at(take, 0.150), FINGER_OPEN, "wait"),
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
