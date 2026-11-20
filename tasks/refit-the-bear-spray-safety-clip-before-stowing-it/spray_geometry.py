"""Sizes, shapes and geometric tests for the bear spray task. Pure Python and NumPy: no simulator, no kit.

Lengths in metres, angles in degrees. Three frames are used:

- the can's: origin at the middle of its base, z up its axis, +x toward the finger loop (the rear), y across. A point of the
  can's side view is (r, h): r along x from the axis, h up from the base. Every (r, h) below was read from the maker's
  square-on photo of the practice can, scaled by its listed 8.5 x 2 in (see README, "Where the sizes come from");
- the clip's (the safety wedge): origin on the trigger tab's centre line when the clip is home, x forward along the tab
  (the way the clip slides on), z square to the tab and away from the handle, y across;
- the world's, as the task file uses it: x forward from the midpoint between the arm bases, y left, z up from the table top.

Shapes are plain data: ("box", centre, size, rot), ("capsule", centre, radius, length, axis), ("cylinder", centre, radius,
height) about z, ("lathe", profile) of (radius, z) turned about z, ("prism", outline, width) of an (x, z) outline, counter-clockwise
seen from -y, as wide as `width` across y. `rot` is a rotation matrix whose columns are the box's own axes in the part's frame.
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * k @ k


def side(r_mm: float, h_mm: float, y_mm: float = 0.0) -> np.ndarray:
    """A point of the side view, as a point of the can's frame."""
    return np.array([r_mm, y_mm, h_mm]) * MM


def bar(p0, p1, thick: float, width: float, name: str = "") -> tuple:
    """A box in the side view's plane from (r, h) p0 to p1 (mm), `thick` across that line, `width` across the can (m)."""
    a, b = np.asarray(p0, float) * MM, np.asarray(p1, float) * MM
    d = b - a
    length = float(np.linalg.norm(d))
    ang = math.degrees(math.atan2(d[1], d[0]))  # above the horizontal
    c = (a + b) / 2
    return ("box", np.array([c[0], 0.0, c[1]]), (length, width, thick), rot_y(-ang))


# -- the can and its head (SABRE Frontiersman MAX 7.9 oz; the maker's practice can FBADX-00 has the same can, head and clip) --------
CAN_R = 0.026  # 52 mm across (listed 2 in)
CAN_WALL_H = 0.150  # the straight wall; the shoulder domes in above it
COLLAR_R, COLLAR_Z = 0.01825, (0.158, 0.171)  # the head's skirt over the valve cup
BLOCK_R, BLOCK_W, BLOCK_Z = (-0.0215, 0.010), 0.030, (0.171, 0.2035)  # the head's body; the nozzle is on its front face (-x)
HEAD_TOP = 0.2035
CAN_H = 0.2185  # to the top of a seated clip (listed 8.5 in is 0.216)
CAN_MASS = 0.15  # the emptied practice can (est: listed 0.85 lb less 7.9 oz of contents, less its card; weigh it)
CAN_COM_Z = 0.105  # est: an empty can with its head
FOOT = (0.048, 0.014, 0.001)  # the can stands on two crossed strips, as on its rim: a cylinder's end rocks on GPU physics
LOOP_W = 0.010  # the finger loop, across (est)
LOOP_TOP = ((23.5, 202.5), -9.46)  # the top of the loop's upper bar: a point (r, h) in mm, and its slope, down to the rear
NOZZLE = side(-21.5, 183.5)

# The trigger tab ("actuator tab"): a lever hinged near the front of the head, its rear end over the finger loop.
TAB_PIVOT = side(-15.0, 201.0)  # est: inside the head, not in any photo
TAB_ANGLE = 21.0  # at rest it rises to the rear by this much
TAB_START = (14.0, 202.3)  # where its centre line comes out of the head, (r, h) mm
TAB_REAR_R = 38.0  # and where it ends, r mm
TAB_T, TAB_W = 0.0042, 0.010  # thickness (photo), width across (est)
TAB_TRAVEL_DEG = 9.0  # est: its rear end comes down about 8 mm
TAB_FIRED_DEG = 4.5  # half of it
TAB_MASS = 0.004
TAB_RATE = 1.3  # N m/rad (est)
TAB_PRELOAD_DEG = 4.5  # the spring is wound this far at rest: about 1.9 N at the rear end at rest, 3.8 N at half travel (est)
RIDGE_R, RIDGE_PROUD = 0.0012, 0.0008  # the one ridge on the tab's top that the clip's catch rides over (est)
RIDGE_AT = 0.0146  # along the tab from TAB_START

_u = np.array([math.cos(math.radians(TAB_ANGLE)), math.sin(math.radians(TAB_ANGLE))])  # along the tab, to the rear, in (r, h)
_n = np.array([-_u[1], _u[0]])  # square to it, away from the handle
TAB_LEN = (TAB_REAR_R - TAB_START[0]) * MM / _u[0]  # from TAB_START to its rear end, along it


def on_tab(t: float, n: float = 0.0) -> np.ndarray:
    """The (r, h) point, in mm, `t` metres along the tab from TAB_START and `n` metres off its centre line."""
    return np.asarray(TAB_START, float) + (t * _u + n * _n) / MM


TAB_REAR = side(*on_tab(TAB_LEN))  # the middle of its rear end
TAB_REACH = float(np.linalg.norm(TAB_REAR - TAB_PIVOT))  # 54 mm: the rear end's radius about the pivot


def loop_top_h(r_mm: float) -> float:
    (r0, h0), slope = LOOP_TOP
    return h0 + (r_mm - r0) * math.tan(math.radians(slope))


def can_frame_shapes() -> list:
    """The can and the fixed part of its head: (name, look, shape)."""
    (r0, h0), slope = LOOP_TOP
    top0, top1 = (14.0, loop_top_h(14.0)), (44.5, loop_top_h(44.5))
    off = 3.0 * np.array([math.sin(math.radians(slope)), -math.cos(math.radians(slope))])  # to the middle of a 6 mm bar, square to the line and under it
    shapes = [
        ("foot_a", "can", ("box", np.array([0.0, 0.0, FOOT[2] / 2]), FOOT, None)),
        ("foot_b", "can", ("box", np.array([0.0, 0.0, FOOT[2] / 2]), (FOOT[1], FOOT[0], FOOT[2]), None)),
        ("wall", "can", ("cylinder", np.array([0.0, 0.0, (FOOT[2] + CAN_WALL_H) / 2]), CAN_R, CAN_WALL_H - FOOT[2])),
        ("shoulder", "can", ("lathe", ((0.0, CAN_WALL_H), (CAN_R, CAN_WALL_H), (COLLAR_R, COLLAR_Z[0]), (0.0, COLLAR_Z[0])))),
        ("collar", "head", ("cylinder", np.array([0.0, 0.0, sum(COLLAR_Z) / 2]), COLLAR_R, COLLAR_Z[1] - COLLAR_Z[0])),
        ("block", "head", ("box", np.array([sum(BLOCK_R) / 2, 0.0, sum(BLOCK_Z) / 2]), (BLOCK_R[1] - BLOCK_R[0], BLOCK_W, BLOCK_Z[1] - BLOCK_Z[0]), None)),
        ("loop_top", "head", bar(np.add(top0, off), np.add(top1, off), 0.006, LOOP_W)),
        ("loop_rear", "head", ("box", side(43.0, 184.0), (0.005, LOOP_W, 0.025), None)),
        ("loop_under", "head", ("box", side(26.5, 168.75), (0.033, LOOP_W, 0.0055), None)),
        ("loop_spur", "head", bar((38.0, 169.0), (43.5, 164.0), 0.003, LOOP_W)),
    ]
    return shapes


def can_marks() -> list:
    """What is seen and nothing collides with: the label band, the nozzle, the head's top cover: (name, look, shape)."""
    return [
        ("label", "label", ("cylinder", np.array([0.0, 0.0, 0.085]), CAN_R + 0.0003, 0.090)),
        ("nozzle", "nozzle", ("box", NOZZLE + [-0.0004, 0.0, 0.0], (0.001, 0.012, 0.008), None)),
    ]


def tab_shapes() -> list:
    """The trigger tab's part that shows, as it rests: a bar with a round rear end, and the ridge on its top."""
    end = TAB_LEN - TAB_T / 2  # the round end's centre
    return [
        ("tab_bar", "head", bar(on_tab(0.0), on_tab(end), TAB_T, TAB_W)),
        ("tab_end", "head", ("capsule", side(*on_tab(end)), TAB_T / 2, TAB_W - TAB_T, "Y")),
        ("tab_ridge", "head", ("capsule", side(*on_tab(RIDGE_AT, TAB_T / 2 + RIDGE_PROUD - RIDGE_R)), RIDGE_R, TAB_W - 2 * RIDGE_R, "Y")),
    ]


# -- the safety clip (SABRE's wedge, part AV-NGW), in its own frame ----------------------------------------------------------------
# A sleeve on the tab's rear end: a strap over the tab, a web under it that fills the gap between tab and handle, a strut that
# joins them behind the tab's end, and two low rails beside the tab. Its underside lies on the loop's top bar when it is home.
CLIP_T0 = 0.020  # the clip's origin, home, is this far along the tab from TAB_START
CLIP_W, STRAP_W = 0.014, 0.012  # across (est): the strap is narrower, so that fingers on the clip's sides do not clamp it
SLOT = (-0.0024, 0.0031)  # the sleeve's floor and roof, off the tab's centre line: 0.3 mm under the tab, 1.0 mm over it (the catch needs the room)
RAIL_GAP = 0.011  # between the rails: the tab, 10 mm, with 0.5 mm a side
STRAP_T = 0.0018
STRAP_X = (-0.0075, 0.014)  # from its hinge at the rear to its front end
CATCH_X = 0.008  # the catch under the strap: a round bump, the same size as the tab's ridge
CATCH_BACKLASH = 0.0010  # home, the catch is this far past the ridge
CLIP_MASS, STRAP_MASS = 0.0045, 0.0008  # the listed 0.01 lb; the strap's share (est)
STRAP_RATE = 2.0  # N m/rad (est): about 3.5 N of push takes the catch over the ridge
STRAP_LIFT_DEG = 8.0
WEB = ((0.0105, -0.0024), (-0.00639, -0.0024), (-0.00729, -0.0140), (0.0105, -0.00354))  # the web's outline, (x, z): floor in front, floor at the rear, rear foot, front foot
STRUT_X = (-0.0086, -0.0062)
GRIP_AT = np.array([-0.011, 0.0, 0.0019])  # where a gripper's TCP goes to hold it: its pads then cover the strut, the rails and the web's rear, and clear the mat by 3 mm
CLIP_COM = np.array([-0.0008, 0.0, -0.0035])


def clip_home() -> tuple:
    """The seated clip in the can's frame: (rotation, position)."""
    x = np.array([-_u[0], 0.0, -_u[1]])
    z = np.array([_n[0], 0.0, _n[1]])
    return np.column_stack([x, np.cross(z, x), z]), side(*on_tab(CLIP_T0))


UNDER_DEG = math.degrees(math.atan2(WEB[3][1] - WEB[2][1], WEB[3][0] - WEB[2][0]))  # the underside's slope in the clip's frame: 30.5


def clip_standing(yaw_deg: float = 0.0) -> tuple:
    """The clip standing on its underside on a level surface, its open end toward `yaw_deg` from +x: (rotation, height of its origin)."""
    rot = rot_y(UNDER_DEG)
    foot = rot @ np.array([WEB[2][0], 0.0, WEB[2][1]])
    return rot_z(yaw_deg) @ rot, float(-foot[2])


def clip_frame_shapes() -> list:
    floor, roof = SLOT
    rail_h = 0.002
    rail_x = (STRUT_X[1], WEB[0][0])
    rail_y = (RAIL_GAP + (CLIP_W - RAIL_GAP) / 2) / 2
    rail_size = (rail_x[1] - rail_x[0], (CLIP_W - RAIL_GAP) / 2, rail_h)
    strut_z = (-0.0085, roof + STRAP_T / 2)
    return [
        ("web", "clip", ("prism", WEB, CLIP_W)),
        ("rail_l", "clip", ("box", np.array([sum(rail_x) / 2, rail_y, floor + rail_h / 2]), rail_size, None)),
        ("rail_r", "clip", ("box", np.array([sum(rail_x) / 2, -rail_y, floor + rail_h / 2]), rail_size, None)),
        ("strut", "clip", ("box", np.array([sum(STRUT_X) / 2, 0.0, sum(strut_z) / 2]), (STRUT_X[1] - STRUT_X[0], CLIP_W, strut_z[1] - strut_z[0]), None)),
    ]


STRAP_HINGE = np.array([STRAP_X[0], 0.0, SLOT[1] + STRAP_T / 2])


def strap_shapes() -> list:
    roof = SLOT[1]
    plate = ("box", np.array([sum(STRAP_X) / 2, 0.0, roof + STRAP_T / 2]), (STRAP_X[1] - STRAP_X[0], STRAP_W, STRAP_T), None)
    catch = ("capsule", np.array([CATCH_X, 0.0, roof + RIDGE_R - RIDGE_PROUD]), RIDGE_R, STRAP_W - 2 * RIDGE_R, "Y")
    c0 = np.array([STRAP_X[1] - 0.0004, roof + STRAP_T / 2])
    c1 = c0 + 0.0065 * np.array([math.cos(math.radians(55.0)), math.sin(math.radians(55.0))])
    mid, d = (c0 + c1) / 2, c1 - c0
    curl = ("box", np.array([mid[0], 0.0, mid[1]]), (float(np.linalg.norm(d)), STRAP_W, 0.0015), rot_y(-math.degrees(math.atan2(d[1], d[0]))))
    return [("strap", "clip", plate), ("catch", "clip", catch), ("curl", "clip", curl)]


def ridge_in_clip() -> np.ndarray:
    """The tab's ridge, in the frame of a clip that is home: for checking the catch against it."""
    return np.array([CLIP_T0 - RIDGE_AT, 0.0, TAB_T / 2 + RIDGE_PROUD - RIDGE_R])


# -- the seat ----------------------------------------------------------------------------------------------------------------------
SEAT_BACK = 0.0012  # seated: no more than this short of home along the tab (the catch is then past the ridge) ...
SEAT_OVER = 0.0010  # ... or past it,
SEAT_OFF = 0.0010  # within this of the tab's line across and square to it,
SEAT_TILT = 8.0  # and within this of square: the sleeve's play lets a seated clip lean 5 degrees when the can lies on it
START_BACK = 0.028  # the clip clears the tab's rear end when it is this far back along the tab


def clip_on_can(can_pos, can_rot, clip_pos, clip_rot) -> tuple:
    """Where the clip is from home, in the home clip's frame: (offset, tilt in degrees). Offset x < 0 is short of home."""
    home_rot, home_pos = clip_home()
    rel_pos = np.asarray(can_rot, float).T @ (np.asarray(clip_pos, float) - np.asarray(can_pos, float))
    rel_rot = np.asarray(can_rot, float).T @ np.asarray(clip_rot, float)
    d = home_rot.T @ (rel_pos - home_pos)
    turn = home_rot.T @ rel_rot
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(turn) - 1.0) / 2.0))))
    return d, tilt


def seated(offset, tilt: float) -> bool:
    return bool(-SEAT_BACK <= offset[0] <= SEAT_OVER and abs(offset[1]) <= SEAT_OFF and abs(offset[2]) <= SEAT_OFF and tilt <= SEAT_TILT)


def tab_free_deg(back: float = 0.0) -> float:
    """How far the tab can turn before a clip `back` short of home stops it. The sleeve's floor is 0.3 mm under the tab; a clip
    short of home also stands off the handle, by back x sin(30.5 deg), and goes down that far with the tab before the handle
    stops it. A clip whose web is not yet under the tab's end blocks nothing."""
    if CLIP_T0 + back - WEB[0][0] > TAB_LEN:
        return TAB_TRAVEL_DEG
    play = (-SLOT[0] - TAB_T / 2) + back * math.tan(math.radians(UNDER_DEG))
    return min(TAB_TRAVEL_DEG, math.degrees(play / TAB_REACH))


# -- points of the can, for the scorer ----------------------------------------------------------------------------------------------
def can_points() -> np.ndarray:
    """Points on the can's outline, in its frame: base and shoulder rims, the head's corners, the loop's rear, the tab's rear end."""
    th = np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
    rims = [[CAN_R * math.cos(t), CAN_R * math.sin(t), z] for z in (0.0, CAN_WALL_H) for t in th]
    head = [[x, y, z] for x in BLOCK_R for y in (-BLOCK_W / 2, BLOCK_W / 2) for z in BLOCK_Z]
    loop = [side(45.5, 199.0, y) for y in (-5.0, 5.0)] + [side(45.5, 172.0), side(43.5, 164.0)]
    return np.vstack([rims, head, loop, [TAB_REAR, side(36.5, 218.5)]])


def can_gap(points, can_pos, can_rot) -> float:
    """The least distance from world points to the can: its body as a cylinder, its head as a box round block, loop and tab."""
    p = (np.atleast_2d(points) - np.asarray(can_pos, float)) @ np.asarray(can_rot, float)
    rad = np.hypot(p[:, 0], p[:, 1])
    dz = np.maximum(np.maximum(-p[:, 2], p[:, 2] - COLLAR_Z[1]), 0.0)
    body = np.hypot(np.maximum(rad - CAN_R, 0.0), dz)
    lo, hi = np.array([BLOCK_R[0], -BLOCK_W / 2, 0.160]), np.array([0.0460, BLOCK_W / 2, CAN_H])
    head = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)
    return float(np.minimum(body, head).min())


HOLD_Z = 0.138  # where a gripper holds the can: on its straight wall just under the shoulder, close under the push that seats the clip, clear of the head

# -- the scene, heights above the table ----------------------------------------------------------------------------------------------
MAT = (0.40, 0.30, 0.003)  # black rubber bench mat, long side along x
MAT_XY = (0.26, -0.10)
CAN_XY = (0.30, -0.02)
CAN_YAW = -90.0  # the finger loop toward -y: the robot's right
CLIP_XY = (0.32, -0.20)
CLIP_YAW = 60.0  # its open end 60 degrees from +x toward +y: 30 degrees off the way it goes on, and side-on to the top camera

BIN = (0.356, 0.279, 0.076)  # Sterilite Large Clip Box base, outside, long side along x; its wall's height without the lid (est; 8.3 cm with it)
BIN_XY = (0.32, 0.27)
BIN_WALL = 0.0025
STOVE = (0.098, 0.077, 0.023)  # Esbit Pocket Stove, folded shut
STOVE_XY = (0.22, 0.175)
STOVE_MASS = 0.092
CUP_R, CUP_H = 0.0405, 0.097  # GSI Glacier Stainless 10 fl oz Camp Cup
CUP_XY = (0.452, 0.177)
CUP_MASS = 0.162
CUP_YAW = 180.0  # its handle toward -x
LANE_Y = (CUP_XY[1] + CUP_R, BIN_XY[1] + BIN[1] / 2 - BIN_WALL)  # the room left in the bin: from the cup's side to the far wall
STOW_XY = (0.31, 0.29)  # where the demo's hand lets the can go, lying along x
BIN_IN = ((BIN_XY[0] - BIN[0] / 2 + BIN_WALL, BIN_XY[0] + BIN[0] / 2 - BIN_WALL), (BIN_XY[1] - BIN[1] / 2 + BIN_WALL, BIN_XY[1] + BIN[1] / 2 - BIN_WALL))


def bin_boxes() -> list:
    """The bin's floor and four walls, in its frame (origin on the table under its middle): (centre, size)."""
    lx, ly, h = BIN
    t = BIN_WALL
    return [((0.0, 0.0, t / 2), (lx, ly, t)),
            ((0.0, -(ly - t) / 2, h / 2), (lx, t, h)), ((0.0, (ly - t) / 2, h / 2), (lx, t, h)),
            ((-(lx - t) / 2, 0.0, h / 2), (t, ly - 2 * t, h)), (((lx - t) / 2, 0.0, h / 2), (t, ly - 2 * t, h))]


def in_bin(points) -> np.ndarray:
    """Which points (world xy, height above the table) are inside the bin's inner walls, under its rim."""
    p = np.atleast_2d(points)
    (x0, x1), (y0, y1) = BIN_IN
    return (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] < BIN[2] + 0.25)


def snap_push_n(friction: float = 0.4) -> float:
    """The push along the tab that takes the clip's catch over the tab's ridge: two round bumps, the strap lifting on its spring."""
    arm = CATCH_X - STRAP_X[0]
    rate = STRAP_RATE / arm**2  # N per metre of lift at the catch
    apart = (SLOT[1] + RIDGE_R - RIDGE_PROUD) - (TAB_T / 2 + RIDGE_PROUD - RIDGE_R)  # the two bumps' centres, square to the tab, at rest
    best = 0.0
    for lift in np.linspace(0.0, 2 * RIDGE_R - apart, 200)[:-1]:
        slope = math.asin(math.sqrt((2 * RIDGE_R) ** 2 - (apart + lift) ** 2) / (2 * RIDGE_R))
        best = max(best, rate * lift * math.tan(slope + math.atan(friction)))
    return best


def tab_press_n(deg: float) -> float:
    """The push on the tab's rear end, square to the tab, that holds it `deg` down."""
    return TAB_RATE * math.radians(deg + TAB_PRELOAD_DEG) / TAB_REACH


def tip_push_n(mass: float = CAN_MASS, at: float = 0.21) -> float:
    """The level push at height `at` that tips the free-standing can over its foot."""
    return mass * 9.81 * (FOOT[0] / 2) / at


LOOKS_RGB = {  # sRGB, 0..1
    "can": (0.93, 0.93, 0.94), "label": (0.16, 0.45, 0.78), "head": (0.05, 0.05, 0.06), "nozzle": (0.95, 0.80, 0.10),
    "clip": (0.96, 0.42, 0.06), "mat": (0.03, 0.03, 0.035), "bin": (0.80, 0.86, 0.90), "steel": (0.72, 0.73, 0.75), "stove": (0.55, 0.57, 0.60),
}
