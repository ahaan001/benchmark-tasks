"""Sizes, shapes and places of everything in the task, in metres. Pure NumPy, no simulator.

Station frame: x forward from the line through the two arm bases, y to the robot's left, z up from the table top.
Published sizes are from the makers' catalogues (README, "Where the sizes come from"); a size marked (est) is not published and is our estimate.
"""

from __future__ import annotations

import math

import numpy as np

# -- 80/20 40-4040 profile (80/20 catalog p. 158) ---------------------------------------------------------------------------------------
BAR = 0.040
SLOT_OPEN = 0.00813
LIP = 0.00432
CAVITY_W = 0.02057
SLOT_DEPTH = 0.01232
CAVITY_H = SLOT_DEPTH - LIP  # 8.00 mm
WALL_UP = 0.0027  # (est, scaled off the catalog drawing) the upright wall under each lip; below it the cavity's walls slope at 45 degrees
FLOOR_W = CAVITY_W - 2 * (CAVITY_H - WALL_UP)  # 9.97 mm

# -- the frame: a closed rectangle, 360 x 300 mm outside, flat on the table --------------------------------------------------------------
CROSS_X = (0.10, 0.42)  # centre lines of the two cross bars, which run along y
SIDE_Y = (-0.08, 0.18)  # centre lines of the two side bars, which run along x between the cross bars
FRAME_Y = (SIDE_Y[0] - BAR / 2, SIDE_Y[1] + BAR / 2)
SIDE_SPAN = (CROSS_X[0] + BAR / 2, CROSS_X[1] - BAR / 2)
MARKED_Y = SIDE_Y[0]  # the right side bar carries the line
LINE_X, LINE_W = 0.24, 0.003
CORNER = (0.040, 0.006, 0.036)  # 80/20 40-4302 inside corner bracket: leg length, thickness, width

# -- the camera bracket (our design; drawing and model in the task folder) ---------------------------------------------------------------
# Its own frame: origin at the middle of the foot's underside, x along the foot's long side, y along the arm, z up.
FOOT = (0.080, 0.040, 0.0064)
HOLE_D, SCREW_DX = 0.0060, 0.030  # the holes are a light push fit on the M6 screws: they stay where they are put and have no play
NOTCH = 0.003  # a V this deep at the middle of the long edge away from the arm
ARM_W, ARM_H, ARM_END = 0.020, 0.012, 0.200
PLATE = (0.004, 0.042, 0.042)  # the side plate at the arm's end that the camera's back screws to
CAM = (0.023, 0.042, 0.042)  # Intel RealSense D405 (datasheet): depth, width, height as it hangs here
CAM_DROP = 0.030  # the plate and camera reach this far below the arm's underside; their tops are level with the arm's
PETG = 1270.0  # kg/m3
CAM_MASS, SCREW_MASS, NUT_MASS = 0.060, 0.0065, 0.0027  # D405 datasheet; M6 x 16 cap screw (est); 80/20 3931 (catalog: .006 lb)
GRIP_Y = 0.118  # where a hand holds the arm, from the foot's centre: near the centre of mass, clear of the camera

# -- M6 x 16 socket head cap screw (ISO 4762) and the 80/20 3931 drop-in T-nut (80/20 catalog p. 3-172) ----------------------------------
SCREW_LEN, HEAD_D, HEAD_H = 0.016, 0.010, 0.006
SOCKET_AF, SOCKET_DEPTH = 0.0051, 0.0035  # 5 mm hex socket: across flats (within ISO 4762's tolerance) and depth (3 mm at least)
PITCH = 0.001
NUT_A, NUT_B, NUT_F, NUT_E = 0.01575, 0.00775, 0.00457, 0.0015  # flange length and width, flange thickness, boss height
BOSS_C = 0.00782  # the boss's length along the flange
NUT_G = NUT_F + NUT_E  # 6.07 mm overall
NUT_PLAY = 0.0  # the nuts collide at their real width: the screws have no play in the foot's holes, and neither have the hinges here
NUT_CUT_R = 0.0036  # (est, from the catalog drawing) radius of the two cut-back corners of the boss
FLANGE_CUT_R = 0.0030  # (est) and of the flange
NUT_BEVEL_FROM = 0.0011  # (est, from the catalog drawing) the flange's ends are bevelled at 45 degrees from this far below its top face
SCREW_END = SCREW_LEN - FOOT[2]  # the screw's end below the foot's underside: 9.6 mm
NUT_TOP = SCREW_END - NUT_F  # the flange's top face below the foot's underside at the start (screw end flush with the nut's underside): 5.03 mm
NUT_DROP = NUT_TOP - LIP  # the flange's top below the lips' underside once the foot is on the bar: 0.71 mm, with 0.79 mm of the boss still in the slot's opening
TRAVEL = (-0.0030, NUT_TOP - NUT_E)  # how far a nut may run down and up its screw from there: up until its boss meets the foot

# -- Wera 454 HF T-handle hex driver 5 x 100 mm (Wera size table and specification sheet) ------------------------------------------------
HEX_AF = 0.00498
BLADE = 0.100
TIP_CHAMFER = (0.0015, 0.0010)  # the blade's tip: this long, cut back this far (a real tip's chamfer is about 0.3 mm: stated in the README)
TIP_CAPTURE = (SOCKET_AF - (HEX_AF - 2 * TIP_CHAMFER[1])) / 2  # how far off a socket's axis the tip may come down and still be led in: 1.06 mm
NECK = (0.030, 0.022, 0.021)  # the handle's lower part round the blade (est split of the 49 mm tall handle)
HANDLE = (0.099, 0.022, 0.028)
SHORT_ARM = 0.020  # the key's short arm, out of one end of the handle
DRIVER_MASS = 0.079
DRIVER_COM = 0.111  # above the tip (est: 28 g of steel key, 51 g of handle)
GRIP_UP = 0.012  # a hand takes the neck with its TCP this far above the neck's underside
TIP_TO_TCP = BLADE + GRIP_UP

# -- the driver's block, the bracket's supports ------------------------------------------------------------------------------------------
BLOCK = (0.060, 0.060, 0.040)
BLOCK_XY = (0.20, -0.25)
HOLE, HOLE_DEPTH = 0.007, 0.035
SUPPORT_H = 0.035  # EVA foam, so that the nuts and the camera hang free
START_XY, START_YAW = (0.33, 0.29), 90.0  # the foot's centre at the start, arm pointing toward the robot
STRIPS = (((0.0, 0.014), (0.080, 0.012)), ((0.0, -0.014), (0.080, 0.012)), ((-0.002, 0.185), (0.014, 0.024)))  # (centre, size) in the bracket's frame: two under the foot, one under the arm's end, beyond the centre of mass
MOUNT_XY, MOUNT_YAW = (LINE_X, MARKED_Y), 0.0  # where the foot's centre belongs, arm pointing left over the inside of the frame

LOOKS_RGB = {"aluminium": (0.78, 0.79, 0.80), "slot": (0.30, 0.31, 0.33), "zinc": (0.72, 0.73, 0.75), "petg": (0.56, 0.58, 0.60), "black": (0.05, 0.05, 0.06),
             "camera": (0.16, 0.17, 0.19), "lens": (0.02, 0.02, 0.03), "line": (0.05, 0.25, 0.85), "beech": (0.80, 0.63, 0.42), "foam": (0.10, 0.10, 0.11),
             "handle": (0.12, 0.13, 0.13), "green": (0.45, 0.70, 0.10), "steel": (0.62, 0.63, 0.65)}


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# -- the profile's cross-section as boxes -------------------------------------------------------------------------------------------------
def section_boxes() -> list:
    """One slotted bar's cross-section as (name, centre (u, z), size (du, dz), slope in degrees): u across the bar from its middle, z up from the
    table. A sloped piece is a slab turned about the bar's axis; its upper face is the cavity's sloping wall."""
    out = [("core", (0.0, (BAR - SLOT_DEPTH) / 2), (BAR, BAR - SLOT_DEPTH), 0.0)]
    t = 0.006
    for s, side in ((1.0, "p"), (-1.0, "n")):
        out.append((f"wall_{side}", (s * (CAVITY_W / 2 + BAR / 2) / 2, BAR - LIP - CAVITY_H / 2), ((BAR - CAVITY_W) / 2, CAVITY_H), 0.0))
        out.append((f"lip_{side}", (s * (SLOT_OPEN / 2 + BAR / 2) / 2, BAR - LIP / 2), ((BAR - SLOT_OPEN) / 2, LIP), 0.0))
        p1 = np.array([s * CAVITY_W / 2, BAR - LIP - WALL_UP])
        p2 = np.array([s * FLOOR_W / 2, BAR - SLOT_DEPTH])
        along = (p2 - p1) / np.linalg.norm(p2 - p1)
        into = np.array([s, -1.0]) / math.sqrt(2.0)  # out of the cavity, into the metal
        centre = (p1 + p2) / 2 + into * t / 2
        out.append((f"slope_{side}", tuple(centre), (float(np.linalg.norm(p2 - p1)) + 0.004, t), math.degrees(math.atan2(along[1], along[0]))))
    return out


def bar_boxes(axis: str, centre_xy, span) -> list:
    """A bar's boxes in the station frame: (name, centre, size, rotation matrix or None). `axis` is "x" or "y"; `span` its two ends along that axis."""
    length, mid = span[1] - span[0], (span[0] + span[1]) / 2
    out = []
    for name, (u, z), (du, dz), slope in section_boxes():
        if axis == "x":
            centre, size = (mid, centre_xy[1] + u, z), (length, du, dz)
        else:
            centre, size = (centre_xy[0] + u, mid, z), (du, length, dz)
        rot = None
        if slope:
            c, s = math.cos(math.radians(slope)), math.sin(math.radians(slope))
            if axis == "x":  # the slab's own axes: along the bar, along the slope, and its normal
                rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            else:
                rot = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
                if np.linalg.det(rot) < 0:
                    rot[:, 2] *= -1.0
        out.append((name, centre, size, rot))
    return out


def frame_bars() -> list:
    """(bar name, axis, centre line, span) of the four bars."""
    return [("near", "y", (CROSS_X[0], 0.0), FRAME_Y), ("far", "y", (CROSS_X[1], 0.0), FRAME_Y),
            ("right", "x", (0.0, SIDE_Y[0]), SIDE_SPAN), ("left", "x", (0.0, SIDE_Y[1]), SIDE_SPAN)]


def corner_boxes() -> list:
    """The four inside corner brackets as (centre, size): one leg on each bar's inner face."""
    leg, t, w = CORNER
    out = []
    for cx, sx in ((SIDE_SPAN[0], 1.0), (SIDE_SPAN[1], -1.0)):
        for cy, sy in ((SIDE_Y[0] + BAR / 2, 1.0), (SIDE_Y[1] - BAR / 2, -1.0)):
            out.append(((cx + sx * leg / 2, cy + sy * t / 2, BAR / 2), (leg, t, w)))
            out.append(((cx + sx * t / 2, cy + sy * leg / 2, BAR / 2), (t, leg, w)))
    return out


def cavity_half_width(depth: float) -> float:
    """Half the cavity's width at a depth below the lips' underside (m): the upright wall, then the 45 degree slope."""
    return CAVITY_W / 2 - max(0.0, depth - WALL_UP)


# -- the bracket ---------------------------------------------------------------------------------------------------------------------------
def bracket_boxes() -> list:
    """The printed bracket and its camera as (name, centre, size, look) in the bracket's frame."""
    arm_len = ARM_END - FOOT[1] / 2
    plate_c = (ARM_W / 2 + PLATE[0] / 2, ARM_END - PLATE[1] / 2, ARM_H - PLATE[2] / 2)
    cam_c = (ARM_W / 2 + PLATE[0] + CAM[0] / 2, ARM_END - CAM[1] / 2, ARM_H - CAM[2] / 2)
    return [("foot", (0.0, 0.0, FOOT[2] / 2), FOOT, "petg"), ("arm", (0.0, FOOT[1] / 2 + arm_len / 2, ARM_H / 2), (ARM_W, arm_len, ARM_H), "petg"),
            ("plate", plate_c, PLATE, "petg"), ("camera", cam_c, CAM, "camera")]


def bracket_mass() -> tuple:
    """(mass, centre of mass, principal inertia about it along the bracket's axes) of the bracket with its camera, screws and nuts."""
    items = []
    for name, c, s, _ in bracket_boxes():
        mass = CAM_MASS if name == "camera" else PETG * s[0] * s[1] * s[2]
        items.append((mass, np.array(c), np.array(s)))
    for sx in (-1.0, 1.0):
        items.append((SCREW_MASS + NUT_MASS, np.array([sx * SCREW_DX, 0.0, 0.0]), np.array([0.006, 0.006, 0.016])))
    total = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / total
    inertia = np.zeros(3)
    for m, c, s in items:
        d = c - com
        inertia += m / 12.0 * np.array([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])
        inertia += m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return float(total), com, inertia


def foot_outline() -> np.ndarray:
    """The foot in plan, with its V notch, anticlockwise from the corner at -x, -y."""
    hx, hy = FOOT[0] / 2, FOOT[1] / 2
    return np.array([(-hx, -hy), (-NOTCH, -hy), (0.0, -hy + NOTCH), (NOTCH, -hy), (hx, -hy), (hx, hy), (-hx, hy)])


FOOT_CORNERS = np.array([[sx * FOOT[0] / 2, sy * FOOT[1] / 2, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
NUT_LOW = np.array([[sc + sx * (NUT_A / 2 - (NUT_F - NUT_BEVEL_FROM)), sy * (NUT_B - NUT_PLAY) / 2, -SCREW_END] for sc in (-SCREW_DX, SCREW_DX) for sx in (-1, 1) for sy in (-1, 1)])  # the corners of both nuts' undersides
SCREWS = np.array([[-SCREW_DX, 0.0, 0.0], [SCREW_DX, 0.0, 0.0]])  # screw a and screw b, on the foot's underside
NOTCH_POINT = np.array([0.0, -FOOT[1] / 2, 0.0])
SOCKET_FLOOR = FOOT[2] + HEAD_H - SOCKET_DEPTH  # above the foot's underside
GRIP_POINT = np.array([0.0, GRIP_Y, ARM_H / 2])


# -- the nut ---------------------------------------------------------------------------------------------------------------------------------
def nut_outline(length: float, width: float, cut: float, small: float = 0.0003, n: int = 6) -> np.ndarray:
    """A nut's plan outline, long axis along x, anticlockwise. The corners at (+x, -y) and (-x, +y) are cut back by radius `cut`: they lead
    when the nut turns clockwise, seen from above, as a tightening screw carries it. The other two are left nearly square and stop it."""
    hx, hy = length / 2, width / 2
    pts = []
    for sx, sy, r, a0 in ((1, 1, small, 0.0), (-1, 1, cut, 0.5 * math.pi), (-1, -1, small, math.pi), (1, -1, cut, 1.5 * math.pi)):
        r = min(r, hx, hy)
        cx, cy = sx * (hx - r), sy * (hy - r)
        for a in np.linspace(a0, a0 + 0.5 * math.pi, n if r > 0.001 else 2):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def boss_outline(collide: bool = True) -> np.ndarray:
    return nut_outline(BOSS_C, NUT_B - (NUT_PLAY if collide else 0.0), NUT_CUT_R)


def flange_outline(collide: bool = True) -> np.ndarray:
    return nut_outline(NUT_A, NUT_B - (NUT_PLAY if collide else 0.0), FLANGE_CUT_R)


def flange_points(collide: bool = True) -> np.ndarray:
    """The flange's corners, z from its top face down: the full outline for the first 1.1 mm, then bevelled ends down to a shorter underside."""
    top = flange_outline(collide)
    cut = NUT_F - NUT_BEVEL_FROM
    hx, hy = NUT_A / 2 - cut, (NUT_B - (NUT_PLAY if collide else 0.0)) / 2
    bottom = np.array([(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)])
    return np.vstack([np.c_[top, np.zeros(len(top))], np.c_[top, np.full(len(top), -NUT_BEVEL_FROM)], np.c_[bottom, np.full(4, -NUT_F)]])


def width_across(outline: np.ndarray, turn_cw_deg: float) -> float:
    """How wide an outline is across the slot once turned clockwise by an angle from lengthwise."""
    a = -math.radians(turn_cw_deg)
    y = outline[:, 0] * math.sin(a) + outline[:, 1] * math.cos(a)
    return float(y.max() - y.min())


def boss_stops() -> tuple:
    """(clockwise, anticlockwise) angles in degrees at which the boss, turning in the slot's opening from lengthwise, is stopped by the lips."""
    boss = boss_outline()
    out = []
    for sense in (1.0, -1.0):
        out.append(next((d for d in np.arange(0.0, 180.0, 0.1) if width_across(boss, sense * d) > SLOT_OPEN), 180.0))
    return float(out[0]), float(out[1])


def wing_reach(turn_deg: float, off_centre: float = 0.0) -> tuple:
    """How far the middle of each end of the flange lies under its lip (m, negative: not under it) for a nut turned from lengthwise and off the
    slot's middle. A corner reaches further; the middle of the end is what carries."""
    half = abs(NUT_A / 2 * math.sin(math.radians(turn_deg)))
    return half + off_centre - SLOT_OPEN / 2, half - off_centre - SLOT_OPEN / 2


# -- the driver ------------------------------------------------------------------------------------------------------------------------------
def hex_ring(across_flats: float, z: float) -> np.ndarray:
    r = across_flats / 2 / math.cos(math.pi / 6)
    return np.array([(r * math.cos(k * math.pi / 3), r * math.sin(k * math.pi / 3), z) for k in range(6)])  # corners on +-x: flats face +-y


def blade_points() -> np.ndarray:
    """The blade's corners, in the driver's frame (origin at the tip, z up the blade): a hexagon with a chamfered tip."""
    return np.vstack([hex_ring(HEX_AF - 2 * TIP_CHAMFER[1], 0.0), hex_ring(HEX_AF, TIP_CHAMFER[0]), hex_ring(HEX_AF, BLADE + 0.001)])


def driver_boxes() -> list:
    """The handle as (name, centre, size, look) in the driver's frame: the bar lies along x, the short arm sticks out of its -x end."""
    return [("neck", (0.0, 0.0, BLADE + NECK[2] / 2), NECK, "handle"), ("bar", (0.0, 0.0, BLADE + NECK[2] + HANDLE[2] / 2), HANDLE, "handle"),
            ("short_arm", (-(HANDLE[0] + SHORT_ARM) / 2, 0.0, BLADE + NECK[2] + HANDLE[2] / 2), (SHORT_ARM, 0.005, 0.005), "steel")]


def socket_walls(n: int = 6, across_flats: float = SOCKET_AF) -> list:
    """The screw head round its hex socket as n slabs: (centre (x, y), size (radial, tangential), turn in degrees) in the head's plan."""
    inner, outer = across_flats / 2, HEAD_D / 2
    out = []
    for k in range(n):
        a = math.pi / 2 + k * 2 * math.pi / n  # flats face +-y, as the blade's
        mid = (inner + outer) / 2
        out.append(((mid * math.cos(a), mid * math.sin(a)), (outer - inner, 2 * outer * math.tan(math.pi / n) * 0.9), math.degrees(a)))
    return out


def block_boxes() -> list:
    """The beech block round its hole as (centre, size) from the block's middle on the table: a floor under the hole and four walls."""
    bx, by, bz = BLOCK
    floor = bz - HOLE_DEPTH
    side = (bx - HOLE) / 2
    z, h = floor + HOLE_DEPTH / 2, HOLE_DEPTH
    return [((0.0, 0.0, floor / 2), (bx, by, floor)), (((HOLE + side) / 2, 0.0, z), (side, by, h)), ((-(HOLE + side) / 2, 0.0, z), (side, by, h)),
            ((0.0, (HOLE + side) / 2, z), (HOLE, side, h)), ((0.0, -(HOLE + side) / 2, z), (HOLE, side, h))]


DRIVER_START = np.array([BLOCK_XY[0], BLOCK_XY[1], BLOCK[2] - HOLE_DEPTH])  # the blade's tip, standing in the block


# -- places ------------------------------------------------------------------------------------------------------------------------------------
def bracket_pose(xy, yaw_deg: float, z: float) -> tuple:
    return np.array([xy[0], xy[1], z]), rot_z(yaw_deg)


START_POSE = bracket_pose(START_XY, START_YAW, SUPPORT_H)
MOUNT_POSE = bracket_pose(MOUNT_XY, MOUNT_YAW, BAR)


def strip_boxes() -> list:
    """The three foam supports as (centre, size) in the station frame."""
    pos, rot = START_POSE
    out = []
    for (cx, cy), (sx, sy) in STRIPS:
        c = rot @ np.array([cx, cy, 0.0]) + np.array([pos[0], pos[1], 0.0])
        size = np.abs(rot @ np.array([sx, sy, 0.0]))
        out.append(((float(c[0]), float(c[1]), SUPPORT_H / 2), (float(size[0]), float(size[1]), SUPPORT_H)))
    return out


def box_distance(points: np.ndarray, centre, size) -> np.ndarray:
    """Distance from points to a box given in the same frame (0 inside it)."""
    q = np.abs(np.atleast_2d(points) - np.asarray(centre, float)) - np.asarray(size, float) / 2
    return np.linalg.norm(np.maximum(q, 0.0), axis=1)


def bracket_distance(local_points: np.ndarray) -> float:
    """The least distance from points, given in the bracket's frame, to the bracket: its print, its camera and its two screw heads."""
    pts = np.atleast_2d(local_points)
    best = min(float(box_distance(pts, c, s).min()) for _, c, s, _ in bracket_boxes())
    for sc in SCREWS:
        best = min(best, float(box_distance(pts, sc + [0.0, 0.0, FOOT[2] + HEAD_H / 2], (HEAD_D, HEAD_D, HEAD_H)).min()))
    return best


def driver_points() -> np.ndarray:
    """Points over the driver, in its own frame, for clearances: along the blade and round the handle."""
    pts = [(0.0, 0.0, z) for z in np.linspace(0.0, BLADE, 11)]
    for _, c, s, _ in driver_boxes():
        pts += [(c[0] + sx * s[0] / 2, c[1] + sy * s[1] / 2, c[2] + sz * s[2] / 2) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return np.array(pts)
