"""Every number of the bench, in one place: the mandrel and its rests, the wax ring and its claws, the stone, the tools
and where they stand. Metres, radians unless a name ends in _DEG. NumPy only.

Frames
- station: x forward from the arm bases, y left, z up; heights here are above the table top.
- mandrel frame M: origin on the stick's axis at the ring; +y along the stick toward its thick end, +z up through the
  ring's head, +x across. At the start M is the station frame moved to RING_XY at AXIS_H.
- stone frame: origin at the centre of the girdle plane, +z out of the table facet.
- a tool's frame: origin at its working end (tweezer tips, pusher face, pen tip), +z up the tool.
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001

# -- the mandrel (a 295 mm wooden ring stick, 23 mm tapering to 12 mm, 57 g) and what it lies on ------------------

RING_XY = (0.24, -0.12)
AXIS_H = 0.025
STICK_LEN = 0.295
STICK_R_THIN, STICK_R_THICK = 0.006, 0.0115
STICK_MASS = 0.057
RING_IN_R = 0.00865  # US size 7: 17.3 mm inside
RING_FROM_THIN = (RING_IN_R - STICK_R_THIN) / ((STICK_R_THICK - STICK_R_THIN) / STICK_LEN)  # where the ring stops on the taper
Y_THIN, Y_THICK = -RING_FROM_THIN, STICK_LEN - RING_FROM_THIN  # the stick's ends in M
V_Y = 0.030  # the V-block's middle: the ring overhangs it, so that pressing on the ring tips a stick nobody holds
REST_Y = 0.085  # the flat rest, toward the thick end
GRASP_Y = 0.133  # where the holding hand takes the stick
V_SIZE = (0.035, 0.010, 0.030)  # across, along the stick, tall
V_EDGE = V_Y - V_SIZE[1] / 2  # the edge of the V nearest the ring: what the stick turns about when the ring is pressed down
REST_SIZE = (0.040, 0.020, 0.0148)
WOOD_ON_STEEL = (0.35, 0.30)  # static, dynamic: an estimate


def stick_r(y: float) -> float:
    """The stick's radius at mandrel-frame y."""
    return STICK_R_THIN + (y - Y_THIN) * (STICK_R_THICK - STICK_R_THIN) / STICK_LEN


def stick_rods() -> tuple:
    """The stick's collider: five round rods along its axis, each as thick as the stick is where that rod works (its thin end, under the ring, in the
    V, on the rest, in the hand). Each is (from y, to y, radius), the ends being the centres of the rod's round ends."""
    hand = stick_r(GRASP_Y)
    return ((Y_THIN + STICK_R_THIN, -0.040, STICK_R_THIN), (-0.040, 0.004, stick_r(-0.040)), (0.010, 0.040, stick_r(V_Y)), (0.066, 0.100, stick_r(REST_Y)),
            (0.108, Y_THICK - hand, hand))


def _stick_com_y() -> float:
    """Where a solid tapered stick balances, in M."""
    a, b = STICK_R_THIN, STICK_R_THICK
    return Y_THIN + STICK_LEN / 4 * (a * a + 2 * a * b + 3 * b * b) / (a * a + a * b + b * b)


STICK_COM_Y = _stick_com_y()


# -- the wax ring: a solitaire with a four-claw head for an 8 mm round ---------------------------------------------

SHANK_T, SHANK_W = 0.0020, 0.0025
RING_OUT_R = RING_IN_R + SHANK_T
Z_S = 0.0147  # the seat line: the head's rim and the claws' hinges, above the stick's axis
CUP_Z0 = 0.0104  # the head's cup starts inside the shank
CUP_R0, CUP_R1 = 0.0036, 0.0050  # its outside radius at the bottom and at the rim: the claws stand 0.9 mm proud of the rim, so a tool on a claw is clear of the head
BEAR_R = 0.0033  # the fixed bearing: the stone's pavilion lands on this circle when it is home
N_CLAWS = 4
CLAW_AZ_DEG = (45.0, 135.0, 225.0, 315.0)  # about +z of M, from +x toward +y
CLAW_RH = 0.0041  # the claws' inside faces at the hinge: 0.1 mm clear of the girdle
CLAW_W = 0.0018  # square section
CLAW_H = 0.0020  # above the seat line
CLAW_LEAN_DEG = 14.0  # as moulded the claws stand open: 9.2 mm between opposite tips
CLAW_MAX_DEG = 75.0
SEAT_R = 0.0022  # the sliding seat's contact circle
SEAT_TRAVEL = 0.0015  # how far the stone stands proud before it is pressed home
SEAT_LIMIT = 0.0017
RING_MASS = 0.0013  # about 1.4 cm3 of wax at 0.93 g/cm3 (estimate)
# What the simulator touches inside the head: four blocks whose inside top edges are the bearing, four on the sliding seat, and a wall round them.
HEAD_WALL_R = 0.00445  # the wall's inside: clear of the stone, which the blocks alone carry
HEAD_WALL_H = 0.0008  # the wall is twelve thin blocks under the rim
BEAR_BLOCK = (HEAD_WALL_R - BEAR_R, 0.0024, 0.0020)  # along the radius, across it, tall
SEAT_BLOCK = (0.0011, 0.0018, 0.0030)

# -- the stone: an 8.0 mm round brilliant cubic zirconia --------------------------------------------------------------

STONE_D = 0.008
STONE_R = STONE_D / 2
TABLE_R = 0.57 * STONE_R
CROWN_H = 0.15 * STONE_D
GIRDLE_H = 0.03 * STONE_D
PAV_H = 0.43 * STONE_D
STONE_DENSITY = 5800.0  # kg/m3
STONE_PROFILE = ((0.0, -GIRDLE_H / 2 - PAV_H), (STONE_R, -GIRDLE_H / 2), (STONE_R, GIRDLE_H / 2), (TABLE_R, GIRDLE_H / 2 + CROWN_H),
                 (0.0, GIRDLE_H / 2 + CROWN_H))  # (radius, z), culet to table
PAV_SLOPE = PAV_H / STONE_R  # drop per metre of radius
STONE_VOLUME = math.pi * STONE_R ** 2 * PAV_H / 3 + math.pi * STONE_R ** 2 * GIRDLE_H + math.pi * CROWN_H / 3 * (STONE_R ** 2 + STONE_R * TABLE_R + TABLE_R ** 2)
STONE_MASS = STONE_VOLUME * STONE_DENSITY
TABLE_OVER_GIRDLE = GIRDLE_H / 2 + CROWN_H
STONE_COM_Z = -0.0003  # below the girdle plane: the pavilion outweighs the crown


def stone_hull() -> np.ndarray:
    """The stone's collider: 13 corners. An eight-sided girdle whose flats face along and across the mandrel and toward the four claws, at the stone's true
    radius; a point culet; a square table with its corners over the flats that face along and across. Along those flats the pavilion is the true cone, and
    toward the claws the crown is the true crown (from the girdle's lower edge). Every corner stands well out of the hull, so cooking keeps them all."""
    corner = STONE_R / math.cos(math.radians(22.5))
    girdle = [(corner * math.cos(math.radians(22.5 + 45.0 * k)), corner * math.sin(math.radians(22.5 + 45.0 * k)), -GIRDLE_H / 2) for k in range(8)]
    table = [(TABLE_R * math.sqrt(2.0) * math.cos(math.radians(90.0 * k)), TABLE_R * math.sqrt(2.0) * math.sin(math.radians(90.0 * k)), TABLE_OVER_GIRDLE) for k in range(4)]
    return np.array([(0.0, 0.0, -GIRDLE_H / 2 - PAV_H)] + girdle + table)


STONE_HULL_FACES = tuple([(0, 1 + (k + 1) % 8, 1 + k) for k in range(8)]
                         + [f for j in range(4) for f in ((1 + (2 * j - 1) % 8, 1 + 2 * j, 9 + j), (1 + 2 * j, 1 + (2 * j + 1) % 8, 9 + (j + 1) % 4), (1 + 2 * j, 9 + (j + 1) % 4, 9 + j))]
                         + [(9, 10, 11), (9, 11, 12)])


def pavilion_drop(radius: float) -> float:
    """How far below the girdle plane the pavilion's surface is at this radius."""
    return GIRDLE_H / 2 + (STONE_R - radius) * PAV_SLOPE


BEAR_Z = Z_S - pavilion_drop(BEAR_R)  # the bearing circle's height, so that a stone at home has its girdle on the seat line
RIM_Z = Z_S - GIRDLE_H / 2  # the head's flat rim
SEAT_TOP0 = Z_S + SEAT_TRAVEL - pavilion_drop(SEAT_R)  # the seat's contact height before it is pressed

# -- the tools --------------------------------------------------------------------------------------------------------

# diamond tweezers, 160 mm: a joined end and two sprung legs
TW_LEN = 0.160
TW_HINGE_Z = 0.135
TW_GAP = 0.016  # between the tips, open
TW_ROOT = 0.002  # between the legs at the hinge
TW_LEG_T = 0.0025  # the legs' thickness as they collide (real ones are about 1.2 mm)
TW_LEG_W = 0.006
TW_TIP_W = 0.002
TW_TIP_H = 0.012
TW_SPRING = 3.0  # N m/rad per leg: about 3 N a finger to close the tips on the stone (estimate)
TW_GRASP_Z = 0.105  # where the fingers squeeze: 55 mm of the tweezers stand above them, inside the gripper
TW_MASS = 0.025
TW_OPEN_DEG = math.degrees(math.atan((TW_GAP - TW_ROOT) / 2 / TW_HINGE_Z))

# setter's pusher: a flat square steel face in a hardwood mushroom handle, 95 mm
PU_LEN = 0.095
PU_FACE = 0.004
PU_SHANK = 0.063
PU_NECK_R, PU_NECK_H = 0.008, 0.012
PU_KNOB_R, PU_KNOB_H = 0.015, 0.012  # the knob's straight band, which the pads take
PU_GRASP_Z = PU_SHANK + PU_NECK_H + PU_KNOB_H / 2
PU_MASS = 0.020

# electric wax pen: a 155 mm handle and a straight round-point tip
PEN_LEN = 0.180
PEN_TIP_LEN = 0.025
PEN_TIP_R = 0.0015  # a round point 3 mm across: the arm places a tool's tip to about a millimetre sideways
PEN_HANDLE_R = 0.007
PEN_GRASP_Z = 0.125  # 55 mm of the handle stand above the fingers
PEN_MASS = 0.040
PEN_HOT = ((0.0, 0.0, PEN_TIP_R), (0.0, 0.0, PEN_TIP_R + 0.0015), (0.0, 0.0, PEN_TIP_R + 0.003))  # ball centres along the last 4 mm of the tip

# -- the stone's block, the tool block and the pen stand (station frame) ------------------------------------------

STONE_BLOCK_XY = (0.36, -0.10)
STONE_BLOCK = (0.040, 0.040, 0.010)
STONE_HOLE = 0.0045  # across, square
STONE_REST_H = STONE_BLOCK[2] + pavilion_drop(STONE_HOLE / 2)  # the girdle plane's height as the stone lies in its hole
# The three tools stand in a row along x. The working hand closes its fingers along x and cannot lift them over the pen or the tweezers where they stand,
# so it takes each from the ring's side of the row (+y) and leaves the same way: it never has to pass one tool to reach another.
TOOL_ROW_Y = -0.30
TOOL_BLOCK_XY = (0.33, TOOL_ROW_Y)
TOOL_BLOCK = (0.140, 0.040, 0.030)
TWEEZER_XY = (0.38, TOOL_ROW_Y)
TWEEZER_SLOT = (0.0224, 0.0026, 0.026)  # a trough: along x, across its floor, deep. The tips stand on its floor
TROUGH_RUN = (TW_LEG_W + 0.0006 - TWEEZER_SLOT[1]) / 2 / TW_TIP_H  # its long walls lean apart going up, by this much a side per unit of height: 0.3 mm off the legs where the tips end
TROUGH_TOP = TWEEZER_SLOT[1] + 2 * TROUGH_RUN * TWEEZER_SLOT[2]  # how wide it is at the block's top
TROUGH_BOARD = 0.002  # the walls are boards this thick
PUSHER_XY = (0.28, TOOL_ROW_Y)
PUSHER_HOLE = (0.005, 0.005, 0.022)
PEN_STAND_XY = (0.185, TOOL_ROW_Y)
PEN_BASE = (0.060, 0.060, 0.005)
PEN_TUBE = (0.024, 0.024, 0.042)  # outside, and its top's height
PEN_BORE = 0.0150
PEN_LEDGE_H = 0.031  # the handle's lower end rests here, its tip hanging 1 mm clear of the base
PEN_TIP_BORE = 0.009
# The pusher's hole and the pen's stand have a funnel mouth of plates at 45 degrees all round, and the trough has one at each end, so that a tool a few
# millimetres off slides in.
MOUTH_W, MOUTH_T = 0.008, 0.0015
MOUTH_RISE = MOUTH_W * math.sqrt(0.5)  # how far a mouth stands above its opening
MOUTH_OFF = (MOUTH_W / 2 + MOUTH_T / 2) * math.sqrt(0.5)  # a plate's centre, out from the opening's edge
MOUTH_UP = (MOUTH_W / 2 - MOUTH_T / 2) * math.sqrt(0.5)  # and up from the opening
# Where each tool's working end rests (heights above the table): the tweezers' tips on the trough's floor, the pusher on the bottom of its hole, the
# pen's handle on the ledge in its stand.
TW_REST_Z = TOOL_BLOCK[2] - TWEEZER_SLOT[2]
PU_REST_Z = TOOL_BLOCK[2] - PUSHER_HOLE[2]
PEN_REST_Z = PEN_LEDGE_H - PEN_TIP_LEN

# -- geometry the scorer and the demo share -------------------------------------------------------------------------------


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def claw_frame(i: int) -> tuple[np.ndarray, np.ndarray]:
    """Claw i's hinge point in M, and its axes as columns: outward radial, the hinge axis (positive closes), up."""
    az = math.radians(CLAW_AZ_DEG[i])
    out = np.array([math.cos(az), math.sin(az), 0.0])
    axis = np.array([math.sin(az), -math.cos(az), 0.0])  # turning about it by a positive angle tips the claw inward
    return CLAW_RH * out + np.array([0.0, 0.0, Z_S]), np.column_stack([out, axis, [0.0, 0.0, 1.0]])


def claw_box(i: int, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Claw i's box at joint `angle` (rad, 0 as moulded): centre, rotation (columns: its outward, sideways and long axes) and half sizes, in M."""
    hinge, frame = claw_frame(i)
    tilt = angle - math.radians(CLAW_LEAN_DEG)  # inward from upright
    c, s = math.cos(tilt), math.sin(tilt)
    long = -s * frame[:, 0] + c * frame[:, 2]
    outward = c * frame[:, 0] + s * frame[:, 2]
    centre = hinge + outward * CLAW_W / 2 + long * CLAW_H / 2
    return centre, np.column_stack([outward, frame[:, 1], long]), np.array([CLAW_W / 2, CLAW_W / 2, CLAW_H / 2])


def claw_overlap(angle: float) -> float:
    """How far claw's inside top edge lies over the stone's edge, at joint `angle` (rad). Negative while it stands clear."""
    return STONE_R - (CLAW_RH - CLAW_H * math.sin(angle - math.radians(CLAW_LEAN_DEG)))


def claw_angle_for(overlap: float) -> float:
    """The joint angle (rad) at which the claw's tip is `overlap` over the stone's edge."""
    return math.radians(CLAW_LEAN_DEG) + math.asin((CLAW_RH - STONE_R + overlap) / CLAW_H)


def claw_on_crown() -> float:
    """The joint angle (rad) at which a claw's inside top edge meets the crown of a stone at home."""
    slope = CROWN_H / (STONE_R - TABLE_R)
    lean = math.radians(CLAW_LEAN_DEG)
    for k in range(0, 9000):
        a = math.radians(k / 100.0)
        r, h = CLAW_RH - CLAW_H * math.sin(a - lean), CLAW_H * math.cos(a - lean)
        if r <= STONE_R and h <= GIRDLE_H / 2 + (STONE_R - r) * slope:
            return a
    return math.radians(CLAW_MAX_DEG)


def box_gap(p, centre, rot, half) -> float:
    """Distance from point p to a box (0 inside)."""
    q = np.abs(rot.T @ (np.asarray(p, float) - centre)) - half
    return float(np.linalg.norm(np.maximum(q, 0.0)))


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def profile_gap(radius: float, z: float, profile) -> float:
    """Distance from the point (radius, z) to a solid of revolution whose section is the convex polygon `profile` of
    (radius, z) corners, given counter-clockwise. 0 inside."""
    pts = [np.array(q, float) for q in profile]
    p = np.array([radius, z], float)
    inside = True
    best = math.inf
    for a, b in zip(pts, pts[1:] + pts[:1]):
        e, w = b - a, p - a
        if e[0] * w[1] - e[1] * w[0] < 0.0:
            inside = False
        best = min(best, _seg_dist(p, a, b))
    return 0.0 if inside else best


STONE_SECTION = tuple(STONE_PROFILE)  # counter-clockwise in (radius, z)
CUP_SECTION = ((BEAR_R, CUP_Z0), (CUP_R0, CUP_Z0), (CUP_R1, RIM_Z), (STONE_R, RIM_Z), (BEAR_R, BEAR_Z))
SHANK_SECTION = ((RING_IN_R, -SHANK_W / 2), (RING_OUT_R, -SHANK_W / 2), (RING_OUT_R, SHANK_W / 2), (RING_IN_R, SHANK_W / 2))  # (radius about the stick, y)


def stone_gap(p_stone) -> float:
    """Distance from a point, given in the stone's frame, to the stone."""
    p = np.asarray(p_stone, float)
    return profile_gap(float(math.hypot(p[0], p[1])), float(p[2]), STONE_SECTION)


def wax_body_gap(p_m) -> float:
    """Distance from a point in M to the ring's wax other than the claws: the shank and the head's cup."""
    p = np.asarray(p_m, float)
    cup = profile_gap(float(math.hypot(p[0], p[1])), float(p[2]), CUP_SECTION)
    shank = profile_gap(float(math.hypot(p[0], p[2])), float(p[1]), SHANK_SECTION)
    return min(cup, shank)


def wax_gap(p_m, claw_angles) -> float:
    """Distance from a point in M to any wax of the ring, claws included."""
    return min(wax_body_gap(p_m), min(box_gap(p_m, *claw_box(i, a)) for i, a in enumerate(claw_angles)))


def stone_in_head(p_m, axis_m) -> tuple[float, float, float]:
    """A stone's place in the head, from its origin and its +z axis in M: (how far off the head's axis, how far its
    girdle plane stands above the seat line, its tilt from the head's axis in degrees)."""
    p = np.asarray(p_m, float)
    tilt = math.degrees(math.acos(float(np.clip(np.asarray(axis_m, float)[2], -1.0, 1.0))))
    return float(math.hypot(p[0], p[1])), float(p[2] - Z_S), tilt


CLAW_CLOSED = claw_angle_for(0.001)  # 1 mm over the stone's edge
CLAW_ON_CROWN = claw_on_crown()
CLAW_STOP_DEG = math.degrees(CLAW_ON_CROWN) + 0.6  # where a claw's hinge ends: just past lying on a seated stone, so that it cannot be driven into the stone
