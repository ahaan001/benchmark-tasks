"""Sizes and poses of everything in the scene, in metres. Pure Python and NumPy: read by the task file, the scorer,
the demo and the offline tests.

World frame: x forward from the line through the two arm bases, y to the robot's left of the midpoint between them,
z up. Heights here are above the table top unless a name ends in _Z_WORLD.

The wall is a section of a 2x4 stud wall framed at 24 in on centre: OSB sheathing, two studs, a fire block laid flat
between them, and a 1/2 in copper riser through the block, centred in the studs' depth. Lumber is at its dressed
size (1.5 x 3.5 in). Sizes marked est are read off the makers' product photos, scaled by a published dimension.
"""

from __future__ import annotations

import math

import numpy as np

IN = 0.0254

# -- wall section (all fixed) ------------------------------------------------------------------------------------------
STUD_T, STUD_D = 1.5 * IN, 3.5 * IN  # a dressed 2x4: 38.1 x 88.9 mm
OSB_T = 0.011  # 7/16 in sheathing
OSB_X = 0.545  # the sheathing's face
WALL_H = 0.50
STUD_FRONT_X = OSB_X - STUD_D
BAY = 22.5 * IN  # clear width between studs at 24 in on centre
STUD_R_Y1 = -0.280  # the right stud's inner face
STUD_L_Y0 = STUD_R_Y1 + BAY  # the left stud's inner face
WALL_Y0, WALL_Y1 = STUD_R_Y1 - STUD_T, STUD_L_Y0 + STUD_T
BLOCK_TOP = 0.140  # the fire block's top face
PIPE_XY = (OSB_X - STUD_D / 2, -0.100)  # the riser's axis: centred in the stud's depth
PIPE_D = 0.625 * IN  # 1/2 in nominal copper water tube: 5/8 in outside
COUPLING_D, COUPLING_L = 0.018, 0.025  # est
COUPLING_Z = 0.340  # the coupling's centre
GAP = STUD_D / 2 - PIPE_D / 2  # the riser's back to the sheathing: 36.5 mm

# -- heat shield: a 12 x 12 in blanket bent once to a right angle ------------------------------------------------------
SHEET = 12 * IN
WING = 0.090
PANEL_W = SHEET - WING
SHIELD_T = 0.010  # est: the maker gives no thickness
SHIELD_KG = 0.15  # est
# Body frame: origin at the panel's centre; x the panel's normal (its white face is at -x), y along the panel
# (the wing at +y), z up. The wing runs from the panel's +y end toward -x.
WING_OFF = np.array([-(SHIELD_T / 2 + WING / 2), PANEL_W / 2 - SHIELD_T / 2, 0.0])
SHIELD_START = np.array([0.403, 0.0925, SHEET / 2])  # the panel's centre, standing on the table
SHIELD_GOAL = np.array([OSB_X - GAP / 2, PIPE_XY[1], BLOCK_TOP + SHEET / 2])  # on the block, midway between the riser's back and the sheathing

# -- torch: a classic brass pencil-flame head on a 14.1 oz cylinder ----------------------------------------------------
CYL_D = 2.9 * IN  # 73.7 mm
CYL_WALL_H = 0.215  # the cylinder's straight wall (est); 27.0 cm to the top of its valve
TORCH_KG = 0.66  # empty cylinder about 0.46 kg, head about 0.20 kg (est)
TORCH_COM_UP = 0.181  # its centre of mass above the cylinder's base (est)
BEND_UP = 0.323  # where the burner tube bends, above the base (est)
TUBE_TILT = math.radians(26.0)  # the burner's axis, from the cylinder's axis (est)
TUBE_LEN = 0.078  # bend to burner mouth (est)
FLAME_LEN, FLAME_D = 0.050, 0.012  # the marker: the maker's 1.5 to 2 in inner flame
# Body frame: origin at the centre of the cylinder's straight wall; z along the cylinder's axis, x toward the burner side.
TORCH_ORIGIN_UP = CYL_WALL_H / 2
TORCH_COM = np.array([0.002, 0.0, TORCH_COM_UP - TORCH_ORIGIN_UP])
_D = np.array([math.sin(TUBE_TILT), 0.0, math.cos(TUBE_TILT)])  # the flame's direction in the body frame
MOUTH = np.array([0.0, 0.0, BEND_UP - TORCH_ORIGIN_UP]) + TUBE_LEN * _D
FLAME_TIP = MOUTH + FLAME_LEN * _D
FLAME_DIR = _D
TORCH_TOP_UP = 0.40  # overall height, about (est)
TORCH_HOME = (0.380, -0.170)  # the mark's centre
MARK_W = 0.100  # the mark: a white square of card
FOOT_PROUD = 0.0005  # the cylinder's foot: two crossed strips this far proud of its base (see the README's design notes)

# colours (sRGB)
OSB, SPRUCE, COPPER, FLUX = (0.80, 0.66, 0.42), (0.88, 0.78, 0.58), (0.78, 0.47, 0.22), (0.95, 0.95, 0.92)
WHITE, FOIL, BLUE, BRASS, BLACK, FLAME, MARK = (0.96, 0.96, 0.94), (0.72, 0.74, 0.78), (0.08, 0.28, 0.70), (0.80, 0.64, 0.22), (0.06, 0.06, 0.06), (0.25, 0.80, 1.00), (0.97, 0.97, 0.97)


def ry(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Fixed boxes: (name, size, centre above the table, colour)
WALL_BOXES = [
    ("sheathing", (OSB_T, WALL_Y1 - WALL_Y0, WALL_H), (OSB_X + OSB_T / 2, (WALL_Y0 + WALL_Y1) / 2, WALL_H / 2), OSB),
    ("stud_right", (STUD_D, STUD_T, WALL_H), (OSB_X - STUD_D / 2, STUD_R_Y1 - STUD_T / 2, WALL_H / 2), SPRUCE),
    ("stud_left", (STUD_D, STUD_T, WALL_H), (OSB_X - STUD_D / 2, STUD_L_Y0 + STUD_T / 2, WALL_H / 2), SPRUCE),
    ("fire_block", (STUD_D, BAY, STUD_T), (OSB_X - STUD_D / 2, (STUD_R_Y1 + STUD_L_Y0) / 2, BLOCK_TOP - STUD_T / 2), SPRUCE),
]

# The shield's plates in its body frame: (size, offset, colour). The first is the wing, which the simulator spawns as
# the body's base shape (so the pinched part has the bench objects' grip); the rest are added to it.
SHIELD_PARTS = [
    ((WING, SHIELD_T, SHEET), tuple(WING_OFF), WHITE),  # the wing
    ((SHIELD_T, PANEL_W, SHEET), (0.0, 0.0, 0.0), WHITE),  # the panel
    ((0.001, PANEL_W, SHEET), (SHIELD_T / 2 + 0.0005, 0.0, 0.0), FOIL),  # foil back of the panel
    ((WING, 0.001, SHEET), (WING_OFF[0], PANEL_W / 2 + 0.0005, 0.0), FOIL),  # foil back of the wing
    ((0.001, PANEL_W - 0.004, 0.030), (-(SHIELD_T / 2 + 0.0005), -0.002, SHEET / 2 - 0.015), FOIL),  # foil border, top
    ((0.001, PANEL_W - 0.004, 0.030), (-(SHIELD_T / 2 + 0.0005), -0.002, -(SHEET / 2 - 0.015)), FOIL),  # foil border, bottom
]

_VOLUMES = [float(np.prod(size)) for size, _, _ in SHIELD_PARTS]
SHIELD_COM = sum(v * np.array(offset) for v, (_, offset, _) in zip(_VOLUMES, SHIELD_PARTS)) / sum(_VOLUMES)  # one density throughout

# The torch's extra parts in its body frame: (size, offset, colour, rotation about y in degrees). A size of two numbers
# is a round part, (diameter, length) along its own z; a size of three is a box.
_Z = lambda up: up - TORCH_ORIGIN_UP  # noqa: E731
_BEND = np.array([0.0, 0.0, _Z(BEND_UP)])
TORCH_PARTS = [
    ((0.068, 0.026, 0.003), (0.0, 0.0, _Z(0.0015 - FOOT_PROUD)), BLUE, 0.0),  # foot, one strip
    ((0.026, 0.068, 0.003), (0.0, 0.0, _Z(0.0015 - FOOT_PROUD)), BLUE, 0.0),  # foot, the strip across it
    ((0.060, 0.012), (0.0, 0.0, _Z(0.221)), BLUE, 0.0),  # the cylinder's shoulder, lower step
    ((0.044, 0.013), (0.0, 0.0, _Z(0.2335)), BLUE, 0.0),  # the cylinder's shoulder, upper step
    ((0.032, 0.022), (0.0, 0.0, _Z(0.251)), BLACK, 0.0),  # valve collar
    ((0.031, 0.015), (0.0, 0.0, _Z(0.2625)), BRASS, 0.0),  # the head's base
    ((0.020, 0.053), (0.0, 0.0, _Z(0.2965)), BRASS, 0.0),  # valve body and the tube's upright part
    ((0.022, 0.018, 0.018), (-0.021, 0.0, _Z(0.292)), BLACK, 0.0),  # valve knob, at the back
    ((0.012, 0.040), tuple(_BEND + 0.019 * _D), BRASS, 26.0),  # the tube above the bend
    ((0.016, 0.040), tuple(_BEND + 0.058 * _D), BRASS, 26.0),  # burner tip
    ((FLAME_D, FLAME_LEN), tuple(MOUTH + FLAME_LEN / 2 * _D), FLAME, 26.0),  # the flame marker
]
