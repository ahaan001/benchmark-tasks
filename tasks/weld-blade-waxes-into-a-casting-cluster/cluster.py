"""The cluster, the blade patterns and the wax knife as numbers: sizes, poses and the boxes the weld rule and the colliders share. NumPy only.

Station frame: metres from the midpoint between the arm bases, x forward, y left, z up from the table top.
A pattern's own frame: origin at the centre of its gate's end face, +y out of that face (into the sprue once welded), z up as it lies and hangs.
The knife's own frame: origin at the centre of its blade, +x along the handle, the blade's flat faces normal to y.
Sizes marked (E) are estimates; the printable master in assets/ is drawn from these same numbers, so the real and simulated parts agree.
"""

from __future__ import annotations

import math

import numpy as np

# -- the cluster
CX, CY = 0.25, 0.06  # the sprue's axis
SPRUE = 0.025  # square (E)
STAND = (0.12, 0.12, 0.008)  # steel plate
CUP_R, CUP_H = 0.025, 0.025  # pour cup, mouth down on the stand's pin
SPRUE_Z0 = STAND[2] + CUP_H
SPRUE_Z1 = SPRUE_Z0 + 0.150
FACE_Y = CY - SPRUE / 2  # the face the slots are on, toward the tray (-y)
SLOTS_Z = (0.075, 0.110, 0.145)  # gate-centre heights of the three empty marks: 35 mm pitch, a 22 mm platform leaves 13 mm between patterns
SLOT_FROM = (-1, -1, -1)  # the knife comes in from -x at every slot: neighbours stand on the +x face at the lower two

# -- a blade pattern (E): gate, fir-tree root, platform, airfoil
GATE = (0.012, 0.045, 0.012)  # x, y, z
ROOT = (0.018, 0.015, 0.014)
PLATFORM = (0.030, 0.004, 0.022)
SPAN, CHORD, THICK, TWIST = 0.050, 0.024, 0.006, 20.0
N_AIR = 5
PATTERN_MASS = 0.017  # blue injection wax, about 0.97 g/cm^3
Y_ROOT = -GATE[1] - ROOT[1] / 2
Y_PLATFORM = -GATE[1] - ROOT[1] - PLATFORM[1] / 2
Y_AIR0 = -GATE[1] - ROOT[1] - PLATFORM[1]
LENGTH = -(Y_AIR0 - SPAN)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def pattern_boxes() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(name, centre, half sizes, rotation) of every box of a pattern, in its own frame. The airfoil is five boxes, each turned by its share of the twist."""
    eye = np.eye(3)
    out = [("gate", np.array([0.0, -GATE[1] / 2, 0.0]), np.array(GATE) / 2, eye),
           ("root", np.array([0.0, Y_ROOT, 0.0]), np.array(ROOT) / 2, eye),
           ("platform", np.array([0.0, Y_PLATFORM, 0.0]), np.array(PLATFORM) / 2, eye)]
    seg = SPAN / N_AIR
    for i in range(N_AIR):
        tw = TWIST * ((i + 0.5) / N_AIR - 0.5)
        out.append((f"air{i}", np.array([0.0, Y_AIR0 - (i + 0.5) * seg, 0.0]), np.array([CHORD / 2, seg / 2, THICK / 2]), rot_y(tw)))
    return out


BOXES = pattern_boxes()

# -- where the patterns are: loose in the tray, welded at the slots, and the three already on the cluster
TRAY_X = (0.20, 0.25, 0.30)
TRAY_Y = -0.17  # the gate end faces, pointing +y at the sprue
TRAY_TOP = 0.012  # the foam slab
LIE_Z = TRAY_TOP + PLATFORM[2] / 2 + 0.002  # a pattern's origin height as it lies: its platform 2 mm clear of the slab
TRAY_BARS = (("gate_bar", -GATE[1] / 2, GATE[2] / 2), ("air_bar", Y_AIR0 - SPAN / 2, 0.0052))  # none under the root and platform: the fingers go down either side of the platform  # (name, y in pattern frame, depth below the origin)


def loose_start(i: int) -> tuple[np.ndarray, np.ndarray]:
    """(R, p) of loose pattern i as it lies in the tray."""
    return np.eye(3), np.array([TRAY_X[i], TRAY_Y, LIE_Z])


def slot_pose(k: int) -> tuple[np.ndarray, np.ndarray]:
    """(R, p) of a pattern welded square on mark k: gate end face on the sprue's face, centred on the mark."""
    return np.eye(3), np.array([CX, FACE_Y, SLOTS_Z[k]])


NEIGHBOURS = (("below", np.array([CX, FACE_Y, 0.040]), rot_z(0.0)),  # under the lowest mark, same face
              ("side_0", np.array([CX + SPRUE / 2, CY, SLOTS_Z[0]]), rot_z(90.0)),  # on the +x face, level with the lowest mark
              ("side_1", np.array([CX + SPRUE / 2, CY, SLOTS_Z[1]]), rot_z(90.0)))  # and with the middle one

# -- the wax knife (E): a flat blade on a shank, a round handle with a collar
BLADE = (0.025, 0.0008, 0.012)  # x (along the knife), y (thickness, US 11,084,113: about 0.8 mm), z
SHANK = (0.030, 0.003, 0.003)
HANDLE_R, HANDLE_L = 0.006, 0.130
HANDLE_X0 = BLADE[0] / 2 + SHANK[0]
GRASP_X = 0.115  # where the hand takes the handle: over the knife's centre of mass (the handpiece and its cord end are the weight)
KNIFE_MASS = 0.040
KNIFE_REST = np.array([0.33, 0.26, 0.050])  # the blade's centre as the knife lies in its rest, blade toward +x, handle toward -x
KNIFE_REST_R = rot_z(180.0)  # the knife's +x (along the handle) points to world -x
REST_X = (HANDLE_X0 + 0.020, HANDLE_X0 + 0.110)  # the two cradles under the handle, in the knife's frame


def blade_points(n: int = 7, m: int = 4) -> np.ndarray:
    """Points over both flat faces of the blade and its edges, in the knife's frame: what the weld rule tests for contact."""
    xs = np.linspace(-BLADE[0] / 2, BLADE[0] / 2, n)
    zs = np.linspace(-BLADE[2] / 2, BLADE[2] / 2, m)
    pts = [[x, s * BLADE[1] / 2, z] for x in xs for z in zs for s in (-1.0, 1.0)]
    return np.array(pts)


BLADE_PTS = blade_points()
KNIFE_PTS = np.vstack([BLADE_PTS, [[x, 0.0, 0.0] for x in np.linspace(HANDLE_X0 - SHANK[0], HANDLE_X0 + HANDLE_L, 12)]])  # the blade and the handle's axis


def to_world(R: np.ndarray, p: np.ndarray, pts) -> np.ndarray:
    return np.atleast_2d(pts) @ np.asarray(R).T + np.asarray(p)


def to_local(R: np.ndarray, p: np.ndarray, pts) -> np.ndarray:
    return (np.atleast_2d(pts) - np.asarray(p)) @ np.asarray(R)


def box_distance(pts_local: np.ndarray, centre: np.ndarray, half: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance of points (in the box's parent frame) from a box, and each point's nearest point on the box in the box's own frame (0 inside)."""
    q = (pts_local - centre) @ rot
    near = np.clip(q, -half, half)
    return np.linalg.norm(q - near, axis=1), near


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
