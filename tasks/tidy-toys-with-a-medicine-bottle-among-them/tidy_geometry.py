"""Every size and position of the task, in one place (metres, kilograms; heights above the table unless said).

A foam play-mat tile with three toys and an open prescription vial on it, the vial's cap beside it and ten loose tablets inside; a
toy basket at the mat's left edge; a combination-lock medicine box, open, to its right. Sizes marked (pub) are published by the
maker or a seller (README, "Sizes and sources"); those marked (est) are estimated from sellers' photos and are to be measured on
the bought parts.
"""

import math

import numpy as np

import mechkit as m
from bodykit import BoxShape, CylShape, plate

IN = 0.0254

# -- the play mat: one 12 x 12 in EVA tile, 3/8 in thick (ProSource Kids puzzle play mat) ------------------------------------------
MAT_XY = (0.25, 0.0)
MAT = (12 * IN, 12 * IN, 0.375 * IN)  # (pub)

# -- the vial: amber 20 dram push-down-and-turn prescription vial (PSA-AVCR20, U.S. Plastic 81852) ---------------------------------
# Its own frame: origin at the middle of its underside, z up its axis.
VIAL_H = 0.063  # without the cap (pub)
VIAL_CAPPED_H = 0.066  # with it (pub)
VIAL_RI = 0.021  # 42 mm opening (pub)
VIAL_RO = 0.022  # (est: a 1 mm wall)
VIAL_FLOOR = 0.0015  # (est)
VIAL_MASS = 0.0113  # 0.025 lb (pub; whether with the cap is not said)
VIAL_XY = (0.25, 0.0)

# -- its cap: white, ribbed, push-down-and-turn (est: no seller gives its size) -----------------------------------------------------
# Its own frame: origin at the middle of its open rim, z toward its top.
CAP_R = 0.0235
CAP_H = 0.015
CAP_TOP_T = 0.0015
CAP_MASS = 0.004
CAP_XY = (0.30, -0.11)
CAP_SEAT_Z = VIAL_CAPPED_H - CAP_H  # the cap's rim this high above the vial's underside when it has snapped: it adds 3 mm to the height (pub)

# -- the tablets: Zeebo Placebo Tablets, 10 mm round (pub); thickness and weight (est) ----------------------------------------------
N_TABLETS = 10
TABLET_R, TABLET_T, TABLET_MASS = 0.005, 0.004, 0.0003

# -- the toys ----------------------------------------------------------------------------------------------------------------------
BLOCK = 1.75 * IN  # Uncle Goose block: a 1.75 in cube (pub)
BLOCK_MASS = 0.040  # basswood (est)
BRICK = (0.0636, 0.0318, 0.0192)  # LEGO DUPLO 2 x 4: twice a LEGO brick's 31.8 x 15.8 x 9.6 mm (derived from LEGO's "twice the size")
STUD_R, STUD_H, STUD_PITCH = 0.0047, 0.0045, 0.016  # (est)
BRICK_MASS = 0.012  # (a catalogue figure)
TOYS = {"block": (0.14, 0.10), "brick_red": (0.36, 0.08), "brick_yellow": (0.26, 0.10)}  # left of the vial: the holding hand comes in over the mat's near side

# -- the toy basket: IKEA TROFAST storage box 105.820.93 ----------------------------------------------------------------------------
BASKET = (11.75 * IN, 7.75 * IN, 4.0 * IN)  # along x, along y, height (pub)
BASKET_WALL = 0.002  # (est)
BASKET_XY = (0.25, 0.28)

# -- the medicine lock box: Vaultz VZ00266 ------------------------------------------------------------------------------------------
# Its own frame: origin on the table under the middle of its footprint; +x to its back (the hinge side), y along its width, z up.
BOX_W, BOX_D, BOX_H = 0.1740, 0.1130, 0.1311  # width along y, depth along x, height shut (pub)
BOX_IN = (0.1031, 0.1651, 0.0991)  # inside: along x, along y, depth of the base (pub)
LID_IN_H = 0.0239  # the lid's inside height (pub)
BOX_FLOOR = (BOX_H - BOX_IN[2] - LID_IN_H) / 2  # 4 mm: floor and lid top taken as equal (derived)
BOX_RIM_Z = BOX_FLOOR + BOX_IN[2]  # 103 mm: where the lid meets the base
LID_OPEN_DEG = 100.0  # it stands open 10 degrees past upright (est)
LID_MASS = 0.10  # of the 0.36 kg (est)
HASP = (0.020, 0.045, 0.002)  # width, length down the front, thickness (est)
LOCK_Z = 0.085  # the lock's slot and dials, on the front (est)
DIAL_R, DIAL_W, DIAL_PROUD = 0.0055, 0.008, 0.0012  # (est)
DIAL_Y = (-0.018, -0.029, -0.040)  # beside the slot, toward the robot's right (est)
DIAL_MARK = (0.0010, 0.0050, 0.0016)  # a black mark on each dial's rim at its code position, in place of numbers: it leaves the front as the dial turns
BOX_XY = (0.33, -0.27)


def in_box_xy(center, size, margin: float = 0.0):
    """A test for points (n, 3): inside a box-shaped outline centred at `center` (x, y) with `size` (along x, along y)."""
    cx, cy = center
    hx, hy = size[0] / 2 + margin, size[1] / 2 + margin

    def test(p):
        p = np.atleast_2d(p)
        return (np.abs(p[:, 0] - cx) <= hx) & (np.abs(p[:, 1] - cy) <= hy)

    return test


IN_BASKET_XY = in_box_xy(BASKET_XY, (BASKET[0] - 2 * BASKET_WALL, BASKET[1] - 2 * BASKET_WALL))
IN_LOCKBOX_XY = in_box_xy(BOX_XY, (BOX_IN[0], BOX_IN[1]), 0.001)
ON_MAT_XY = in_box_xy(MAT_XY, (MAT[0], MAT[1]))


def in_basket(points) -> np.ndarray:
    """Inside the toy basket's walls, below its rim."""
    p = np.atleast_2d(points)
    return IN_BASKET_XY(p) & (p[:, 2] < BASKET[2]) & (p[:, 2] > 0.0)


def in_lock_box(points) -> np.ndarray:
    """Inside the lock box's inner walls, below its rim."""
    p = np.atleast_2d(points)
    return IN_LOCKBOX_XY(p) & (p[:, 2] < BOX_RIM_Z + 0.001) & (p[:, 2] > 0.0)


def vial_points() -> np.ndarray:
    """Points on the vial's outside, in its own frame: rings at its base, middle and lip."""
    a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    return np.array([[VIAL_RO * math.cos(t), VIAL_RO * math.sin(t), z] for z in (0.0, VIAL_H / 2, VIAL_H) for t in a])


def in_vial(points_local) -> np.ndarray:
    """For points in the vial's frame: inside it (a tablet's centre), below the cap's inner top."""
    p = np.atleast_2d(points_local)
    return (np.hypot(p[:, 0], p[:, 1]) <= VIAL_RI) & (p[:, 2] >= 0.0) & (p[:, 2] <= VIAL_CAPPED_H)


# -- looks and colours ----------------------------------------------------------------------------------------------------------------
LOOKS = {
    "mat": m.Look(m.rgb(0.33, 0.60, 0.80), roughness=0.9),
    "basket": m.Look(m.rgb(0.95, 0.95, 0.93), roughness=0.5),
    "case": m.Look(m.rgb(0.07, 0.07, 0.08), roughness=0.75),
    "chrome": m.Look(m.rgb(0.80, 0.81, 0.83), metallic=0.9, roughness=0.25),
    "label": m.Look(m.rgb(0.97, 0.97, 0.97), roughness=0.6),
    "ink": m.Look(m.rgb(0.05, 0.05, 0.05), roughness=0.6),
}
AMBER, WHITE, WOOD, RED, YELLOW, BLUE = (0.80, 0.42, 0.08), (0.97, 0.97, 0.97), (0.86, 0.72, 0.52), (0.85, 0.10, 0.10), (0.98, 0.80, 0.08), (0.10, 0.30, 0.75)


def rot_axis(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, sn = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + sn * k + (1 - c) * (k @ k)


def ring(r, z0, z1, t, n, color, collide=True):
    """n flat wall panels round a body's z axis, mid-thickness radius r, from z0 to z1: a cylinder's wall."""
    w = 2.0 * (r + t / 2) * math.tan(math.pi / n) * 1.04
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        c, sn = math.cos(a), math.sin(a)
        out.append(plate((r * c, r * sn, z0), (r * c, r * sn, z1), w, t, (c, sn, 0.0), color, collide))
    return out


# -- the vial, its cap and the tablets: loose bodies (bodykit shapes) ------------------------------------------------------------------
VIAL_LINER = 0.0025  # hidden: the wall collides this much thicker inside, so that a tablet cannot pass through it
VIAL_FLOOR_X = 0.003  # hidden: and the floor this much thicker


def vial_shapes() -> list:
    s = [CylShape((0.0, 0.0, VIAL_FLOOR / 2), VIAL_RO, VIAL_FLOOR, AMBER)]
    s += ring((VIAL_RO + VIAL_RI) / 2, 0.0, VIAL_H, VIAL_RO - VIAL_RI, 24, AMBER)
    s += ring(VIAL_RI - VIAL_LINER / 2, VIAL_FLOOR, VIAL_H - 0.002, VIAL_LINER, 24, None)
    s.append(CylShape((0.0, 0.0, VIAL_FLOOR + VIAL_FLOOR_X / 2), VIAL_RI - VIAL_LINER, VIAL_FLOOR_X, None))
    s += [CylShape((0.0, 0.0, 0.029), VIAL_RO + 0.0003, 0.036, WHITE, collide=False),  # the pharmacy label, drawn only
          CylShape((0.0, 0.0, 0.041), VIAL_RO + 0.0005, 0.007, BLUE, collide=False)]
    return s


def cap_shapes() -> list:
    s = [CylShape((0.0, 0.0, CAP_H - CAP_TOP_T / 2), CAP_R, CAP_TOP_T, WHITE)]
    s += ring(CAP_R - 0.0005, 0.0, CAP_H - CAP_TOP_T, 0.001, 20, WHITE)
    for k in range(30):  # ribs, drawn only
        a = 2 * math.pi * k / 30
        s.append(BoxShape(((CAP_R + 0.0002) * math.cos(a), (CAP_R + 0.0002) * math.sin(a), CAP_H / 2 - 0.0005), (0.0008, 0.0012, CAP_H - 0.002),
                          (0.88, 0.88, 0.88), yaw=a, collide=False))
    return s


def tablet_starts() -> list:
    """Where the ten tablets lie in the standing vial, in its frame: seven on the floor, three on top of them."""
    z0 = VIAL_FLOOR + VIAL_FLOOR_X + TABLET_T / 2 + 0.0005
    out = [(0.0, 0.0, z0)] + [(0.0105 * math.cos(a), 0.0105 * math.sin(a), z0) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
    out += [(0.006 * math.cos(a), 0.006 * math.sin(a), z0 + TABLET_T + 0.0007) for a in np.linspace(0.5, 0.5 + 2 * math.pi, 3, endpoint=False)]
    return out[:N_TABLETS]


def block_shapes() -> list:
    s = [BoxShape((0.0, 0.0, BLOCK / 2), (BLOCK, BLOCK, BLOCK), WOOD)]
    for k, (n, c) in enumerate((((0, 0, 1), RED), ((1, 0, 0), BLUE), ((0, 1, 0), YELLOW), ((-1, 0, 0), RED), ((0, -1, 0), BLUE))):  # a coloured panel a face, drawn only
        n = np.array(n, float)
        size = np.where(np.abs(n) > 0, 0.0006, BLOCK * 0.62)
        s.append(BoxShape(tuple(n * (BLOCK / 2) + np.array([0.0, 0.0, BLOCK / 2])), tuple(size), c, collide=False))
    return s


def brick_shapes(color) -> list:
    s = [BoxShape((0.0, 0.0, BRICK[2] / 2), BRICK, color)]
    s += [CylShape(((i - 1.5) * STUD_PITCH, (j - 0.5) * STUD_PITCH, BRICK[2] + STUD_H / 2), STUD_R, STUD_H, color, collide=False) for i in range(4) for j in range(2)]
    return s


def brick_points() -> np.ndarray:
    return np.array([[sx * BRICK[0] / 2, sy * BRICK[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, BRICK[2])])


# -- fixed parts (mechkit meshes): mat, basket, lock box ------------------------------------------------------------------------------
def walls(size, wall: float, floor: float) -> list:
    """(centre, size) of an open box's floor and four walls: `size` is (along x, along y, height)."""
    sx, sy, h = size
    return [((0.0, 0.0, floor / 2), (sx, sy, floor)),
            ((-(sx - wall) / 2, 0.0, h / 2), (wall, sy, h)), (((sx - wall) / 2, 0.0, h / 2), (wall, sy, h)),
            ((0.0, -(sy - wall) / 2, h / 2), (sx, wall, h)), ((0.0, (sy - wall) / 2, h / 2), (sx, wall, h))]


BOX_WALL = (BOX_D - BOX_IN[0]) / 2  # 4.95 mm, front and back; the sides are taken the same
LID_HINGE = np.array([BOX_D / 2, 0.0, BOX_RIM_Z])  # along y, at the back of the rim
LOCK_T = 0.003  # the lock's plate stands this far off the front (est)
LOCK_PLATE = ((-BOX_D / 2 - LOCK_T / 2, -0.017, LOCK_Z), (LOCK_T, 0.062, 0.024))  # (centre, size): slot at y 0, dials and button to the robot's right
HASP_HINGE = np.array([-BOX_D / 2 - LOCK_T - 0.0005, 0.0, BOX_RIM_Z + 0.006])  # on the lid's front skirt (est)
HASP_REST_DEG = 8.0  # with the lid down it hangs this far off the front, on the lock's sprung catch (est)
HASP_MASS, DIAL_MASS = 0.008, 0.002


def box_base_boxes() -> list:
    return walls((BOX_D, BOX_W, BOX_RIM_Z), BOX_WALL, BOX_FLOOR)


def lid_boxes() -> list:
    """(centre, size) of the lid as it lies shut, in the box's frame: its top and four skirts."""
    z0, z1 = BOX_RIM_Z, BOX_H - BOX_FLOOR
    zs, hs = (z0 + z1) / 2, z1 - z0
    return [((0.0, 0.0, BOX_H - BOX_FLOOR / 2), (BOX_D, BOX_W, BOX_FLOOR)),
            ((-(BOX_D - BOX_WALL) / 2, 0.0, zs), (BOX_WALL, BOX_W, hs)), (((BOX_D - BOX_WALL) / 2, 0.0, zs), (BOX_WALL, BOX_W, hs)),
            ((0.0, -(BOX_W - BOX_WALL) / 2, zs), (BOX_D, BOX_WALL, hs)), ((0.0, (BOX_W - BOX_WALL) / 2, zs), (BOX_D, BOX_WALL, hs))]


# -- the MEDICINE labels: one artwork (medicine_label.svg), two copies on the lid ------------------------------------------------------
LABEL = (0.100, 0.028)  # white paper: along the word, across it
LETTER_H = 0.022  # the black capitals' height; the word is 4.38 times as long (96 mm)
STEM, BAR, GAP = 0.17, 0.18, 0.10  # of that height: an upright stroke's width, a level stroke's, the space between letters
LABEL_T, INK_T = 0.0006, 0.0004  # drawn only: the paper stands this far off the lid, the letters this far off the paper
# (centre on the shut lid, the way the word runs, the way the letters' tops point), in the box's frame. Both read from the robot's side.
LABEL_TOP = ((0.0, 0.0, BOX_H), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))  # on the lid's top, for when the box is shut
# Inside the lid, for while it stands open: 2 mm from the hinge-side wall and 5 mm from the wall on the robot's left. That corner of the
# lid's inside is what neither the hasp nor the right arm at its ready pose hides from the top camera or the right wrist camera.
LABEL_IN = ((BOX_IN[0] / 2 - 0.002 - LABEL[1] / 2, BOX_IN[1] / 2 - 0.005 - LABEL[0] / 2, BOX_H - BOX_FLOOR), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0))


def _bar(x0: float, y0: float, x1: float, y1: float) -> list:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def glyphs() -> dict:
    """Condensed block capitals as {letter: (width, strokes)}, in units of their height. A stroke is a convex outline, counter-clockwise."""
    s, b, c, hook = STEM, BAR, 0.26, 0.29  # c: how far a D's or a C's corners are cut; hook: how far a C's ends turn in
    m_w, n_w, d_w, c_w, e_w, diag, low = 0.72, 0.58, 0.55, 0.53, 0.48, 0.19, 0.25
    return {
        "M": (m_w, [_bar(0, 0, s, 1), _bar(m_w - s, 0, m_w, 1), [(0, 1), (m_w / 2 - diag / 2, low), (m_w / 2 + diag / 2, low), (diag, 1)],
                    [(m_w / 2 - diag / 2, low), (m_w / 2 + diag / 2, low), (m_w, 1), (m_w - diag, 1)]]),
        "E": (e_w, [_bar(0, 0, s, 1), _bar(s, 1 - b, e_w, 1), _bar(s, 0.5 - b / 2, e_w - 0.07, 0.5 + b / 2), _bar(s, 0, e_w, b)]),
        "D": (d_w, [_bar(0, 0, s, 1), _bar(s, 1 - b, d_w - c, 1), _bar(s, 0, d_w - c, b), _bar(d_w - s, c, d_w, 1 - c),
                    [(d_w - c, 1), (d_w - c, 1 - b), (d_w - s, 1 - c), (d_w, 1 - c)], [(d_w - c, 0), (d_w, c), (d_w - s, c), (d_w - c, b)]]),
        "I": (s, [_bar(0, 0, s, 1)]),
        "C": (c_w, [_bar(0, c, s, 1 - c), _bar(c, 1 - b, c_w, 1), _bar(c, 0, c_w, b), _bar(c_w - s, 1 - hook, c_w, 1 - b), _bar(c_w - s, b, c_w, hook),
                    [(c, 1), (0, 1 - c), (s, 1 - c), (c, 1 - b)], [(c, 0), (c, b), (s, c), (0, c)]]),
        "N": (n_w, [_bar(0, 0, s, 1), _bar(n_w - s, 0, n_w, 1), [(0, 1), (n_w - diag, 0), (n_w, 0), (diag, 1)]]),
    }


def label_strokes(word: str = "MEDICINE") -> list:
    """The word's strokes as outlines (n, 2) in metres, in the label's frame: x along the word, y up the letters, from the label's middle."""
    font, x, out = glyphs(), 0.0, []
    for letter in word:
        width, strokes = font[letter]
        out += [np.array(o, float) + (x, 0.0) for o in strokes]
        x += width + GAP
    return [(o - ((x - GAP) / 2, 0.5)) * LETTER_H for o in out]


def label_meshes(centre, along, up) -> tuple:
    """A label's paper and its word, as meshes in the box's frame with the lid shut."""
    turn = np.column_stack([along, up, np.cross(along, up)])
    paper = m.box((LABEL[0], LABEL[1], LABEL_T), (0.0, 0.0, LABEL_T / 2))
    word = m.join(*[m.prism(o, LABEL_T / 2, LABEL_T + INK_T) for o in label_strokes()])
    return tuple(m.Mesh(part.v @ turn.T + np.asarray(centre, float), part.f) for part in (paper, word))


def label_svg() -> str:
    """The label at full size, for printing: what medicine_label.svg holds."""
    w, h = LABEL[0] * 1000, LABEL[1] * 1000
    paths = "".join("M" + " L".join(f"{w / 2 + x * 1000:.2f},{h / 2 - y * 1000:.2f}" for x, y in o) + " Z " for o in label_strokes())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" height="{h:g}mm" viewBox="0 0 {w:g} {h:g}">\n'
            f'<rect width="{w:g}" height="{h:g}" fill="#ffffff" stroke="#000000" stroke-width="0.1"/>\n<path fill="#000000" d="{paths.strip()}"/>\n</svg>\n')


def lid_rot(deg: float = LID_OPEN_DEG) -> np.ndarray:
    """The lid swung `deg` up and back from shut, about its hinge (the box's y)."""
    return rot_axis((0, 1, 0), deg)


def on_lid(point, deg: float = LID_OPEN_DEG) -> np.ndarray:
    """A point of the shut lid, with the lid `deg` open."""
    return lid_rot(deg) @ (np.asarray(point, float) - LID_HINGE) + LID_HINGE


def hasp_box() -> tuple:
    """(centre, size) of the hasp as it lies home, flat on the lock, with the lid shut."""
    w, length, t = HASP
    return (HASP_HINGE[0] - t / 2, 0.0, HASP_HINGE[2] - length / 2), (t, w, length)


def hasp_rot(off_deg: float) -> np.ndarray:
    """The hasp swung `off_deg` off the front about its hinge: its lower end toward the robot."""
    return rot_axis((0, 1, 0), off_deg)


def hasp_points(lid_deg: float, off_deg: float) -> np.ndarray:
    """The hasp's four corners and middle in the box's frame, with the lid `lid_deg` open and the hasp `off_deg` off the front."""
    (cx, cy, cz), (t, w, length) = hasp_box()
    pts = np.array([[cx, sy * w / 2, cz + sz * length / 2] for sy in (-1, 1) for sz in (-1, 1)] + [[cx, 0.0, cz]])
    pts = (pts - HASP_HINGE) @ hasp_rot(off_deg).T + HASP_HINGE
    return (pts - LID_HINGE) @ lid_rot(lid_deg).T + LID_HINGE


def dial_at(k: int) -> tuple:
    """Dial k's axis (along y): its rim stands DIAL_PROUD in front of the lock's plate."""
    return (-BOX_D / 2 - LOCK_T - DIAL_PROUD + DIAL_R, DIAL_Y[k], LOCK_Z)


def lid_top_point(lid_deg: float) -> np.ndarray:
    """The middle of the lid's front edge, in the box's frame."""
    return on_lid((-BOX_D / 2, 0.0, BOX_H), lid_deg)


def box_gap(points_local) -> float:
    """The least distance from points (in the box's frame) to the shut box's outside."""
    p = np.atleast_2d(points_local)
    lo, hi = np.array([-BOX_D / 2, -BOX_W / 2, 0.0]), np.array([BOX_D / 2, BOX_W / 2, BOX_H])
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1).min())
