"""Wax setting on a jeweller's bench: an 8 mm red stone is set into a turquoise wax ring that sits on a wooden mandrel.

On the table: the mandrel (a 295 mm tapered stick) lying level in a steel V-block and on a wooden rest, the wax ring on
it overhanging the V-block, head up, its four claws standing open; the stone, table-up in a small block; and, in a row,
tweezers and a pusher standing in a tool block and a warm electric wax pen standing in its stand. One gripper holds the
mandrel by its thick end. The other picks the stone up with the tweezers and lays it between the claws, presses it home
with the pusher, then closes each claw with the pen: the tip rests on the claw until the wax softens, takes it round over
the stone's edge and lifts off. A claw pushed cold springs back; one kept under the pen too long melts onto the stone.
Tools go back in their places.

Files: this one (the scorer's wiring, the props, the demo), `wax_rule.py` (the wax model and the score), `ring_geometry.py`
(every number), `mechkit.py` and `demokit.py` (kits from the accepted task heat-shrink-lug).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mechkit as m  # noqa: E402
import ring_geometry as g  # noqa: E402
import wax_rule as w  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
INSTRUCTION = ("Set the red stone into the wax ring. With one gripper, hold the wooden mandrel by its thick end: the ring overhangs the V-block, so a mandrel nobody "
               "holds tips when the stone is pressed. With the other, take the tweezers, pick the stone up by its girdle, lay it table-up between the four wax claws "
               "and put the tweezers back. Take the pusher, press the stone down onto its seat and put the pusher back. Take the warm wax pen and close the claws one "
               "at a time: rest its tip against the outside of a claw until the wax softens, push the claw over the stone's edge until it lies on the stone, and lift "
               "off. Do not push a claw before it has softened or keep the pen on one long enough to melt it. Touch nothing but the claws with the pen's tip, lay the "
               "pen nowhere but in its stand, and never touch the stone or the wax with a gripper. Finish with the pen in its stand, the tweezers and pusher in their "
               "block and the mandrel on its rests, and move both grippers 5 cm or more away from all of them.")
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
UP = np.array([0.0, 0.0, 1.0])
LIFT = np.array([0.0, 0.0, TOP])

# -- the scorer's wiring: poses in, joint targets and looks out (the rule itself is wax_rule.py) ---------------------------

FINGER_DY = (-0.018, -0.009, 0.0, 0.009, 0.018)  # across each finger
FINGER_DZ = (0.147, 0.137, 0.127, 0.117, 0.107)  # from the pad tips up, in the gripper link's frame


def finger_points(env) -> dict:
    """Points down the middle of each finger of both grippers (station frame, heights above the table): arm -> (one finger's, the other's)."""
    base = env.unwrapped
    origin = m._np(base.scene.env_origins[0])
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        i = robot.find_bodies("gripper")[0][0]
        pose = m._np(robot.data.body_link_pose_w)[0, i]
        rot = m.quat_matrix(pose[3:7])
        opening = abs(float(m._np(robot.data.joint_pos)[0, robot.find_joints("joint7")[0][0]]))
        out[side] = tuple(np.array([pose[:3] - origin - LIFT + rot @ np.array([sx * (opening + 0.0045), dy, dz]) for dy in FINGER_DY for dz in FINGER_DZ]) for sx in (-1.0, 1.0))
    return out


def _pose(state) -> w.Pose:
    return w.Pose(np.asarray(state.pos, float) - LIFT, state.rotation())


def snapshot(env, trail: list) -> w.Snapshot:
    """One control step as the scorer sees it. `trail` keeps the last second of the loose parts' poses, from which the rule reads whether they are at rest."""
    st = {n: m.state(env, n) for n in ("mandrel", "stone", "wax_pen", "tweezers", "pusher")}
    dt = float(env.unwrapped.step_dt)
    poses = {n: _pose(s) for n, s in st.items()}
    trail.append(poses)
    return w.Snapshot(dt=dt, mandrel=poses["mandrel"], claws=tuple(st["mandrel"].joint(f"claw{i}") for i in range(g.N_CLAWS)), seat=st["mandrel"].joint("seat"),
                      stone=poses["stone"], pen=poses["wax_pen"], tweezers=poses["tweezers"], pusher=poses["pusher"], fingers=finger_points(env), at_rest=w.settled(trail, dt))


class Trial:
    """Runs the wax rule once per control step, sets the claws' and the seat's drives from it, and shows what the wax looks like."""

    def __init__(self):
        self.wax = w.Wax()
        self.sent: dict = {}
        self.shown: dict = {}
        self.loose: dict = {}
        self.trail: list = []

    def reset(self, env=None):
        self.wax.reset()
        self.sent, self.shown, self.loose, self.trail = {}, {}, {}, []  # the kit's reset has given every joint back its own spring and damping
        if env is not None:
            self.show(env)

    def step(self, env):
        self.wax.step(snapshot(env, self.trail))
        for i in range(g.N_CLAWS):
            self.send(env, f"claw{i}", self.wax.target[i])
            loose = self.wax.soft[i] or self.wax.melted[i]  # soft wax has no spring: the claw goes where it is pushed, and sets where it stands
            if self.loose.get(i, False) != loose:
                m.set_gains(env, "mandrel", f"claw{i}", 0.0 if loose else w.COLD_RATE, w.SOFT_DAMP if loose else w.COLD_DAMP)
                self.loose[i] = loose
        self.send(env, "seat", self.wax.seat_target)
        self.show(env)

    def send(self, env, link, value):
        if self.sent.get(link) is None or abs(self.sent[link] - value) > 1e-7:
            m.set_target(env, "mandrel", link, value)
            self.sent[link] = value

    def show(self, env):
        """A soft claw looks wet; a melted one is a puddle on the stone's edge; a broken one is gone."""
        for i in range(g.N_CLAWS):
            gone = self.wax.melted[i] or self.wax.broken[i]
            looks = {f"claw{i}/visual/cold": not gone and not self.wax.soft[i], f"claw{i}/visual/soft": not gone and self.wax.soft[i], f"frame/visual/blob{i}": self.wax.melted[i]}
            for path, on in looks.items():
                if self.shown.get(path) != on:
                    try:
                        m.set_visible(f"{m.ENV_PATH}/mandrel/{path}", on)
                    except Exception as e:  # a look that cannot be shown must not stop the scorer
                        print(f"[wax] cannot show {path}: {e}", flush=True)
                    self.shown[path] = on


TRIAL = Trial()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    return TRIAL.wax.full


def score(objects) -> float:
    return TRIAL.wax.score


# -- the props -------------------------------------------------------------------------------------------------------------
# Whatever must be an exact size is a box or a rod; a convex hull is used only where a few tenths of a millimetre do not matter (mechkit.py says why).

LOOKS = {
    "wood": m.Look(color=m.rgb(0.70, 0.52, 0.32), roughness=0.75),
    "beech": m.Look(color=m.rgb(0.82, 0.68, 0.48), roughness=0.75),
    "handle": m.Look(color=m.rgb(0.42, 0.24, 0.12), roughness=0.55),
    "steel": m.Look(color=m.rgb(0.64, 0.65, 0.68), roughness=0.35, metallic=0.9),
    "wax": m.Look(color=m.rgb(0.05, 0.62, 0.62), roughness=0.55),
    "wax_soft": m.Look(color=m.rgb(0.02, 0.36, 0.40), roughness=0.05),
    "stone": m.Look(color=m.rgb(0.80, 0.04, 0.08), roughness=0.08, metallic=0.2),
    "pen": m.Look(color=m.rgb(0.10, 0.10, 0.12), roughness=0.5),
    "brass": m.Look(color=m.rgb(0.78, 0.62, 0.30), roughness=0.3, metallic=0.9),
}
BENCH_FRICTION = g.WOOD_ON_STEEL


def _turn(deg: float, axis: str) -> np.ndarray:
    """The rotation `Mesh.turned` applies."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array({"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]], "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis], float)


def _box(name, size, centre, look, parts, cols):
    parts.append((name, m.box(size, centre), look))
    cols.append(m.col_box(name + "_c", size, centre))


def _slab(name, size, centre, turn_deg, axis, look, parts, cols):
    """A box turned about one axis through its centre: seen, and an exact box to touch."""
    parts.append((name, m.box(size).turned(turn_deg, axis).moved(centre), look))
    cols.append(m.col_box(name + "_c", size, centre, _turn(turn_deg, axis)))


def pocket_boxes(centre_xy, size, pockets):
    """A block with square-cornered pockets cut down from its top, as boxes: (size, centre) each. A pocket is (centre x, y, width along x, along y, depth)."""
    cx, cy = centre_xy
    sx, sy, h = size
    xs = sorted({cx - sx / 2, cx + sx / 2, *(p[0] + s * p[2] / 2 for p in pockets for s in (-1, 1))})
    ys = sorted({cy - sy / 2, cy + sy / 2, *(p[1] + s * p[3] / 2 for p in pockets for s in (-1, 1))})
    out = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            depth = max([p[4] for p in pockets if abs(mx - p[0]) < p[2] / 2 and abs(my - p[1]) < p[3] / 2], default=0.0)
            if h - depth > 1e-6:
                out.append(((x1 - x0, y1 - y0, h - depth), (mx, my, (h - depth) / 2)))
    return out


def _block(name, centre_xy, size, pockets, look, parts, cols):
    for k, (sz, c) in enumerate(pocket_boxes(centre_xy, size, pockets)):
        _box(f"{name}{k}", sz, c, look, parts, cols)


def _mouth(name, centre_xy, opening, top, length, sides, look, parts, cols):
    """A funnel: plates at 45 degrees along the sides of an opening in a top face, each plate's lower edge on the opening's edge."""
    cx, cy = centre_xy
    for k, (dx, dy) in enumerate(sides):
        if dx:
            _slab(f"{name}{k}", (g.MOUTH_W, length, g.MOUTH_T), (cx + dx * (opening[0] / 2 + g.MOUTH_OFF), cy, top + g.MOUTH_UP), -45.0 * dx, "y", look, parts, cols)
        else:
            _slab(f"{name}{k}", (length, g.MOUTH_W, g.MOUTH_T), (cx, cy + dy * (opening[1] / 2 + g.MOUTH_OFF), top + g.MOUTH_UP), 45.0 * dy, "x", look, parts, cols)


ALL_ROUND = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bench():
    """Everything that does not move: the V-block and the rest under the mandrel, the stone's block, the tool block, the pen's stand."""
    parts, cols = [], []
    x0, y0 = g.RING_XY
    # the V-block: a base and two faces at 45 degrees that carry the stick's axis at AXIS_H
    vy = y0 + g.V_Y
    r = g.stick_r(g.V_Y)
    _box("v_base", (g.V_SIZE[0], g.V_SIZE[1], 0.008), (x0, vy, 0.004), "steel", parts, cols)
    for s, name in ((1.0, "v_left"), (-1.0, "v_right")):
        n = np.array([s * math.sqrt(0.5), 0.0, math.sqrt(0.5)])  # from the face to the stick's axis
        c = np.array([x0, vy, g.AXIS_H]) - n * (r + 0.003)
        _slab(name, (0.022, g.V_SIZE[1], 0.006), tuple(c), 45.0 * s, "y", "steel", parts, cols)
    _box("rest", g.REST_SIZE, (x0, y0 + g.REST_Y, g.REST_SIZE[2] / 2), "beech", parts, cols)
    _block("stone_block", g.STONE_BLOCK_XY, g.STONE_BLOCK, [(*g.STONE_BLOCK_XY, g.STONE_HOLE, g.STONE_HOLE, 0.006)], "beech", parts, cols)
    trough = (g.TWEEZER_SLOT[0], g.TROUGH_TOP, g.TWEEZER_SLOT[2])
    _block("tool_block", g.TOOL_BLOCK_XY, g.TOOL_BLOCK, [(*g.TWEEZER_XY, *trough), (*g.PUSHER_XY, *g.PUSHER_HOLE)], "beech", parts, cols)
    lean = math.atan(g.TROUGH_RUN)
    for s, name in ((1.0, "trough_a"), (-1.0, "trough_b")):  # the trough's long walls: two boards that lean apart, the floor between their feet
        face = np.array([g.TWEEZER_XY[0], g.TWEEZER_XY[1] + s * (g.TWEEZER_SLOT[1] + g.TROUGH_TOP) / 4, g.TOOL_BLOCK[2] - g.TWEEZER_SLOT[2] / 2])
        back = np.array([0.0, s * math.cos(lean), -math.sin(lean)])  # from the wall's face into the board
        _slab(name, (g.TWEEZER_SLOT[0], g.TROUGH_BOARD, g.TWEEZER_SLOT[2] / math.cos(lean)), tuple(face + back * g.TROUGH_BOARD / 2), -s * math.degrees(lean), "x", "beech", parts, cols)
    _mouth("tweezer_mouth", g.TWEEZER_XY, trough, g.TOOL_BLOCK[2], g.TROUGH_TOP + 0.002, ((1, 0), (-1, 0)), "steel", parts, cols)
    _mouth("pusher_mouth", g.PUSHER_XY, g.PUSHER_HOLE, g.TOOL_BLOCK[2], 0.016, ALL_ROUND, "steel", parts, cols)
    px, py = g.PEN_STAND_XY
    _box("pen_base", g.PEN_BASE, (px, py, g.PEN_BASE[2] / 2), "steel", parts, cols)
    top = g.PEN_TUBE[2]
    for k, (sz, c) in enumerate(pocket_boxes(g.PEN_STAND_XY, (g.PEN_TUBE[0], g.PEN_TUBE[1], top - g.PEN_BASE[2]),
                                             [(px, py, g.PEN_BORE, g.PEN_BORE, top - g.PEN_LEDGE_H), (px, py, g.PEN_TIP_BORE, g.PEN_TIP_BORE, top - g.PEN_BASE[2])])):
        _box(f"pen_tube{k}", sz, (c[0], c[1], c[2] + g.PEN_BASE[2]), "steel", parts, cols)
    _mouth("pen_mouth", g.PEN_STAND_XY, (g.PEN_BORE, g.PEN_BORE), top, g.PEN_TUBE[0], ALL_ROUND, "steel", parts, cols)
    return parts, cols


_bp, _bc = _bench()
BENCH = m.fixed("bench", (0.0, 0.0, TOP), _bp, _bc, LOOKS, friction=BENCH_FRICTION)


def _proper(rot: np.ndarray) -> np.ndarray:
    """A box's axes as a rotation: one axis turned round if they came left-handed (a box looks the same)."""
    rot = np.asarray(rot, float)
    return rot if np.linalg.det(rot) > 0 else rot * np.array([1.0, -1.0, 1.0])


def _placed(mesh: m.Mesh, rot: np.ndarray, centre) -> m.Mesh:
    return m.Mesh(mesh.v @ np.asarray(rot, float).T + np.asarray(centre, float), mesh.f)


def _loop(profile, segments=32) -> m.Mesh:
    """A ring: a closed (radius, z) section, counter-clockwise, revolved about z."""
    return m.lathe(list(profile) + [profile[0]], segments)


def _octagon(inradius: float) -> list:
    """An octagon's corners, counter-clockwise, its flats square to x and to y at `inradius`: what flat pads hold, and what cooking keeps whole."""
    r = inradius / math.cos(math.radians(22.5))
    return [(r * math.cos(math.radians(22.5 + 45.0 * k)), r * math.sin(math.radians(22.5 + 45.0 * k))) for k in range(8)]


def _mandrel():
    """The stick with the wax ring on it: one free body. The ring's seat slides and its four claws hinge; the rule in wax_rule.py sets where their drives rest."""
    stick = m.lathe([(0.0, g.Y_THIN), (g.STICK_R_THIN, g.Y_THIN), (g.STICK_R_THICK, g.Y_THICK), (0.0, g.Y_THICK)], 32).aimed("+Y")
    e = 0.0003
    shank = _loop([(g.RING_IN_R, -g.SHANK_W / 2), (g.RING_OUT_R - e, -g.SHANK_W / 2), (g.RING_OUT_R, -g.SHANK_W / 2 + e), (g.RING_OUT_R, g.SHANK_W / 2 - e),
                   (g.RING_OUT_R - e, g.SHANK_W / 2), (g.RING_IN_R, g.SHANK_W / 2)], 48).aimed("+Y")
    cup = _loop(g.CUP_SECTION, 32)
    parts = [("stick", stick, "wood"), ("shank", shank, "wax"), ("cup", cup, "wax")]
    cols = [m.col_capsule(f"stick_c{k}", r, y1 - y0, (0.0, (y0 + y1) / 2, 0.0), "Y") for k, (y0, y1, r) in enumerate(g.stick_rods())]
    chord = 2 * g.RING_OUT_R * math.tan(math.radians(15.0))
    for k in range(12):  # the shank as twelve short bars round the stick
        rot = _turn(30.0 * k, "y")
        cols.append(m.col_box(f"shank_c{k}", (g.SHANK_T, g.SHANK_W, chord), tuple(rot @ np.array([g.RING_IN_R + g.SHANK_T / 2, 0.0, 0.0])), rot))
    wall = (g.CUP_R1 - g.HEAD_WALL_R, 2 * g.HEAD_WALL_R * math.tan(math.radians(15.0)), g.HEAD_WALL_H)
    for k in range(12):  # the head's rim as twelve thin blocks, clear of the stone
        rot = g.rot_z(30.0 * k + 15.0)
        cols.append(m.col_box(f"cup_c{k}", wall, tuple(rot @ np.array([(g.CUP_R1 + g.HEAD_WALL_R) / 2, 0.0, g.RIM_Z - g.HEAD_WALL_H / 2])), rot))
    for k in range(4):  # the bearing: four blocks along and across the mandrel, whose inside top edges carry the pavilion of a stone at home
        rot = g.rot_z(90.0 * k)
        cols.append(m.col_box(f"bearing_c{k}", g.BEAR_BLOCK, tuple(rot @ np.array([g.BEAR_R + g.BEAR_BLOCK[0] / 2, 0.0, g.BEAR_Z - g.BEAR_BLOCK[2] / 2])), rot))
    links = []
    for i in range(g.N_CLAWS):
        centre, rot, half = g.claw_box(i, 0.0)
        rot = _proper(rot)
        claw = _placed(m.box(tuple(2 * half)), rot, centre)
        hinge, frame = g.claw_frame(i)
        tip = hinge + frame[:, 2] * g.TABLE_OVER_GIRDLE - frame[:, 0] * 0.0008  # where a melted claw's wax would lie: on the crown
        parts.append((f"blob{i}", m.sphere(0.0015, tuple(tip), 6, 12), "wax_soft", True))
        psi = math.radians(g.CLAW_AZ_DEG[i] - 90.0) / 2  # turns the joint's X onto the claw's hinge axis
        links.append(m.hinge(f"claw{i}", "X", tuple(hinge), -5.0, g.CLAW_STOP_DEG, [("cold", claw, "wax"), ("soft", claw, "wax_soft", True)],
                             [m.col_box(f"claw{i}_c", tuple(2 * half), tuple(centre), rot)], mass=0.0003, spring=w.COLD_RATE, damping=w.COLD_DAMP, friction=0.0, effort=1.0, armature=1e-7,
                             inertia=(1e-7, 1e-7, 1e-7), rot=(math.cos(psi), 0.0, 0.0, math.sin(psi))))
    seat_ring = _loop([(g.SEAT_R, g.SEAT_TOP0 - 0.0015), (g.BEAR_R - 0.0001, g.SEAT_TOP0 - 0.0015), (g.BEAR_R - 0.0001, g.SEAT_TOP0), (g.SEAT_R, g.SEAT_TOP0)], 24)
    seat_cols = []
    for k in range(4):  # four blocks under the pavilion: it rests on their inside top edges
        rot = g.rot_z(90.0 * k)
        seat_cols.append(m.col_box(f"seat_c{k}", g.SEAT_BLOCK, tuple(rot @ np.array([g.SEAT_R + g.SEAT_BLOCK[0] / 2, 0.0, g.SEAT_TOP0 - g.SEAT_BLOCK[2] / 2])), rot))
    links.append(m.slider("seat", "Z", -g.SEAT_LIMIT, 0.0, [("ring", seat_ring, "wax")], seat_cols, at=(0.0, 0.0, g.SEAT_TOP0), mass=0.002, spring=w.SEAT_RATE, damping=12.0, friction=0.0,
                          effort=20.0, armature=0.02))
    mass = g.STICK_MASS + g.RING_MASS
    return m.mechanism("mandrel", (g.RING_XY[0], g.RING_XY[1], TOP + g.AXIS_H + 0.0004), links, LOOKS, parts, cols, friction=g.WOOD_ON_STEEL, solver_iters=64, free_mass=mass,
                       frame_inertia=(mass * g.STICK_LEN ** 2 / 12, mass * 0.009 ** 2 / 2, mass * g.STICK_LEN ** 2 / 12), frame_com=(0.0, g.STICK_COM_Y, 0.0))


MANDREL = _mandrel()


def _stone():
    gem = m.lathe(list(g.STONE_PROFILE), 16)
    hull = m.Mesh(g.stone_hull(), np.array(g.STONE_HULL_FACES))
    return m.free("stone", (g.STONE_BLOCK_XY[0], g.STONE_BLOCK_XY[1], TOP + g.STONE_REST_H + 0.0003), [("gem", gem, "stone")], [m.col_hull("gem_c", hull)], LOOKS, mass=g.STONE_MASS,
                  friction=(0.5, 0.4), torsional_patch=0.001, solver_iters=64, com=(0.0, 0.0, g.STONE_COM_Z))


STONE = _stone()


def _tweezers():
    """A joined end and two legs, each on a sprung hinge: squeezing the legs closes the tips, as far as they are squeezed."""
    a = g.TW_OPEN_DEG
    top = (g.TW_LEN + g.TW_HINGE_Z) / 2
    head = (0.006, 0.009, g.TW_LEN - g.TW_HINGE_Z)
    parts = [("joint", m.box(head, (0.0, 0.0, top)), "steel")]
    cols = [m.col_box("joint_c", head, (0.0, 0.0, top))]
    links = []
    long = g.TW_HINGE_Z / math.cos(math.radians(a))
    for name, s in (("leg_a", -1.0), ("leg_b", 1.0)):
        hinge = np.array([s * g.TW_ROOT / 2, 0.0, g.TW_HINGE_Z])
        down = np.array([s * math.sin(math.radians(a)), 0.0, -math.cos(math.radians(a))])
        out = np.array([s * math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))])
        upper_len = long - g.TW_TIP_H
        seen, touched = [], []
        _slab("leg", (g.TW_LEG_T, g.TW_LEG_W, upper_len), tuple(hinge + down * upper_len / 2 + out * g.TW_LEG_T / 2), -s * a, "y", "steel", seen, touched)
        _slab("tip", (g.TW_LEG_T, g.TW_TIP_W, g.TW_TIP_H), tuple(hinge + down * (upper_len + g.TW_TIP_H / 2) + out * g.TW_LEG_T / 2), -s * a, "y", "steel", seen, touched)
        lim = (-0.5, a + 0.4) if s > 0 else (-a - 0.4, 0.5)  # leg_b closes with a positive angle, leg_a with a negative one
        links.append(m.hinge(name, "Y", tuple(hinge), lim[0], lim[1], seen, touched, mass=0.008, spring=g.TW_SPRING, damping=0.003, friction=0.0, effort=20.0, armature=1e-5,
                             gravity=True, inertia=(1.2e-5, 1.2e-5, 3e-7)))
    return m.mechanism("tweezers", (g.TWEEZER_XY[0], g.TWEEZER_XY[1], TOP + w.TWEEZERS_START.pos[2] + 0.0004), links, LOOKS, parts, cols, friction=(0.9, 0.8), solver_iters=64,
                       free_mass=g.TW_MASS - 0.016, frame_inertia=(1e-6, 1e-6, 5e-7))


TWEEZERS = _tweezers()


def _pusher():
    z0 = g.PU_SHANK + g.PU_NECK_H
    shank = (g.PU_FACE, g.PU_FACE, g.PU_SHANK)
    neck = m.cylinder(g.PU_NECK_R, g.PU_NECK_H, (0.0, 0.0, g.PU_SHANK + g.PU_NECK_H / 2))
    band = m.cylinder(g.PU_KNOB_R, g.PU_KNOB_H, (0.0, 0.0, z0 + g.PU_KNOB_H / 2))
    dome = m.lathe([(0.0, z0 + g.PU_KNOB_H), (g.PU_KNOB_R, z0 + g.PU_KNOB_H), (0.013, z0 + g.PU_KNOB_H + 0.004), (0.008, g.PU_LEN - 0.001), (0.0, g.PU_LEN)], 24)
    parts = [("shank", m.box(shank, (0.0, 0.0, g.PU_SHANK / 2)), "steel"), ("neck", neck, "handle"), ("band", band, "handle"), ("dome", dome, "handle")]
    cols = [m.col_box("shank_c", shank, (0.0, 0.0, g.PU_SHANK / 2)), m.col_hull("neck_c", m.prism(_octagon(g.PU_NECK_R), g.PU_SHANK, z0)),
            m.col_hull("band_c", m.prism(_octagon(g.PU_KNOB_R), z0, z0 + g.PU_KNOB_H)), m.col_hull("dome_c", m.prism(_octagon(0.008), z0 + g.PU_KNOB_H, g.PU_LEN))]
    return m.free("pusher", (g.PUSHER_XY[0], g.PUSHER_XY[1], TOP + w.PUSHER_START.pos[2] + 0.0004), parts, cols, LOOKS, mass=g.PU_MASS, friction=(0.9, 0.8), torsional_patch=0.004,
                  com=(0.0, 0.0, 0.050))


PUSHER = _pusher()


def _pen():
    tip = m.lathe([(0.0, 0.0), (0.7 * g.PEN_TIP_R, 0.29 * g.PEN_TIP_R), (g.PEN_TIP_R, g.PEN_TIP_R), (g.PEN_TIP_R, g.PEN_TIP_LEN), (0.0, g.PEN_TIP_LEN)], 16)
    ferrule = m.cylinder(0.0035, 0.010, (0.0, 0.0, g.PEN_TIP_LEN - 0.005))
    handle = m.cylinder(g.PEN_HANDLE_R, g.PEN_LEN - g.PEN_TIP_LEN, (0.0, 0.0, (g.PEN_LEN + g.PEN_TIP_LEN) / 2))
    parts = [("tip", tip, "brass"), ("ferrule", ferrule, "steel"), ("handle", handle, "pen")]
    cols = [m.col_capsule("tip_c", g.PEN_TIP_R, g.PEN_TIP_LEN - g.PEN_TIP_R, (0.0, 0.0, (g.PEN_TIP_LEN + g.PEN_TIP_R) / 2)),  # its round end is the ball the rule measures from
            m.col_hull("handle_c", m.prism(_octagon(g.PEN_HANDLE_R), g.PEN_TIP_LEN, g.PEN_LEN))]
    return m.free("wax_pen", (g.PEN_STAND_XY[0], g.PEN_STAND_XY[1], TOP + w.PEN_START.pos[2] + 0.0004), parts, cols, LOOKS, mass=g.PEN_MASS, friction=(0.9, 0.8), torsional_patch=0.003,
                  com=(0.0, 0.0, 0.100))


PEN = _pen()
OBJECTS = [BENCH, MANDREL, STONE, TWEEZERS, PUSHER, PEN]


# -- the demo --------------------------------------------------------------------------------------------------------------
# Closed loop, for demokit.py: it reads where the parts are and how far each claw has turned (all of which the cameras show), never the scorer.

MODES = ("full", "none", "hover", "cold", "long", "early", "unheld", "shank", "probe", "claws", "tools")
CLAW_ORDER = (3, 1, 2, 0)  # opposite pairs, starting on the side the pen comes from
PUSH_FACE = (g.CLAW_W + g.PEN_TIP_R, 0.0012)  # the tip's ball rides this far out from a claw's inside face, and this far up it
HOLD_TILT_DEG = 30.0  # the holding hand leans back along the stick: upright, its wrist would stand where the other hand works at the ring
APPROACH = (0.007, 0.009)  # the tip comes to a claw from this far outside it and this far above its place on it
STAND_OFF = 0.0025  # the tip first stands this far off a claw's face
WAIT_S = 2.4  # how long the tip rests on a claw before it pushes: the wax has softened by then, and stays soft if the tip slips off for a moment
TURN_RATE = 60.0  # deg/s: how fast the tip then takes the claw round
PRESS_IN = 0.0003  # and how far into the claw's face it aims, to stay on it
STOP_SHORT_DEG = 4.0  # the tip stops this far short of where the claw would lie on the crown
LONGEST_S = 4.2  # the tip never stays on a claw longer than this: it would melt
TRIES = 3  # a claw that is not over the stone after a push is warmed and pushed again
HELD_OPEN = 0.013  # how far apart the tweezers' tips stand while the hand carries them empty
LANE = 0.065  # the working hand comes level with a standing tool this far to the ring's side of it (+y), slides in, and leaves the same way
FINGER_ROOM = (0.020, 0.015)  # how far each finger stands off a tool as the hand slides onto it, and as it slides away: no wider, the next tool is 95 mm off
OUT_OVER = (0.33, g.TOOL_ROW_Y + 0.09, 0.17)  # where it first goes from its start pose: out over the bench, clear of the row of tools under it
RIGHT_AWAY = (0.14, -0.20, 0.16)  # where the working hand ends, clear of everything (heights above the table)


def claw_out(i: int, angle: float) -> np.ndarray:
    """The way out of claw i's outside face (in M), at joint `angle`: pushing against it turns the claw toward the stone."""
    hinge, frame = g.claw_frame(i)
    tilt = angle - math.radians(g.CLAW_LEAN_DEG)
    return math.cos(tilt) * frame[:, 0] + math.sin(tilt) * frame[:, 2]


def claw_ball(i: int, angle: float) -> np.ndarray:
    """Where the pen tip's ball sits (in M) to rest on claw i's outside face, near its top, at joint `angle`."""
    hinge, frame = g.claw_frame(i)
    tilt = angle - math.radians(g.CLAW_LEAN_DEG)
    return hinge + claw_out(i, angle) * PUSH_FACE[0] + (-math.sin(tilt) * frame[:, 0] + math.cos(tilt) * frame[:, 2]) * PUSH_FACE[1]


def _report(ctx, env, label=""):
    now = TRIAL.wax.now
    st = m.state(env, "mandrel")
    stone = m.state(env, "stone")
    print(f"[wax] t {ctx.t:6.2f} {label}: score {now.get('score')}, in head {now.get('in_head')}, home {now.get('home')} (up {1000 * now.get('up', 0):+.2f} mm, off {1000 * now.get('off', 0):.2f} mm, "
          f"tilt {now.get('tilt', 0):.1f}), closed {now.get('n_closed')}, put back {now.get('put_back')} (pen {now.get('pen_home')}, tools {now.get('tools_home')}, mandrel "
          f"{now.get('mandrel_home')}, clear {now.get('clear')}, at rest {now.get('at_rest')}), mark {now.get('mark')}, zero "
          + (", ".join(k for k in ("melted", "broken", "fingered", "pen_down", "tipped", "off_table") if now.get(k)) or "no"), flush=True)
    print(f"[wax]   claws deg {[round(st.deg(f'claw{i}'), 1) for i in range(g.N_CLAWS)]}, heat {[round(h, 2) for h in TRIAL.wax.heat]}, seat {1000 * st.joint('seat'):+.2f} mm, "
          f"mandrel {np.round(st.pos - LIFT, 4)} roll {math.degrees(math.atan2(st.rotation()[0, 2], st.rotation()[2, 2])):+.1f} deg, its thin end "
          f"{1000 * (st.to_world(np.array([[0.0, g.Y_THIN + g.STICK_R_THIN, 0.0]]))[0][2] - TOP - g.STICK_R_THIN):.1f} mm up, stone {np.round(stone.pos - LIFT, 4)}", flush=True)
    if label.endswith(("start", "end")):  # where each loose part rests, against where it was designed to, and how far it has moved in the last second
        for name, home in (("mandrel", w.MANDREL_START), ("stone", w.STONE_START), ("tweezers", w.TWEEZERS_START), ("pusher", w.PUSHER_START), ("wax_pen", w.PEN_START)):
            part = m.state(env, name)
            seen = [poses[name] for poses in TRIAL.trail]
            shift = max((float(np.linalg.norm(p.pos - seen[-1].pos)) for p in seen), default=0.0)
            turn = max((math.degrees(math.acos(float(np.clip((np.trace(p.rot.T @ seen[-1].rot) - 1.0) / 2.0, -1.0, 1.0)))) for p in seen), default=0.0)
            print(f"[wax]   {name}: {np.round(1000 * (part.pos - LIFT - home.pos), 2)} mm from its place, tilted {part.tilt_deg():.1f} deg; in the last second it moved "
                  f"{1000 * shift:.3f} mm and turned {turn:.3f} deg; speed {1000 * float(np.linalg.norm(part.lin_vel)):.2f} mm/s, {float(np.linalg.norm(part.ang_vel)):.3f} rad/s", flush=True)


def demo_script(ctx):
    """The demo, with a watch on the tools that stand in their places: a line in the log whenever one of them is knocked."""
    env = ctx.emb._env
    leans = {}
    for _ in _demo(ctx):
        for name in ("wax_pen", "tweezers", "pusher"):
            part = m.state(env, name)
            lean = part.tilt_deg()
            if abs(lean - leans.get(name, 0.0)) > 1.5:
                print(f"[wax] t {ctx.t:6.2f} the {name} leans {lean:.1f} deg (it leant {leans.get(name, 0.0):.1f}); right hand at {np.round(ctx.right.tcp() - LIFT, 3)}, "
                      f"fingers {1000 * 2 * abs(ctx.right.finger):.0f} mm apart", flush=True)
                leans[name] = lean
        yield


def _demo(ctx):
    """Left hand: takes the mandrel by its thick end and holds it. Right hand: tweezers (stone into the head), pusher (stone pressed home), pen (its tip
    rested on each claw until the wax has softened, then the claw taken round over the stone), every tool put back. Failure checks (--mode): none; hover
    (the pen's tip 1.5 mm off every claw); cold (the claws pushed at once, without waiting); long (the pen kept on one claw); early (the claws closed
    before the stone is pressed); unheld (nobody holds the mandrel); shank (the pen laid on the ring's shank first). Three more check nothing: probe
    measures how exactly the hand places the pen's tip, claws runs the pen's part alone on an empty head, and tools takes each tool out and puts it back."""
    import demokit as d
    from yam_arena import ik
    from yam_arena.constants import FINGER_OPEN

    env = ctx.emb._env
    left, right = ctx.left, ctx.right
    mode = ctx.mode
    if mode not in MODES:
        raise ValueError(mode)
    grip_l, grip_r = d.tilted(HOLD_TILT_DEG, -90), d.down(90)  # both close their fingers along x; the holding hand's wrist leans away from the ring
    seen_at = {}

    def st(name):
        """A part's state, read once in each control step."""
        if seen_at.get(name, (None, None))[0] != ctx.t:
            seen_at[name] = (ctx.t, m.state(env, name))
        return seen_at[name][1]

    def in_m(p):
        return st("mandrel").to_world(np.asarray(p, float)[None, :])[0]

    def wait(n):
        for _ in range(n):
            yield

    def strain():
        """How far each hand stands from where it is sent (mm): a hand that is held back by something stands far off."""
        off = lambda arm: 1000 * (arm.tcp() - ik.tcp_pose(arm.q, arm.base)[0])  # noqa: E731
        return f"hands from their commands: left {np.round(off(left), 1)}, right {np.round(off(right), 1)} mm"

    def hold_still(arm):
        """Send the arm where it stands. A command left ahead of the joints goes on drawing them toward it for seconds after the tip has arrived."""
        arm.q = np.array(ctx.observed(arm.side)[:6], float)
        arm.cmd = np.r_[arm.q, float(np.clip(arm.finger / FINGER_OPEN, 0.0, 1.0))]

    def tip_servo(arm, tip, goal, tol=0.0004, timeout=3.0, gain=0.15, step=0.002, floor=0.00004, until=None, steady=2, abort=None, label="tip"):
        """Shift the arm's command toward where it brings `tip()` (a point on what it holds, read from its pose) to `goal` (a point, or a function giving
        one), until the tip has been within `tol` of it for `steady` steps running, or until `until()`. The command moves `gain` of the error a step, never
        more than `step` and never less than `floor`: the arm's joints stick until their command is a millimetre or two ahead of them, so a command that
        slowed with the error would stall. Once the tip is there the arm is sent where it stands, so that it stays. `abort()`, if it gives a reason, ends
        it at once; so does a command that has run 15 mm ahead of the tip or out of the arm's reach, which means the tip is held by something."""
        c = ik.tcp_pose(arm.q, arm.base)[0]
        lead0 = c - tip()
        err, good = np.zeros(3), 0
        for _ in range(max(1, round(timeout * d.CONTROL_HZ))):
            target = np.asarray(goal() if callable(goal) else goal, float)
            now = tip()
            err = target - now
            n = float(np.linalg.norm(err))
            if until is not None and until():
                return True
            why = abort() if abort is not None else ""
            if not why and float(np.linalg.norm(c - now - lead0)) > 0.015:
                why = f"the command is {1000 * float(np.linalg.norm(c - now - lead0)):.0f} mm ahead of the tip"
            if why:
                print(f"[wax] {label}: given up, {why}; {strain()}", flush=True)
                return False
            if until is None and n < tol:
                if not good:
                    hold_still(arm)
                    c = ik.tcp_pose(arm.q, arm.base)[0]
                    lead0 = c - now
                good += 1
                if good >= steady:
                    return True
            else:
                good = 0
                c = c + err / max(n, 1e-12) * min(step, max(gain * n, floor))
                miss = arm._solve(c, arm.grip.rot)
                if miss[0] > 3.0:
                    print(f"[wax] {label}: given up, the command is {miss[0]:.1f} mm out of the arm's reach at {np.round(c - LIFT, 3)}; {strain()}", flush=True)
                    return False
            yield
        print(f"[wax] {label}: stopped {1000 * float(np.linalg.norm(err)):.2f} mm off" + (f"; {strain()}" if float(np.linalg.norm(err)) > 0.003 else ""), flush=True)
        return False

    def stopped(read, span, least):
        """A check for `until`: what `read()` gives has fallen by less than `least` over the last `span` steps."""
        seen = []

        def check():
            seen.append(read())
            return len(seen) > span and seen[-span - 1] - seen[-1] < least

        return check

    def fingers_to(arm, width, t=0.4):
        yield from arm.move(arm.at(), finger=float(np.clip(-width / 2, FINGER_OPEN, 0.0)), t=t, label="fingers")

    def nudge_fingers(arm, by):
        arm.finger = float(np.clip(arm.finger + by, FINGER_OPEN, 0.0))
        arm.cmd = np.r_[arm.q, float(np.clip(arm.finger / FINGER_OPEN, 0.0, 1.0))]

    def tw_close_deg():
        s = st("tweezers")
        return (s.deg("leg_b") - s.deg("leg_a")) / 2

    def squeeze_to(deg, tol=0.04, timeout=2.5):
        """Close or open the fingers on the tweezers' legs until the legs have turned `deg` toward each other."""
        for _ in range(round(timeout * d.CONTROL_HZ)):
            e = deg - tw_close_deg()
            if abs(e) < tol:
                return
            nudge_fingers(right, float(np.clip(0.5 * math.radians(e) * (g.TW_HINGE_Z - g.TW_GRASP_Z), -0.0002, 0.0002)))
            yield
        print(f"[wax] tweezers: legs at {tw_close_deg():.2f} deg, wanted {deg:.2f}", flush=True)

    def stone_up():
        """How far the stone's girdle stands above the head's seat line, from the two poses."""
        return float(st("mandrel").to_local(st("stone").pos[None, :])[0][2] - g.Z_S)

    tw_tips = lambda: st("tweezers").pos  # noqa: E731
    pen_tip = lambda: st("wax_pen").pos  # noqa: E731
    pen_ball = lambda: st("wax_pen").to_world(np.array(g.PEN_HOT[0])[None, :])[0]  # noqa: E731
    pusher_face = lambda: st("pusher").pos  # noqa: E731
    hold_deg = math.degrees(math.atan((g.TW_GAP / 2 - HELD_OPEN / 2) / g.TW_HINGE_Z))  # the squeeze that carries them: tips 13 mm apart, 2.5 mm clear of the stone a side
    grip_deg = math.degrees(math.atan((g.TW_GAP / 2 - g.STONE_R) / g.TW_HINGE_Z))  # tips on the girdle

    yield from wait(10)
    _report(ctx, env, f"{mode} start")
    if mode == "none":
        yield from wait(40)
        return

    # 1. the left hand takes the mandrel by its thick end and holds it
    if mode != "unheld":
        grasp = in_m((0.0, g.GRASP_Y, 0.0))
        yield from left.move(grasp + 0.07 * UP, grip=grip_l, finger=-0.025, t=2.0, label="over the mandrel's thick end")
        yield from left.servo(grasp + 0.07 * UP, tol=0.003, timeout=2.0, label="square over the stick")
        yield from left.move(grasp, t=1.5, label="fingers either side of the stick")
        yield from left.servo(in_m((0.0, g.GRASP_Y, 0.0)), tol=0.002, timeout=2.0, label="centred on the stick")
        yield from left.close(2 * g.stick_r(g.GRASP_Y), 15.0, t=0.6, label="hold the mandrel")
        yield from wait(6)
    _report(ctx, env, "mandrel held")

    def lean_in_hand(name):
        """How far a held tool leans from the gripper's axis, deg."""
        rot = ik.tcp_pose(ctx.observed("right")[:6], right.base)[1]
        return math.degrees(math.acos(float(np.clip(-st(name).rotation()[:, 2] @ rot[:, 2], -1.0, 1.0))))

    def take_standing(tool, grasp_z, width, label):
        """Come level with a standing tool from the ring's side of the row, fingers open either side of it, slide in and centre on it."""
        at = lambda: st(tool).to_world(np.array([0.0, 0.0, grasp_z])[None, :])[0]  # noqa: E731
        away = at() + np.array([0.0, LANE, 0.0])
        yield from right.move(away + 0.03 * UP, grip=grip_r, finger=max(-(width / 2 + FINGER_ROOM[0]), FINGER_OPEN), t=2.0, label=f"beside the {label}")
        yield from right.move(away, t=0.8, label=f"level with the {label}")
        yield from right.servo(away, tol=0.002, timeout=2.0, label=f"square beside the {label}")
        yield from right.move(at(), t=1.6, label=f"fingers either side of the {label}")
        yield from right.servo(at(), tol=0.001, timeout=3.0, label=f"centred on the {label}")
        hold_still(right)

    def stands(name, home, what):
        part = st(name)
        print(f"[wax]   {name} {what}: {np.round(1000 * (part.pos - home), 2)} mm from its place, tilted {part.tilt_deg():.1f} deg", flush=True)

    def put_standing(tip, home, mouth, label, name, width, seat=None):
        """Bring a held tool's lower end (`tip`) over its place by way of the lane (`mouth`: the top of the funnel it goes into), lower it until it rests, let
        go and leave by the lane. `seat`, a point on the tool and where that point rests, is what is centred and lowered when it is not the lower end:
        the pen's handle, whose tip hangs below it. A tool that catches on the way down is lifted a little, centred again and lowered again."""
        over = np.array([home[0], home[1], mouth + 0.003])
        point, rest = (tip, home) if seat is None else seat
        yield from right.move(right.at() + np.array([0.0, 0.0, max(0.0, over[2] - tip()[2])]), t=0.8, label=f"the {label} up")
        yield from tip_servo(right, tip, over + np.array([0.0, LANE, 0.0]), tol=0.003, timeout=6.0, step=0.004, label=f"the {label} to the lane")
        yield from tip_servo(right, tip, over, tol=0.002, timeout=5.0, step=0.003, label=f"the {label} over its place")
        level = point()[2]
        for attempt in range(3):
            yield from tip_servo(right, point, np.array([rest[0], rest[1], level]), tol=0.0006, timeout=4.0, steady=3, label=f"the {label} square over its place")
            stands(name, home, "over its place")
            stuck = stopped(lambda: point()[2], 10, 0.0004)
            resting = lambda: point()[2] < rest[2] + 0.0015 or stuck()  # noqa: E731
            yield from tip_servo(right, point, lambda: np.array([rest[0], rest[1], max(rest[2], point()[2] - 0.003)]), timeout=6.0, step=0.0008, until=resting, label=f"the {label} down into its place")
            if point()[2] < rest[2] + 0.004 or attempt == 2:
                break
            print(f"[wax] the {label} caught {1000 * (point()[2] - rest[2]):.1f} mm above its place, {np.round(1000 * (point()[:2] - rest[:2]), 2)} mm off it: lifted and lowered again", flush=True)
            level = point()[2] + 0.004
            yield from tip_servo(right, point, point() + 0.004 * UP, tol=0.001, timeout=2.0, step=0.001, label=f"the {label} up a little")
        stands(name, home, "lowered")
        hold_still(right)  # the hand stops pressing before it lets go
        yield from right.move(right.at(), finger=max(-(width / 2 + FINGER_ROOM[1]), FINGER_OPEN), t=0.5, label=f"let go of the {label}")
        right.trim = np.zeros(3)
        yield from wait(4)
        yield from right.move(right.at() + np.array([0.0, LANE, 0.0]), t=1.4, label=f"away from the {label}")
        stands(name, home, "left standing")

    def legs_across(z):
        """How far apart the outsides of the tweezers' legs stand, `z` above their tips, when nothing squeezes them."""
        return g.TW_ROOT + (g.TW_GAP - g.TW_ROOT) * (1.0 - z / g.TW_HINGE_Z) + 2 * g.TW_LEG_T

    def take_tweezers():
        yield from take_standing("tweezers", g.TW_GRASP_Z, legs_across(g.TW_GRASP_Z), "tweezers")
        yield from fingers_to(right, legs_across(g.TW_GRASP_Z) - 0.0003)
        yield from squeeze_to(hold_deg)
        yield from right.move(right.at() + 0.042 * UP, t=1.6, label="the tweezers out of their trough")
        print(f"[wax]   tweezers in the hand: lean {lean_in_hand('tweezers'):.1f} deg, legs closed {tw_close_deg():.2f} deg (carried at {hold_deg:.2f})", flush=True)

    def put_tweezers():
        yield from put_standing(tw_tips, np.array([*g.TWEEZER_XY, TOP + g.TW_REST_Z]), TOP + g.TOOL_BLOCK[2] + g.MOUTH_RISE, "tweezers", "tweezers", legs_across(g.TW_GRASP_Z))

    def from_tips(point):
        """Where a point lies from the middle of the tweezers' tips' ends, in the tweezers' own frame: across the tips, along their width, up."""
        tw = st("tweezers")
        return tw.rotation().T @ (np.asarray(point, float) - tw.pos)

    def set_stone():
        # 2. tweezers: the stone out of its block and into the head
        yield from take_tweezers()
        block = TOP + g.STONE_BLOCK[2]
        for attempt in range(4):  # the tips come down either side of the stone; if they stand off it, or have pushed it, they go up and come down again
            was = st("stone").pos.copy()
            yield from tip_servo(right, tw_tips, np.array([was[0], was[1], block + 0.014]), tol=0.002, timeout=5.0, step=0.004, label="tweezers over the stone")
            yield from tip_servo(right, tw_tips, np.array([was[0], was[1], block + 0.0045]), tol=0.0005, timeout=5.0, step=0.0008, steady=3, label="tips over the stone")
            off = from_tips(was)
            if abs(off[0]) > 0.0015 or abs(off[1]) > 0.0006:
                print(f"[wax] tips over the stone: it lies {np.round(1000 * off[:2], 2)} mm from their middle (across, along): up and again", flush=True)
                continue
            yield from tip_servo(right, tw_tips, np.array([was[0], was[1], block + 0.0004]), tol=0.0004, timeout=4.0, step=0.0004, label="tips down either side of the stone")
            off, moved = from_tips(st("stone").pos), float(np.linalg.norm(st("stone").pos - was))
            if abs(off[0]) < 0.002 and abs(off[1]) < 0.0007 and moved < 0.0004 and tw_tips()[2] < block + 0.001:
                break
            print(f"[wax] tips down: the stone lies {np.round(1000 * off[:2], 2)} mm from their middle, has moved {1000 * moved:.2f} mm, tips {1000 * (tw_tips()[2] - block):.2f} mm above the block: "
                  "up and again", flush=True)
        hold_still(right)
        yield from squeeze_to(grip_deg - 0.03, timeout=3.0)
        nudge_fingers(right, 0.00015)  # a little more: the pinch that carries it
        yield from wait(6)
        yield from right.move(right.at() + 0.035 * UP, t=1.6, label="the stone up")
        _report(ctx, env, "stone lifted")
        gem = lambda: st("stone").pos  # noqa: E731
        over = lambda h: in_m((0.0, 0.0, g.Z_S + g.SEAT_TRAVEL + h))  # noqa: E731
        yield from tip_servo(right, gem, lambda: over(0.02), tol=0.002, timeout=5.0, step=0.004, label="the stone over the ring")
        yield from tip_servo(right, gem, lambda: over(0.006), tol=0.0004, timeout=4.0, steady=4, label="the stone over the head")
        yield from tip_servo(right, gem, lambda: over(0.0006), tol=0.0003, timeout=4.0, step=0.0003, steady=4, label="the stone down between the claws")
        hold_still(right)
        yield from squeeze_to(hold_deg, timeout=2.0)
        yield from wait(8)
        yield from right.move(right.at() + 0.03 * UP, t=1.2, label="the tweezers up")
        _report(ctx, env, "stone laid in")
        yield from put_tweezers()

    def take_pusher():
        yield from take_standing("pusher", g.PU_GRASP_Z, 2 * g.PU_KNOB_R, "pusher")
        yield from right.close(2 * g.PU_KNOB_R, 12.0, t=0.5, label="take the pusher")
        yield from right.move(right.at() + 0.035 * UP, t=1.6, label="the pusher out of its hole")
        print(f"[wax]   pusher in the hand: lean {lean_in_hand('pusher'):.1f} deg", flush=True)

    def put_pusher():
        yield from put_standing(pusher_face, np.array([*g.PUSHER_XY, TOP + g.PU_REST_Z]), TOP + g.TOOL_BLOCK[2] + g.MOUTH_RISE, "pusher", "pusher", 2 * g.PU_KNOB_R)

    def press_stone():
        # 3. pusher: the stone pressed home
        yield from take_pusher()
        table = lambda h: st("stone").to_world(np.array([0.0, 0.0, g.TABLE_OVER_GIRDLE + h])[None, :])[0]  # noqa: E731
        yield from tip_servo(right, pusher_face, lambda: table(0.02), tol=0.002, timeout=5.0, step=0.004, label="the pusher over the stone")
        yield from tip_servo(right, pusher_face, lambda: table(0.002), tol=0.0008, timeout=4.0, label="the pusher's face over the table")
        still = stopped(stone_up, 12, 0.00003)
        down = lambda: still() and stone_up() < 0.0006  # noqa: E731  (it has gone down and stopped: it is on its bearing)
        yield from tip_servo(right, pusher_face, lambda: table(-0.001), timeout=4.0, step=0.00012, gain=0.3, until=down, label="press the stone home")
        hold_still(right)
        yield from right.move(right.at() + 0.03 * UP, t=1.0, label="the pusher up")
        _report(ctx, env, "stone pressed")
        yield from put_pusher()

    def take_pen():
        yield from take_standing("wax_pen", g.PEN_GRASP_Z, 2 * g.PEN_HANDLE_R, "pen")
        yield from right.close(2 * g.PEN_HANDLE_R, 25.0, t=0.5, label="take the pen")  # firmly: the claws push back on its tip, 125 mm below
        yield from right.move(right.at() + 0.047 * UP, t=2.4, label="the pen out of its stand")  # slowly: dragged against its stand, it would turn in the fingers
        print(f"[wax]   pen in the hand: lean {lean_in_hand('wax_pen'):.1f} deg", flush=True)

    def close_claw(i, long=False, trace=False):
        """The tip beside claw i, then onto its outside face and resting there until the wax has softened, then round with the claw along the path its face
        takes, until the claw is over the stone. The tip is steered by where it is on the claw, read from the two poses at every step."""
        deg = lambda: st("mandrel").deg(f"claw{i}")  # noqa: E731
        turn = lambda v: st("mandrel").rotation() @ v  # noqa: E731
        hinge, frame = g.claw_frame(i)
        aim = lambda a, into: in_m(claw_ball(i, math.radians(a))) - into * turn(claw_out(i, math.radians(a)))  # noqa: E731  (the ball on the face of the claw at `a` degrees, `into` deep)
        above = lambda a: aim(a, 0.0) + APPROACH[0] * turn(frame[:, 0]) + APPROACH[1] * UP  # noqa: E731
        off = lambda: tip_servo(right, pen_ball, lambda: above(deg()), tol=0.002, timeout=3.0, step=0.003, label="the pen off the claw")  # noqa: E731
        stop, top = math.degrees(g.CLAW_ON_CROWN) - STOP_SHORT_DEG, math.degrees(g.CLAW_ON_CROWN) + 4.0
        over = math.degrees(g.CLAW_CLOSED) + 6.0  # the claw is over the stone's edge, with something to spare

        def on_claw():
            """The tip's ball against the claw as it stands: off its outside face, across it, up it (m)."""
            centre, rot, half = g.claw_box(i, math.radians(deg()))
            loc = rot.T @ (st("mandrel").to_local(pen_ball()[None, :])[0] - centre)
            return loc[0] - half[0] - g.PEN_TIP_R, loc[1], loc[2] + half[2]

        def show(what, since, was):
            gap, across, up = on_claw()
            print(f"[wax]   claw {i} {what} t {since:5.2f} angle {deg() - was:6.2f}; ball off the face {1000 * gap:+.2f}, across {1000 * across:+.2f}, up the claw {1000 * up:.2f} mm; "
                  f"heat {TRIAL.wax.heat[i]:.2f}", flush=True)

        def one_try():
            was, ring0 = deg(), in_m((0.0, 0.0, g.Z_S))
            def pushed():
                """Why the approach must stop: the claw or the ring has been moved."""
                moved = 1000 * float(np.linalg.norm(in_m((0.0, 0.0, g.Z_S)) - ring0))
                return f"claw {i} has turned {deg() - was:+.1f} deg" if abs(deg() - was) > 1.5 else f"the ring has moved {moved:.1f} mm" if moved > 1.0 else ""

            wide = lambda: aim(was, 0.0) + APPROACH[0] * turn(frame[:, 0])  # noqa: E731  (level with the tip's place on the claw, outside it)
            beside = lambda: aim(was, -STAND_OFF)  # noqa: E731
            for attempt in range(3):  # over the claw's outside, down well clear of it, then in: slowly, so that an arm that runs on touches nothing
                ok = yield from tip_servo(right, pen_ball, lambda: wide() + APPROACH[1] * UP, tol=0.0015, timeout=12.0, step=0.0008, steady=3, abort=pushed, label=f"the pen above claw {i}")
                ok = ok and (yield from tip_servo(right, pen_ball, wide, tol=0.001, timeout=8.0, step=0.0008, steady=3, abort=pushed, label=f"the tip level with claw {i}"))
                ok = ok and (yield from tip_servo(right, pen_ball, beside, tol=0.0005, timeout=8.0, step=0.0003, steady=3, abort=pushed, label=f"the tip beside claw {i}"))
                yield from wait(6)
                miss = beside() - pen_ball()
                if ok and float(np.linalg.norm(miss)) < 0.0008:
                    break
                yield from tip_servo(right, pen_ball, pen_ball() + np.array([0.0, 0.0, max(0.0, (wide() + APPROACH[1] * UP - pen_ball())[2])]), tol=0.002, timeout=3.0, step=0.001, label="the pen up again")
            else:
                print(f"[wax] claw {i}: the tip did not come to rest beside it ({np.round(1000 * miss, 2)} mm off): left alone", flush=True)
                return
            if mode == "hover":
                yield from tip_servo(right, pen_ball, lambda: aim(was, -0.0015), tol=0.0005, timeout=4.0, label=f"the tip 1.5 mm off claw {i}")
                yield from wait(50)
                return
            clock = {"near": None, "k": 0}

            def watch(what, done):
                """An `until` for the servo: notes when the wax starts to warm, prints the trace, and ends when `done()`."""
                def check():
                    if clock["near"] is None and on_claw()[0] < 0.0005:
                        clock["near"] = ctx.t
                    clock["k"] += 1
                    if trace and clock["k"] % 4 == 0:
                        show(what, 0.0 if clock["near"] is None else ctx.t - clock["near"], was)
                    return done()
                return check

            since = lambda: 0.0 if clock["near"] is None else ctx.t - clock["near"]  # noqa: E731
            met = yield from tip_servo(right, pen_ball, lambda: aim(was, 0.0003), timeout=5.0, step=0.0001, gain=0.3, floor=0.00005,
                                       until=watch("in  ", lambda: deg() - was > 0.3 or on_claw()[0] < 0.00015), label=f"the tip onto claw {i}")
            if not met:
                print(f"[wax] claw {i}: the tip did not meet it: left alone", flush=True)
                return
            if mode != "cold":  # the tip rests on the claw while the wax softens, without leaning on it; pushed at once, a cold claw bends a little and springs back
                resting = watch("rest", lambda: since() >= (6.3 if long else WAIT_S))
                c = ik.tcp_pose(right.q, right.base)[0]
                while not resting():
                    if on_claw()[0] > 0.00018:  # it has drifted off: back onto the face, slowly
                        c = c - 0.00003 * turn(claw_out(i, math.radians(deg())))
                        right._solve(c, right.grip.rot)
                    yield
            if not long:
                a0, t1 = deg(), ctx.t
                yield from tip_servo(right, pen_ball, lambda: aim(max(min(a0 + TURN_RATE * (ctx.t - t1), top), deg()), PRESS_IN), timeout=3.0, step=0.0005, gain=0.25, floor=0.00005,
                                     until=watch("push", lambda: deg() >= stop or since() > LONGEST_S or ctx.t - t1 > (0.9 if mode == "cold" else 2.6)), label=f"claw {i} round")
            if trace:
                show("off ", since(), was)

        for attempt in range(TRIES):
            before = deg()
            yield from one_try()
            yield from off()
            if deg() >= over or long or mode in ("hover", "cold"):
                break
            print(f"[wax] claw {i}: at {deg():.1f} deg after try {attempt + 1}, not yet over the stone", flush=True)
            if attempt and deg() - before < 3.0:
                break  # it goes no further: something is under it
            yield from wait(20)  # it cools before it is warmed again
        _report(ctx, env, f"claw {i} " + ("pushed cold" if mode == "cold" else "kept under the pen" if long else "done"))

    def put_pen():
        handle = lambda: st("wax_pen").to_world(np.array([0.0, 0.0, g.PEN_TIP_LEN])[None, :])[0]  # noqa: E731  (the handle's lower end: it is what the stand's bore takes)
        yield from put_standing(pen_tip, np.array([*g.PEN_STAND_XY, TOP + g.PEN_REST_Z]), TOP + g.PEN_TUBE[2] + g.MOUTH_RISE, "pen", "wax_pen", 2 * g.PEN_HANDLE_R,
                                seat=(handle, np.array([*g.PEN_STAND_XY, TOP + g.PEN_LEDGE_H])))

    def touch_shank():
        reach = g.RING_OUT_R + g.PEN_TIP_R
        spot = lambda out=0.0, up=0.0: in_m((reach * math.sin(math.radians(60.0)) + out, 0.0, reach * math.cos(math.radians(60.0)) + up))  # noqa: E731  (on the shank's side, below the head)
        on_shank = lambda: g.wax_body_gap(st("mandrel").to_local(pen_ball()[None, :])[0]) - g.PEN_TIP_R < 0.00015  # noqa: E731
        yield from tip_servo(right, pen_ball, lambda: spot(0.006, 0.014), tol=0.002, timeout=10.0, step=0.0008, label="the pen above the shank")
        yield from tip_servo(right, pen_ball, lambda: spot(0.006), tol=0.001, timeout=8.0, step=0.0008, label="the pen beside the shank")
        yield from tip_servo(right, pen_ball, lambda: spot(-0.0005), timeout=6.0, step=0.0002, until=on_shank, label="the tip onto the shank")
        yield from wait(30)
        yield from tip_servo(right, pen_ball, lambda: spot(0.008, 0.012), tol=0.002, timeout=3.0, step=0.003, label="the pen off the shank")

    def probe():
        """Not a check of the task: it measures how exactly the hand can place the pen's tip, in free air beside the ring, and prints it."""
        yield from take_pen()
        home = in_m((0.012, -0.012, g.Z_S + 0.012))

        def look():
            """The observed TCP, the pen's tip ball, the pen's grasp point as the hand sees it, and the pen's lean from the gripper's axis (deg)."""
            tcp, rot = ik.tcp_pose(ctx.observed("right")[:6], right.base)
            pen = st("wax_pen")
            grasp = pen.to_world(np.array([0.0, 0.0, g.PEN_GRASP_Z])[None, :])[0]
            lean = math.degrees(math.acos(float(np.clip(-pen.rotation()[:, 2] @ rot[:, 2], -1.0, 1.0))))
            return tcp, pen_ball(), rot.T @ (grasp - tcp), lean

        t1 = ctx.t
        ok = yield from tip_servo(right, pen_ball, home, tol=0.0003, timeout=10.0, steady=4, label="probe: to its start")
        tcp, ball, hand, lean = look()
        print(f"[wax] probe: to its start {'in' if ok else 'not within'} {ctx.t - t1:.2f} s, {1000 * float(np.linalg.norm(home - ball)):.2f} mm off; command less observed TCP "
              f"{np.round(1000 * (ik.tcp_pose(right.q, right.base)[0] - tcp), 1)} mm; grasp point in the hand {np.round(1000 * hand, 1)} mm, pen leans {lean:.2f} deg from the gripper's axis", flush=True)
        rows = []
        for _ in range(40):  # held still for 2 s: does the tip stay put
            yield
            rows.append(look())
        tips, tcps = np.array([r[1] for r in rows]), np.array([r[0] for r in rows])
        print(f"[wax] probe: held still 2 s: tip moves {np.round(1000 * np.ptp(tips, axis=0), 2)} mm peak to peak, TCP {np.round(1000 * np.ptp(tcps, axis=0), 2)} mm, "
              f"lean {min(r[3] for r in rows):.2f} to {max(r[3] for r in rows):.2f} deg", flush=True)
        for name, v, speed in (("x", (1, 0, 0), 0.003), ("z", (0, 0, 1), 0.003)):
            # the command goes 6 mm out and back (3 mm when slow): what the hand and the tip do, every quarter second
            c0, v, per = ik.tcp_pose(right.q, right.base)[0], np.array(v, float), speed / d.CONTROL_HZ
            tcp0, ball0, hand0, lean0 = look()
            top = 40 if speed > 0.002 else 60
            print(f"[wax] probe: command along {name} at {1000 * speed:g} mm/s, out and back. Columns: command, observed TCP, tip (mm along it); tip sideways; grasp point's shift in the hand (mm); lean change (deg)", flush=True)
            for k in range(1, 2 * top + 41):
                out = per * (min(k, 2 * top - k) if k <= 2 * top else 0)
                right._solve(c0 + out * v, right.grip.rot)
                yield
                if k % 5 == 0:
                    tcp, ball, hand, lean = look()
                    side = (ball - ball0) - ((ball - ball0) @ v) * v
                    print(f"[wax]   {1000 * out:5.2f} {1000 * float((tcp - tcp0) @ v):6.2f} {1000 * float((ball - ball0) @ v):6.2f}; {1000 * float(np.linalg.norm(side)):5.2f}; "
                          f"{np.round(1000 * (hand - hand0), 2)}; {lean - lean0:+.2f}", flush=True)
        for gain, floor in ((0.15, 0.00004), (0.10, 0.00002), (0.25, 0.00006)):  # which servo law places the tip best
            for name, v, size in (("+x", (1, 0, 0), 0.003), ("-x", (-1, 0, 0), 0.003), ("+y", (0, 1, 0), 0.0006), ("-y", (0, -1, 0), 0.0006), ("-z", (0, 0, -1), 0.003), ("+z", (0, 0, 1), 0.0006),
                                  ("+x+y-z", (0.6, 0.6, -0.5), 0.004)):
                goal = pen_ball() + size * np.array(v, float)
                t1, seen = ctx.t, []
                ok = yield from tip_servo(right, pen_ball, goal, tol=0.00025, timeout=8.0, steady=4, gain=gain, floor=floor, label=f"probe: {1000 * size:g} mm {name}")
                took = ctx.t - t1
                for _ in range(20):
                    yield
                    seen.append(float(np.linalg.norm(goal - pen_ball())))
                print(f"[wax] probe: gain {gain}, floor {1000 * floor:g} mm: {1000 * size:g} mm {name}: {'in' if ok else 'not within'} {took:.2f} s; over the next second {1000 * min(seen):.2f} to "
                      f"{1000 * max(seen):.2f} mm off", flush=True)
        yield from put_pen()

    def close_claws():
        yield from take_pen()
        yield from tip_servo(right, pen_ball, lambda: in_m((0.0, -0.040, g.Z_S + 0.011)), tol=0.003, timeout=10.0, step=0.003, label="the pen toward the ring")  # 4 cm short of it, over the stick
        if mode == "shank":
            yield from touch_shank()
        for k, i in enumerate(CLAW_ORDER):
            yield from close_claw(i, long=mode == "long" and k == 0, trace=mode == "claws")
        yield from put_pen()

    yield from right.move(np.array(OUT_OVER) + LIFT, grip=grip_r, t=1.5, label="out over the bench")
    if mode == "probe":
        yield from probe()
        return
    if mode == "claws":  # the pen's part alone, on an empty head: for tuning
        yield from close_claws()
        return
    if mode == "tools":  # each tool out of its place and back, twice: for tuning
        for _ in range(2):
            yield from take_tweezers()
            yield from put_tweezers()
            yield from take_pusher()
            yield from put_pusher()
            yield from take_pen()
            yield from put_pen()
        _report(ctx, env, "tools end")
        return
    yield from set_stone()
    if mode == "early":
        yield from close_claws()
        yield from press_stone()
    else:
        yield from press_stone()
        yield from close_claws()

    # the left hand lets go and both hands back away
    if mode != "unheld":
        yield from left.open(t=0.5, label="let go of the mandrel")
        yield from left.move(left.at() + 0.08 * UP, t=1.0, label="the left hand up")
        yield from left.move(left.at() + np.array([-0.06, 0.10, 0.0]), t=1.2, label="the left hand away")
    yield from right.move(np.array(RIGHT_AWAY) + LIFT, t=1.5, label="the right hand away")
    yield from wait(10)
    _report(ctx, env, f"{mode} end")
