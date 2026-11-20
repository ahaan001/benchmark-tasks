"""A built frame of 40 mm T-slot profile lies fastened to the table: a closed rectangle, so that the ends of the side bars' slots are shut.
Beside it waits a printed camera bracket: a foot with two M6 cap screws, each with a drop-in T-nut hanging on it lengthwise, an arm, and a
RealSense D405 camera on a side plate at the arm's end. A T-handle hex driver stands in a block.

The job is to bolt the bracket onto the bar that carries the blue line. Both nuts have to go down through the 8.13 mm opening of the slot,
out of sight under the foot. The camera end overhangs, so one hand must keep the foot flat on the bar while the other turns the screws:
as a screw starts to turn its nut swings a quarter turn under the slot's lips, where its boss stops it against the opening, and the
thread then draws it up tight. A nut that never went in, or a foot that tipped up before a nut gripped, leaves a screw that goes tight
on nothing.

Sizes are in tslot_geometry.py; the thread rule and the score in tslot_rule.py (no simulator needed for either). The parts are mechkit
parts; the demo runs on demokit:

    python demokit.py task.py OUT --mode full
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mechkit as m  # noqa: E402
import tslot_geometry as g  # noqa: E402
import tslot_rule as r  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
INSTRUCTION = ("Mount the camera bracket on the frame bar that has the blue line, with its arm pointing in over the frame. Pick the bracket up by its arm and lower "
               "its foot onto the bar so that both drop-in T-nuts, hanging lengthwise under the foot, go down into the bar's slot, with the notch in the foot on "
               "the line. Keep hold of the arm and keep the foot flat on the bar: the camera end overhangs, and until a nut grips the bracket tips and its nuts "
               "lift out. With the other gripper take the T-handle hex driver from its block and tighten both screws until they are snug. As a screw starts to "
               "turn, its nut swings a quarter turn under the lips of the slot and is then drawn up tight. If a screw keeps turning without tightening, or goes "
               "tight while the foot is still loose, back it off and try again. Put the driver down, then let go of the bracket.")
SUCCESS_HOLD_S = 0.0  # the score already asks for one still second
TOP = TABLE_TOP[SCENE]
UP = np.array([0.0, 0.0, TOP])
NAMES = ("a", "b")  # the two screws and their nuts: a nearer the robot once mounted, b farther

# Start variants for the mechanism tests in the README (never the submitted demo, which starts as the setup says):
# TSLOT_START=mounted has the bracket start on the marked bar, both nuts down in the slot, and the driver's block close by;
# TSLOT_START=askew has it start just above the bar with nut a turned 20 degrees off the slot's line;
# TSLOT_START=onlips has it start on the bar 4 mm off the slot, standing on its nuts on a lip, its arm propped level;
# TSLOT_START=swung is mounted with nut a already swung under the lips and drawn 0.6 mm up its screw, 0.1 mm short of them.
START = os.environ.get("TSLOT_START", "")
if START not in ("", "mounted", "askew", "onlips", "swung"):
    raise ValueError(f"TSLOT_START={START!r}")
ON_BAR = START in ("mounted", "swung")
BLOCK_XY = (0.20, -0.17) if ON_BAR else g.BLOCK_XY
DRIVER_YAW = 75.0 if ON_BAR else 0.0  # in those tests the driver stands turned, so that the hand takes it near the top of its stroke
SWUNG = (-308.0, 216.0)  # screw a and its nut on it, in degrees: the nut 92 degrees round, 0.6 turns up its thread
ASKEW_DEG, ASKEW_UP = 20.0, 0.003
ONLIPS = (0.001, 0.004)  # where the foot starts in the on-the-lips test, from where it belongs: the nuts on the inner lip by 3.7 mm
TRAVEL_ARMATURE = 0.2  # kg in the slide's own direction: lets the solver carry the thread's spring through the light nut

LOOKS = {name: m.Look(color=m.rgb(*rgb), metallic=0.6 if name in ("aluminium", "zinc", "steel") else 0.0, roughness=0.35 if name in ("aluminium", "zinc", "steel") else 0.7)
         for name, rgb in g.LOOKS_RGB.items()}


# -- the parts ------------------------------------------------------------------------------------------------------------------------------
def box_mesh(size, centre, rot=None) -> m.Mesh:
    mesh = m.box(size)
    v = mesh.v if rot is None else mesh.v @ np.asarray(rot, float).T
    return m.Mesh(v + np.asarray(centre, float), mesh.f)


def loft(rings) -> m.Mesh:
    """A closed solid through rings of equal length, lowest first, each anticlockwise seen from above."""
    rings = [np.asarray(ring, float) for ring in rings]
    n, k = len(rings[0]), len(rings)
    v = np.vstack(rings + [rings[0].mean(axis=0)[None], rings[-1].mean(axis=0)[None]])
    f = []
    for level in range(k - 1):
        a, b = level * n, (level + 1) * n
        for i in range(n):
            j = (i + 1) % n
            f += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    for i in range(n):
        j = (i + 1) % n
        f += [(k * n, j, i), (k * n + 1, (k - 1) * n + i, (k - 1) * n + j)]
    return m.Mesh(v, np.array(f))


def frame():
    """The four bars, each with its top slot as boxes (lips, upright walls, sloping walls, floor), the corner brackets and the line."""
    parts, cols = [], []
    for bar, axis, centre, span in g.frame_bars():
        for name, c, size, rot in g.bar_boxes(axis, centre, span):
            parts.append((f"{bar}_{name}", box_mesh(size, c, rot), "aluminium"))
            col = m.col_box(f"c_{bar}_{name}", size, c, rot)
            cols.append(col if name == "core" else m.fine(col))
    for k, (c, size) in enumerate(g.corner_boxes()):
        parts.append((f"corner{k}", box_mesh(size, c), "aluminium"))
        cols.append(m.col_box(f"c_corner{k}", size, c))
    lip = (g.BAR - g.SLOT_OPEN) / 2
    for k, s in enumerate((1.0, -1.0)):  # the line, drawn only: across both lips and down the bar's outer face
        parts.append((f"line{k}", box_mesh((g.LINE_W, lip, 0.0002), (g.LINE_X, g.MARKED_Y + s * (g.SLOT_OPEN / 2 + lip / 2), g.BAR + 0.0001)), "line"))
    parts.append(("line_face", box_mesh((g.LINE_W, 0.0002, g.BAR), (g.LINE_X, g.MARKED_Y - g.BAR / 2 - 0.0001, g.BAR / 2)), "line"))
    return m.fixed("frame", (0.0, 0.0, TOP), parts, cols, LOOKS, friction=(0.4, 0.35))


def supports():
    parts = [(f"strip{k}", box_mesh(size, c), "foam") for k, (c, size) in enumerate(g.strip_boxes())]
    cols = [m.col_box(f"c_strip{k}", size, c) for k, (c, size) in enumerate(g.strip_boxes())]
    return m.fixed("supports", (0.0, 0.0, TOP), parts, cols, LOOKS, friction=(0.9, 0.8))


def block():
    parts = [(f"block{k}", box_mesh(size, c), "beech") for k, (c, size) in enumerate(g.block_boxes())]
    cols = [m.col_box(f"c_block{k}", size, c) for k, (c, size) in enumerate(g.block_boxes())]
    return m.fixed("block", (BLOCK_XY[0], BLOCK_XY[1], TOP), parts, cols, LOOKS, friction=(0.6, 0.5))


def screw_link(name: str, x: float):
    """A screw: a hinge in the foot. Its head is a floor and six slabs round a 5 mm hex socket, so that only a hex blade in the socket turns it."""
    floor_h, z0 = g.HEAD_H - g.SOCKET_DEPTH, g.FOOT[2]
    parts = [(f"head_{name}", m.cylinder(g.HEAD_D / 2, floor_h, (x, 0.0, z0 + floor_h / 2), segments=24), "black"),
             (f"shank_{name}", m.cylinder(0.003, g.SCREW_END, (x, 0.0, -g.SCREW_END / 2), segments=12), "black")]
    cols = [m.col_cylinder(f"c_head_{name}", g.HEAD_D / 2, floor_h, (x, 0.0, z0 + floor_h / 2))]
    for k, ((cx, cy), (radial, tangent), turn) in enumerate(g.socket_walls()):
        c, size, rot = (x + cx, cy, z0 + floor_h + g.SOCKET_DEPTH / 2), (radial, tangent, g.SOCKET_DEPTH), g.rot_z(turn)
        parts.append((f"wall_{name}{k}", box_mesh(size, c, rot), "black"))
        cols.append(m.fine(m.col_box(f"c_wall_{name}{k}", size, c, rot)))
    return m.hinge(f"screw_{name}", "Z", (x, 0.0, 0.0), -math.inf, math.inf, parts, cols, mass=g.SCREW_MASS, spring=0.0, damping=1e-5, friction=r.T_HEAD,
                   armature=1e-6, inertia=(1e-7, 1e-7, 1e-7))


def nut_meshes(x: float, collide: bool) -> tuple:
    """A nut's flange and boss, hanging on the screw at `x` with its long side along the foot's. The colliding boss also runs down inside the
    flange, which makes its hull 4.5 mm thick and changes nothing outside."""
    zt = -g.NUT_TOP
    top = g.flange_outline(collide)
    reach = g.NUT_A / 2 - (g.NUT_F - g.NUT_BEVEL_FROM)
    low = np.c_[np.clip(top[:, 0], -reach, reach), top[:, 1]]
    shift = np.array([x, 0.0, 0.0])
    flange = loft([np.c_[low, np.full(len(low), zt - g.NUT_F)] + shift, np.c_[top, np.full(len(top), zt - g.NUT_BEVEL_FROM)] + shift, np.c_[top, np.full(len(top), zt)] + shift])
    boss = g.boss_outline(collide)
    under = zt - 0.003 if collide else zt
    return flange, loft([np.c_[boss, np.full(len(boss), under)] + shift, np.c_[boss, np.full(len(boss), zt + g.NUT_E)] + shift])


def nut_links(name: str, x: float) -> list:
    """The nut on its screw: a slide along the screw, which the thread rule drives, and on it a hinge with the thread's friction."""
    flange, boss = nut_meshes(x, collide=False)
    c_flange, c_boss = nut_meshes(x, collide=True)
    travel = m.slider(f"travel_{name}", "Z", g.TRAVEL[0], g.TRAVEL[1], (), (), at=(x, 0.0, 0.0), parent=f"screw_{name}", mass=0.001, spring=r.K_THREAD,
                      damping=2.0 * math.sqrt(r.K_THREAD * TRAVEL_ARMATURE), friction=0.0, effort=150.0, armature=TRAVEL_ARMATURE, inertia=(1e-7, 1e-7, 1e-7))
    nut = m.hinge(f"nut_{name}", "Z", (x, 0.0, 0.0), -math.inf, math.inf, [(f"flange_{name}", flange, "zinc"), (f"boss_{name}", boss, "zinc")],
                  [m.fine(m.col_hull(f"c_flange_{name}", c_flange)), m.fine(m.col_hull(f"c_boss_{name}", c_boss))], parent=f"travel_{name}", mass=g.NUT_MASS,
                  spring=0.0, damping=1e-5, friction=r.T_RUN, armature=2e-7, inertia=(5e-8, 5e-8, 1e-7))
    return [travel, nut]


def bracket_start() -> tuple:
    """(origin above the table, yaw) of the bracket at a reset."""
    if ON_BAR:
        return np.array([g.MOUNT_XY[0], g.MOUNT_XY[1], g.BAR]), g.MOUNT_YAW
    if START == "askew":
        return np.array([g.MOUNT_XY[0], g.MOUNT_XY[1], g.BAR + g.SCREW_END + ASKEW_UP]), g.MOUNT_YAW
    if START == "onlips":
        return np.array([g.MOUNT_XY[0] + ONLIPS[0], g.MOUNT_XY[1] + ONLIPS[1], g.BAR + g.SCREW_END]), g.MOUNT_YAW
    return np.array([g.START_XY[0], g.START_XY[1], g.SUPPORT_H]), g.START_YAW


def bracket():
    """The bracket: one free body (foot, arm, side plate, camera) carrying two screws on hinges and a nut on each."""
    mass, com, inertia = g.bracket_mass()
    boxes = g.bracket_boxes()
    parts = [("foot", m.prism(g.foot_outline(), 0.0, g.FOOT[2]), "petg")] + [(name, box_mesh(size, c), look) for name, c, size, look in boxes if name != "foot"]
    lens_c = (g.ARM_W / 2 + g.PLATE[0] + g.CAM[0] + 0.0003, g.ARM_END - g.CAM[1] / 2, g.ARM_H - g.CAM[2] / 2)
    parts.append(("lens", box_mesh((0.0006, 0.034, 0.012), lens_c), "lens"))
    cols = []
    for name, c, size, _ in boxes:
        col = m.col_box(f"c_{name}", size, c)
        cols.append(m.patched(col) if name == "arm" else col)  # a hand holds it by the arm's two sides
    links = []
    for name, sc in zip(NAMES, g.SCREWS):
        links.append(screw_link(name, float(sc[0])))
        links += nut_links(name, float(sc[0]))
    origin, yaw = bracket_start()
    return m.mechanism("bracket", tuple(origin + UP + [0.0, 0.0, 0.0003]), links, LOOKS, frame_parts=parts, frame_colliders=cols, yaw_deg=yaw, friction=(0.4, 0.35),
                       solver_iters=64, free_mass=mass, frame_inertia=tuple(float(v) for v in inertia), frame_com=tuple(float(v) for v in com))


def driver():
    """The T-handle hex driver: a 5 mm hex blade with a chamfered tip, a neck that a hand takes by its two flat sides, the bar and the short arm."""
    pts = g.blade_points()
    blade = loft([pts[0:6], pts[6:12], pts[12:18]])
    parts = [("blade", blade, "steel")] + [(name, box_mesh(size, c), look) for name, c, size, look in g.driver_boxes()]
    parts.append(("band", box_mesh((g.NECK[0] + 0.0006, g.NECK[1] + 0.0006, 0.004), (0.0, 0.0, g.BLADE + 0.004)), "green"))
    cols = [m.col_hull("c_blade", blade)] + [m.col_box(f"c_{name}", size, c) for name, c, size, _ in g.driver_boxes()]
    tip = np.array([BLOCK_XY[0], BLOCK_XY[1], g.DRIVER_START[2]])
    return m.free("driver", tuple(tip + UP + [0.0, 0.0, 0.0005]), parts, cols, LOOKS, mass=g.DRIVER_MASS, friction=(1.0, 0.9), torsional_patch=0.015,
                  yaw_deg=DRIVER_YAW, com=(0.0, 0.0, g.DRIVER_COM))


FRAME, SUPPORTS, BLOCK, BRACKET, DRIVER = frame(), supports(), block(), bracket(), driver()
PROP_H = g.BAR + (g.SCREW_END if START == "onlips" else 0.0)
PROP = m.fixed("prop", (g.MOUNT_XY[0] - 0.006, g.MOUNT_XY[1] + (ONLIPS[1] if START == "onlips" else 0.0) + 0.150, TOP), [("prop", box_mesh((0.016, 0.012, PROP_H), (0.0, 0.0, PROP_H / 2)), "foam")],
               [m.col_box("c_prop", (0.016, 0.012, PROP_H), (0.0, 0.0, PROP_H / 2))], LOOKS, friction=(0.9, 0.8))  # holds the arm level in the tests that start on the bar, where no hand does; it stands clear of the camera's plate
OBJECTS = [FRAME, BLOCK, BRACKET, DRIVER] + ([PROP] if ON_BAR or START == "onlips" else [] if START else [SUPPORTS])


# -- reading the scene ----------------------------------------------------------------------------------------------------------------------
NUT_CENTRES = [np.array([sc[0], 0.0, -g.NUT_TOP]) for sc in g.SCREWS]  # the middle of each flange's top face, in the bracket's frame at joint zero
DRIVER_POINTS = g.driver_points()


def states(env, names) -> dict:
    """Poses by name: `.pos` (world) and `.rotation()`; the bracket's also has `.joint(link)`."""
    from yam_arena.scene import object_states

    return object_states(env, tuple(names))


def nut_poses(env) -> list:
    """Each nut's (rotation, position above the table)."""
    out = []
    for name in NAMES:
        rot, pos = m.link_pose(env, "bracket", f"nut_{name}")
        out.append((rot, pos - UP))
    return out


def see_nut(rot, pos, k: int, lag: float) -> r.Nut:
    """One nut against the marked bar's slot, from its link's pose."""
    c = rot @ NUT_CENTRES[k] + pos
    axis = rot[:, 0]
    turn = math.degrees(math.atan2(abs(float(axis[1])), abs(float(axis[0]))))
    return r.Nut(along=float(c[0]), across=float(c[1] - g.MARKED_Y), top=float(g.BAR - c[2]), turn=turn, lag=lag)


def gaps(origin, rot, fingers, driver_pos, driver_rot) -> tuple:
    """The least distances from the fingers and from the driver to the bracket."""
    to_local = lambda pts: (np.atleast_2d(pts) - origin) @ rot  # noqa: E731
    finger_gap = g.bracket_distance(to_local(fingers)) if np.all(np.isfinite(fingers)) else float("nan")
    return finger_gap, g.bracket_distance(to_local(DRIVER_POINTS @ driver_rot.T + driver_pos))


def write_pose(env, name: str, pos, rot, body: str | None = None) -> None:
    """Put a free part at a pose (above the table), at rest: the technique of the accepted tasks bayonet-mate (partkit `write_pose`) and
    casement-stay-last-hole (mechkit `_write_pose`). For a jointed part the pose is that of `body`; its joints stay as they are."""
    import torch

    base = env.unwrapped
    asset = base.scene[name]
    origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0]
    pos, rot = np.asarray(pos, float) + UP, np.asarray(rot, float)
    names = list(getattr(asset, "body_names", ()))
    if body is not None and names and names[0] != body:  # the root link is another body: carry it with `body`
        poses = getattr(asset.data.body_link_pose_w, "torch", asset.data.body_link_pose_w)[0].detach().cpu().numpy().astype(float)
        root, held = poses[0], poses[names.index(body)]
        turn = rot @ m.quat_matrix(held[3:7]).T
        pos, rot = pos + turn @ (root[:3] - held[:3]), turn @ m.quat_matrix(root[3:7])
    w, x, y, z = m._quat_wxyz(rot)
    pose = torch.tensor([[*(float(v) for v in pos), x, y, z, w]], dtype=origin.dtype, device=origin.device)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=origin.dtype, device=origin.device))


def start_joints() -> dict:
    if START == "swung":
        return {"screw_a": math.radians(SWUNG[0]), "nut_a": math.radians(SWUNG[1])}
    return {"nut_a": math.radians(ASKEW_DEG)} if START == "askew" else {}


def put_back(env) -> None:
    """A reset puts the arms back, not a free mechanism: its body goes back to its start, its joints to theirs."""
    import torch

    origin, yaw = bracket_start()
    asset = env.unwrapped.scene["bracket"]
    q = asset.data.default_joint_pos.torch.clone() * 0.0
    for link, value in start_joints().items():
        q[0, asset.find_joints(f"{link}_joint")[0][0]] = float(value)
    asset.write_joint_position_to_sim_index(position=q)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    write_pose(env, "bracket", origin + [0.0, 0.0, 0.0003], g.rot_z(yaw), body="frame")


class Trial:
    """Runs the thread rule on both nuts every control step and keeps the score."""

    def __init__(self):
        self.rule = r.Rule()
        self.threads = [r.Thread(), r.Thread()]
        self.reset()

    def reset(self, env=None) -> None:
        self.rule.reset()
        for t in self.threads:
            t.reset()
        self.written = [r.T_RUN, r.T_RUN]
        self.n, self.full, self.score = 0, False, 0.0
        self.lags = [0.0, 0.0]
        if env is not None:
            put_back(env)
            for name in NAMES:
                m.set_friction(env, "bracket", f"nut_{name}", r.T_RUN)

    def obs(self, env) -> r.Obs:
        s = states(env, ("bracket", "driver"))
        b, d = s["bracket"], s["driver"]
        origin, rot = np.asarray(b.pos, float) - UP, b.rotation()
        nuts = tuple(see_nut(nr, npos, k, self.lags[k]) for k, (nr, npos) in enumerate(nut_poses(env)))
        finger_gap, driver_gap = gaps(origin, rot, m.fingertips(env) - UP, np.asarray(d.pos, float) - UP, d.rotation())
        return r.Obs(nuts=nuts, origin=origin, rot=rot, finger_gap=finger_gap, driver_gap=driver_gap)

    def step(self, env) -> None:
        b = states(env, ("bracket",))["bracket"]
        for k, name in enumerate(NAMES):
            target, lag, friction = self.threads[k].step(b.joint(f"nut_{name}"), b.joint(f"travel_{name}"))
            self.lags[k] = lag
            m.set_target(env, "bracket", f"travel_{name}", target)
            if abs(friction - self.written[k]) >= r.FRICTION_STEP:
                m.set_friction(env, "bracket", f"nut_{name}", friction)
                self.written[k] = friction
        self.score = self.rule.step(self.obs(env))
        self.full = self.full or self.score >= 1.0
        self.n += 1


TRIAL = Trial()
m.on_step(TRIAL.step)
m.on_reset(TRIAL.reset)


def success(objects) -> bool:
    return TRIAL.full


def score(objects) -> float:
    return 1.0 if TRIAL.full else TRIAL.score


def report(env, label: str = "") -> None:
    o = TRIAL.obs(env)
    b = states(env, ("bracket", "driver"))
    br, dr = b["bracket"], b["driver"]
    yaw = math.degrees(math.atan2(o.rot[1, 0], o.rot[0, 0]))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(o.rot[2, 2])))))
    corners = (g.FOOT_CORNERS @ o.rot.T + o.origin)[:, 2] - g.BAR
    nuts = "; ".join(f"nut {n}: turned {nut.turn:5.1f}, top {nut.top * 1000:6.2f} mm under the face, {nut.across * 1000:+5.2f} across, lag {nut.lag * 1000:5.2f} mm, "
                     f"screw {br.deg(f'screw_{n}'):7.1f}, {'grips' if nut.grips else 'under the lips' if nut.under_lips else 'in the slot' if nut.in_slot else 'out'}"
                     for n, nut in zip(NAMES, o.nuts))
    lean = math.degrees(math.acos(max(-1.0, min(1.0, float(dr.rotation()[2, 2])))))
    print(f"[tslot] t {TRIAL.n * r.DT:5.1f} {label}foot at {np.round(o.origin * 100, 2)} cm yaw {yaw:6.1f} tilt {tilt:4.1f}, corners {np.round(corners * 1000, 1)} mm up; {nuts}; "
          f"driver tip {np.round((np.asarray(dr.pos, float) - UP) * 100, 1)} cm leaning {lean:4.1f}; gaps fingers {o.finger_gap * 1000:5.1f} driver {o.driver_gap * 1000:5.1f} mm; "
          f"now {TRIAL.rule.now}; score {score(None):.2f}", flush=True)


# -- the scripted demo --------------------------------------------------------------------------------------------------------------------------
HOLD_TILT = 30.0  # the holding hand leans this far back along the arm, toward its camera end: its wrist then stands clear of the hand with the driver
PAD_MID = 0.0045  # the middle of a pad's flat face is this far down the fingers from the TCP
CARRY_UP = 0.055  # the bracket is carried this high above its start: its camera and nuts pass over the frame's bars
CARRY_OVER = 0.05  # and, commanded, this far above where it will hover over the bar: the arm gives up to 2 cm under the bracket's weight
OVER = 0.013  # it comes down from this far above the bar: the nuts' undersides 3.4 mm above its face
ROLL = 9.0  # degrees the camera end is tipped up: the nuts' outer lower edges then lead
EDGE_AT = -0.0028  # those edges are brought over the opening, this far across its middle line: 1.3 mm inside the outer lip
BESIDE = EDGE_AT + (g.NUT_B - g.NUT_PLAY) / 2  # so the foot is carried to this far from the slot's line
HOOK = 0.0005  # the edges go this far down into the opening before anything is drawn sideways
DEEP = 0.0014  # and, once the inner lip is clear, this far, still tipped: any deeper and a tipped nut has no room sideways
IN_OPENING = 0.0010  # an edge deeper than this has its nut between the lips
KEEP_IN, KEEP_DOWN = 0.0004, 0.0010  # the foot is only brought level while the middle of each nut's underside is this far down, and is let down to keep it there
PUSH_LEAD, PUSH_STEP = 0.004, 0.0002  # the hand's command goes outward at 4 mm a second, never further ahead of the nuts than this
LEAD = 0.004  # the furthest a lowering command runs ahead of what the hand holds: the arm follows late and in jumps of 2 to 3 mm
HOVER = 0.002  # the blade's tip is lined up with a socket this far above the screw's head
BLADE_LEAD = 0.008  # and how far the command runs ahead of the blade on its way down: reaching over the bar the arm follows later still
SLIDE_LEAD = 0.006  # and ahead of a foot that is slid along the slot
TRIES, LAST_TRY_S = 6, 85.0  # tries at the slot, none begun later than this: the screws and the driver take 75 s
STROKE_FROM, STROKE_TO = 138.0, -55.0  # the driving hand's yaw window over the screws
TURN_RATE = 45.0  # degrees a second
MODES = ("full", "none", "held", "unheld", "one_screw", "off_line", "settle", "take", "drive", "seat", "back")


def demo_script(ctx):
    """The left hand takes the bracket by its arm, carries it a quarter turn round to the marked bar, lowers the foot until both nuts are down
    in the slot, slides the notch to the line and holds. The right hand takes the driver from its block, sets the blade in each screw's socket
    and turns it in half-turn strokes of the wrist until the thread is wound half a turn past where the nut met the lips, lifting the blade out
    and turning back between strokes. It then stands the driver back in its block, and the left hand lets go last.
    Other modes, for the README's table: none does nothing; held never lets go of the arm; unheld lets go as soon as the foot is on the bar;
    one_screw tightens screw a only; off_line mounts the bracket 4 mm from the line. Mechanism tests of 200 steps or fewer: settle (any start),
    take (grip and lift the bracket), drive (with TSLOT_START=mounted: the driver into screw a and one stroke; with TSLOT_START=swung that
    stroke draws the nut tight), seat (with TSLOT_START=onlips: the hand takes a bracket whose nuts stand on a lip and gets them into the
    slot), back (the driver out of its block and back into it)."""
    import demokit as d
    from yam_arena import ik
    from yam_arena.constants import FINGER_OPEN
    from yam_arena.urdf import axis_angle_matrix

    env = ctx.emb._env
    left, right, mode = ctx.left, ctx.right, ctx.mode
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    for _ in range(10):
        yield
    report(env, f"{mode} start: ")
    flags = {"seated": False, "done": False, "failed": False}
    threads = [r.Thread(), r.Thread()]  # the demo's own count of how far each thread is wound, from the joints it can see
    lags = [0.0, 0.0]
    hold_grip = d.tilted(HOLD_TILT, g.MOUNT_YAW - 90.0)
    pace = {"take": 0.6, "drive": 0.4, "seat": 0.6, "back": 0.4}.get(mode, 1.0)  # the mechanism tests move faster, to fit 200 steps
    T = lambda seconds: seconds * pace  # noqa: E731
    line_x = g.LINE_X + (0.004 if mode == "off_line" else 0.0)

    def bracket() -> tuple:
        s = ctx.state("bracket")
        return np.asarray(s.pos, float), s.rotation()

    def point(local) -> np.ndarray:
        pos, rot = bracket()
        return pos + rot @ np.asarray(local, float)

    def yaw(rot) -> float:
        return math.degrees(math.atan2(rot[1, 0], rot[0, 0]))

    def height() -> float:
        """How far the foot's underside stands above the marked bar's face, at its higher end."""
        pos, rot = bracket()
        return float((g.FOOT_CORNERS @ rot.T + pos)[:, 2].max() - TOP - g.BAR)

    def watch():
        """Follows both threads from the bracket's joints, every step, for as long as the demo runs."""
        while True:
            s = ctx.state("bracket")
            for k, name in enumerate(NAMES):
                lags[k] = threads[k].step(s.joint(f"nut_{name}"), s.joint(f"travel_{name}"))[1]
            yield

    def nudge(arm, step) -> None:
        """Shift an arm's command by a small step, keeping its grip."""
        arm._solve(ik.tcp_pose(arm.q, arm.base)[0] + np.asarray(step, float), arm.grip.rot)

    def ease_down(arm, read, low: float, steps: int = 240, step: float = 0.00025, keep: bool = False):
        """Lower an arm until `read()`, a height, is at or under `low`. The arm follows a command late and in jumps, so the command never
        runs more than LEAD ahead of what it holds, and it is only called held up when nothing has come down for a second and a half at
        that lead. Ends with the lead taken off again, unless it is to `keep` pressing. Returns whether it got down."""
        lead, quiet, done = 0.0, 0, False
        for _ in range(steps):
            h = read()
            if h <= low:
                done = True
                break
            if lead < LEAD:
                nudge(arm, (0.0, 0.0, -step))
                lead += step
            yield
            drop = h - read()
            lead = max(lead - max(drop, 0.0), 0.0)
            quiet = quiet + 1 if drop < 0.00003 and lead >= LEAD - 1e-9 else 0
            if quiet >= 30:
                break
        if not keep:
            nudge(arm, (0.0, 0.0, max(lead - 0.001, 0.0)))  # all but a millimetre of the lead off again
        return done

    # -- the left hand: the bracket ------------------------------------------------------------------------------------------------------
    def take_bracket(lift: float = CARRY_UP):
        take_grip = d.tilted(HOLD_TILT, yaw(bracket()[1]) - 90.0)  # leaning back along the arm, whichever way the bracket lies
        a = left.approach_axis(take_grip)
        spot = point(g.GRIP_POINT) - a * PAD_MID  # read once: a servo's goal must stand still
        back = spot - 0.07 * a
        yield from left.move(back, grip=take_grip, finger=-(g.ARM_W / 2 + 0.012), t=T(2.5), label="over the bracket's arm")
        yield from left.servo(back, tol=0.003, timeout=T(1.5), label="over the bracket's arm")
        yield from left.move(spot, t=T(1.5), label="down round the arm")
        yield from left.servo(spot, tol=0.0015, timeout=T(1.5), label="down round the arm")
        yield from left.close(g.ARM_W, 20.0, label="take the arm")
        yield from left.wait(0.2)
        grip_at = lambda: point(g.GRIP_POINT)  # noqa: E731
        up = point(g.GRIP_POINT) + [0.0, 0.0, lift]
        yield from left.servo(lambda t: up - [0.0, 0.0, max(lift - 0.05 / pace * t, 0.0)], tol=0.003, timeout=T(3.0), read=grip_at, label="lift the bracket")
        report(env, "bracket lifted: ")

    def carry():
        """Round a quarter turn to over the marked bar: the foot's middle over the line, the nuts 3.4 mm above the bar's face."""
        a = left.approach_axis(hold_grip)
        over = np.array([line_x, g.MARKED_Y, TOP + g.BAR + OVER])
        hand = over + g.rot_z(g.MOUNT_YAW) @ g.GRIP_POINT - a * PAD_MID
        yield from left.move(hand + [0.0, 0.0, CARRY_OVER], grip=hold_grip, t=T(4.5), label="carry the bracket round to the bar")
        yield from left.wait(0.3)
        yield from over_slot()

    def over_slot():
        """Square the foot to the bar and bring its middle to a millimetre inside the slot's line, the nuts' undersides 3.4 mm above the bar's face."""
        over = np.array([line_x, g.MARKED_Y + BESIDE, TOP + g.BAR + OVER])
        left.trim = np.zeros(3)
        foot = lambda: point((0.0, 0.0, 0.0))  # noqa: E731
        yield from left.servo(over, tol=0.0015, timeout=T(2.5), gain=0.3, read=foot, label="the foot beside the slot")
        off = yaw(bracket()[1]) - g.MOUNT_YAW
        if abs(off) > 0.5:
            yield from left.arc(foot(), (0.0, 0.0, 1.0), -off, t=max(0.4, abs(off) / 10.0), label="square the foot to the bar")
            yield from left.servo(over, tol=0.0015, timeout=T(1.5), gain=0.3, read=foot, label="the foot beside the slot")
        report(env, "foot beside the slot: ")

    def low_edge() -> np.ndarray:
        """The middle of the outer lower edges of the two nuts' undersides, in the world."""
        pos, rot = bracket()
        return (g.NUT_LOW[g.NUT_LOW[:, 1] < 0] @ rot.T + pos).mean(axis=0)

    def edges() -> np.ndarray:
        """The outer lower edge of each nut's underside, a row a nut: (across the slot from its middle line, above the bar's face), in
        metres. Tipped camera end up, those edges are the nuts' lowest."""
        pos, rot = bracket()
        p = (g.NUT_LOW[g.NUT_LOW[:, 1] < 0] @ rot.T + pos).reshape(2, 2, 3).mean(axis=1)
        return np.c_[p[:, 1] - g.MARKED_Y, p[:, 2] - TOP - g.BAR]

    def sunk() -> float:
        """How far the middle of a nut's underside is below the bar's face: the less deep of the two."""
        pos, rot = bracket()
        return float(TOP + g.BAR - (g.NUT_LOW @ rot.T + pos)[:, 2].reshape(2, 4).mean(axis=1).max())

    def lowest() -> float:
        """The lowest corner of the nuts' undersides, above the bar's face."""
        pos, rot = bracket()
        return float((g.NUT_LOW @ rot.T + pos)[:, 2].min() - TOP - g.BAR)

    def tipped() -> float:
        rot = bracket()[1]
        return math.degrees(math.atan2(rot[2, 1], rot[1, 1]))

    def skew() -> float:
        return (yaw(bracket()[1]) - g.MOUNT_YAW + 180.0) % 360.0 - 180.0

    def slope() -> float:
        """How far the foot slopes along the bar, in degrees: positive with screw b's end up."""
        return math.degrees(math.asin(float(np.clip(bracket()[1][2, 0], -1.0, 1.0))))

    def gap(arm) -> np.ndarray:
        """An arm's command minus where its hand is, seen from its joints: its lag, its droop under a load and its push."""
        return ik.tcp_pose(arm.q, arm.base)[0] - arm.tcp()

    def turn_hand(axis, deg: float, about=None, arm=None) -> None:
        """One step: turn a hand (the left if none is named), and what it holds, about a line along `axis` through `about` (its own
        TCP if none)."""
        arm = left if arm is None else arm
        pos, rot = ik.tcp_pose(arm.q, arm.base)
        centre = pos if about is None else np.asarray(about, float)
        turn = axis_angle_matrix(np.asarray(axis, float), math.radians(deg))
        arm._solve(centre + turn @ (pos - centre), turn @ rot)
        arm.grip = d._as_grip(turn @ rot)

    def hold_still(arm) -> None:
        """An arm's command put where the arm is: a command that was wound on ahead leaves it nothing to creep to."""
        arm.q = np.asarray(ctx.observed(arm.side)[:6], float).copy()
        arm.cmd = np.r_[arm.q, arm.cmd[6]]
        arm.grip = d._as_grip(ik.tcp_pose(arm.q, arm.base)[1])

    def tilt(deg: float) -> None:
        """About a line along the bar through the middle of the two nuts' undersides: the camera end up or down, the nuts where they are."""
        turn_hand((1.0, 0.0, 0.0), deg, point((0.0, 0.0, -g.SCREW_END)))

    def trim(axis, sense: float, read, wound: float, at: float, most: float) -> float:
        """One step of turning the hand about a line through its own TCP, to bring `read()`, an angle of the foot, to zero. The wrist
        sticks, so the command is wound on a little a step until the foot moves, but never more than `most` degrees ahead of it.
        `sense` is the way the angle goes with the turn, `wound` the command so far and `at` the angle when it began. Returns the
        command so far."""
        off = read()
        step = -sense * float(np.clip(0.3 * off, -0.12, 0.12))
        if abs(wound + step - sense * (off - at)) <= most:
            turn_hand(axis, step)
            wound += step
        return wound

    def square(tol: float = 0.12, steps: int = 40, most: float = 1.5):
        """The foot along the bar within `tol` degrees and level along it within twice that: two nuts 60 mm apart leave the foot half a
        degree in the slot's opening, and a foot that slopes holds one nut higher than the other."""
        good, wound, at = 0, [0.0, 0.0], (skew(), slope())
        for _ in range(steps):
            good = good + 1 if abs(skew()) <= tol and abs(slope()) <= 2 * tol else 0
            if good >= 3:
                return True
            if abs(skew()) > 0.5 * tol:
                wound[0] = trim((0.0, 0.0, 1.0), 1.0, skew, wound[0], at[0], most)
            if abs(slope()) > tol:
                wound[1] = trim((0.0, 1.0, 0.0), -1.0, slope, wound[1], at[1], most)
            yield
        return False

    def trace(label: str) -> None:
        e, ahead, s = edges() * 1000, gap(left) * 1000, ctx.state("bracket")
        turned = [s.deg(f"screw_{name}") + s.deg(f"nut_{name}") + skew() for name in NAMES]
        print(f"[tslot] t {ctx.t:5.1f} {label}: tipped {tipped():+.1f}, skew {skew():+.2f}, slope {slope():+.2f}; nut edges {e[0, 0]:+.2f} {e[1, 0]:+.2f} mm across, {e[0, 1]:+.2f} {e[1, 1]:+.2f} mm up, "
              f"nuts {turned[0]:+.1f} {turned[1]:+.1f} degrees off the slot's line; foot {height() * 1000:.1f} mm up; command ahead of the hand ({ahead[0]:+.1f} {ahead[1]:+.1f} {ahead[2]:+.1f}) mm", flush=True)

    def draw_out(clear: float):
        """The hand's command outward, never more than PUSH_LEAD ahead of the nuts, until both nuts' outer edges are past `clear` or have
        stopped, which is against the outer lip. Then the push is taken off again."""
        y0, sent, quiet = float(edges()[:, 0].mean()), 0.0, 0
        for _ in range(100):
            at = edges()[:, 0]
            if at.max() <= clear:
                break
            ahead = sent - (y0 - float(at.mean()))
            if ahead < PUSH_LEAD:
                nudge(left, (0.0, -PUSH_STEP, 0.0))
                sent += PUSH_STEP
            yield
            quiet = quiet + 1 if abs(float(edges()[:, 0].mean() - at.mean())) < 0.00002 and ahead >= PUSH_LEAD - PUSH_STEP else 0
            if quiet >= 25:
                break
        nudge(left, (0.0, max(sent - (y0 - float(edges()[:, 0].mean())), 0.0), 0.0))
        for _ in range(4):
            yield

    def into_slot(n: int):
        """One try at the slot, by feel: the arm cannot place two nuts to the 0.19 mm either side that the opening leaves. The hand tips
        the camera end up 9 degrees, squares the foot to the bar and brings the nuts' outer lower edges over the opening, which is 8 mm
        wide for an edge; lets them down into it; draws them outward until the inner lip is clear of the nuts or the outer lip stops
        them; lets them a little further down, still tipped; then brings the foot level, steering by the bracket's own tilt and only
        while the nuts stay down in the opening, and lets it down onto the bar. Returns whether the foot is down flat, which it cannot
        be with a nut on a lip."""
        if abs(ROLL - tipped()) > 2.0:
            yield from left.arc(point((0.0, 0.0, -g.SCREW_END)), (1.0, 0.0, 0.0), ROLL - tipped(), t=0.8, label="tip the camera end up")
        for _ in range(12):
            off = ROLL - tipped()
            if abs(off) < 0.7:
                break
            tilt(float(np.clip(0.3 * off, -0.4, 0.4)))
            yield
        yield from square()
        over = np.array([line_x, g.MARKED_Y + EDGE_AT, TOP + g.BAR + 0.0015])
        yield from left.servo(over, tol=0.0006, timeout=1.5, gain=0.3, read=low_edge, label="the nuts' edges over the opening")
        yield from square(steps=15)
        trace(f"try {n + 1}, edges over the opening")
        yield from ease_down(left, lambda: float(edges()[:, 1].mean()), -HOOK)
        if edges()[:, 1].min() > -0.0003:
            trace(f"try {n + 1}, the nuts stay on a lip")
            return False
        clear = g.SLOT_OPEN / 2 - (g.NUT_B - g.NUT_PLAY) * math.cos(math.radians(tipped())) - 0.00015  # an outer edge this far across: that nut is off the inner lip
        yield from draw_out(clear)
        trace(f"try {n + 1}, drawn outward (clear of the inner lip from {clear * 1000:+.2f})")
        for again in range(3):  # down into the opening, tipped as they are
            down = yield from ease_down(left, lambda: float(edges()[:, 1].max()), -DEEP, keep=True)
            if down or again == 2:
                break
            trace(f"try {n + 1}, held up")
            yield from square(steps=15, most=0.6)
            if edges()[:, 0].max() > clear:
                yield from draw_out(clear)
        if edges()[:, 1].max() > -IN_OPENING:
            trace(f"try {n + 1}, the nuts are NOT in the opening")
            return False
        trace(f"try {n + 1}, in the opening")
        start, stuck, roll0, turned, wound, at = gap(left), 0, tipped(), 0.0, [0.0, 0.0], (skew(), slope())
        for k in range(400):  # level, then down. No command runs far ahead of what it moves: a foot that is held up is not forced
            roll, h = tipped(), height()
            if h <= 0.0008 and abs(roll) < 1.0:
                break
            if lowest() > -0.0002:  # a nut has come up out of the opening
                break
            ahead = start - gap(left)
            if abs(roll) > 0.3 and sunk() >= KEEP_IN and ahead[2] < 0.005 and abs(turned - (roll - roll0)) < 2.5:
                step = -float(np.clip(0.4 * roll, -0.25, 0.25))
                tilt(step)
                turned += step
            if abs(skew()) > 0.15:
                wound[0] = trim((0.0, 0.0, 1.0), 1.0, skew, wound[0], at[0], 0.6)
            if abs(slope()) > 0.25:
                wound[1] = trim((0.0, 1.0, 0.0), -1.0, slope, wound[1], at[1], 0.6)
            if (abs(roll) < 2.0 or sunk() < KEEP_DOWN) and ahead[2] < LEAD:  # down once it is nearly level; before that, only to keep the nuts in
                nudge(left, (0.0, 0.0, -0.0003))
            if abs(ahead[1]) > 0.002:  # sideways the hand gives way to the lips
                nudge(left, (0.0, 0.0003 * float(np.sign(ahead[1])), 0.0))
            yield
            stuck = stuck + 1 if h - height() < 0.00003 and abs(tipped() - roll) < 0.02 else 0
            if stuck > 50:  # two and a half seconds with nothing moving
                break
            if k % 10 == 9:
                trace(f"try {n + 1}, levelling")
        nudge(left, (0.0, 0.0, max(float((start - gap(left))[2]) - 0.001, 0.0)))
        for _ in range(4):
            yield
        seated = height() <= 0.0008
        trace(f"try {n + 1}, {'the foot is down' if seated else 'the foot is NOT down'}")
        return seated

    def lower():
        """Both nuts into the slot: tries until the foot is down or there is no time left to fasten it."""
        for n in range(TRIES):
            seated = yield from into_slot(n)
            if seated:
                report(env, "nuts in: ")
                return True
            if n + 1 == TRIES or ctx.t > LAST_TRY_S:
                break
            for _ in range(60):  # up again, until both nuts hang clear of the bar
                if lowest() >= 0.003:
                    break
                nudge(left, (0.0, 0.0, 0.0005))
                yield
            yield from left.wait(0.3)
        report(env, "foot NOT flat: ")
        return False

    def to_line():
        """With the nuts in the slot the foot slides along it: bring the notch onto the line, if it is more than 1.5 mm off."""
        notch = lambda: float(point(g.NOTCH_POINT)[0])  # noqa: E731
        x0, sent = notch(), 0.0
        if abs(line_x - x0) <= 0.0015:
            return
        for _ in range(80):
            off = line_x - notch()
            if abs(off) <= 0.0010:
                break
            step = float(np.clip(0.3 * off, -0.0003, 0.0003))
            if abs(sent + step - (notch() - x0)) <= SLIDE_LEAD:
                nudge(left, (step, 0.0, 0.0))
                sent += step
            yield
        yield from left.wait(0.3)

    def left_hand():
        yield from take_bracket()
        if mode == "take":
            return
        yield from carry()
        seated = yield from lower()
        if seated:
            yield from to_line()
            report(env, "foot on the bar: ")
        if not seated:  # nothing to fasten: both hands stop here
            flags["failed"] = True
            return
        flags["seated"] = True
        if mode == "unheld":
            yield from let_go()
            return
        while not flags["done"]:
            yield
        if mode != "held":
            yield from let_go()

    def let_go():
        a = left.approach_axis()
        yield from left.move(left.at(), finger=-(g.ARM_W / 2 + 0.014), t=0.5, label="let go of the arm")
        left.trim = np.zeros(3)
        yield from left.move(left.at() - 0.08 * a, t=T(1.5), label="back off the arm")
        yield from left.move(np.array([0.16, 0.40, TOP + 0.20]), grip=d.down(0.0), finger=FINGER_OPEN, t=T(2.5), label="clear")

    # -- the right hand: the driver --------------------------------------------------------------------------------------------------------
    def tip() -> np.ndarray:
        return np.asarray(ctx.state("driver").pos, float)

    def blade_yaw() -> float:
        return yaw(ctx.state("driver").rotation())

    def socket(k: int) -> tuple:
        """(the middle of screw k's socket floor in the world, the socket's yaw)."""
        s = ctx.state("bracket")
        pos, rot = np.asarray(s.pos, float), s.rotation()
        return pos + rot @ (g.SCREWS[k] + [0.0, 0.0, g.SOCKET_FLOOR]), yaw(rot) + s.deg(f"screw_{NAMES[k]}")

    def wrap60(deg: float) -> float:
        return (deg + 30.0) % 60.0 - 30.0

    def hand_yaw() -> float:
        return yaw(ik.tcp_pose(right.q, right.base)[1] @ d.down(0.0).rot.T)

    def take_driver():
        over = tip() + [0.0, 0.0, g.TIP_TO_TCP]  # read once
        across = (blade_yaw() + 90.0) % 180.0 - 90.0  # the fingers close across the neck's two flat sides, whichever way the driver stands
        yield from right.move(over + [0.0, 0.0, 0.05], grip=d.down(across), finger=-(g.NECK[1] / 2 + 0.012), t=T(2.5), label="over the driver")
        yield from right.servo(over + [0.0, 0.0, 0.05], tol=0.003, timeout=1.5, label="over the driver")
        yield from right.move(over, t=T(1.2), label="down round the driver's neck")
        yield from right.servo(over, tol=0.002, timeout=1.0, label="down round the driver's neck")
        yield from right.close(g.NECK[1], 20.0, label="take the driver")
        yield from right.wait(0.2)
        yield from right.move(right.at() + [0.0, 0.0, g.HOLE_DEPTH + 0.015], t=T(1.5), label="lift the driver out of its block")
        report(env, "driver taken: ")

    def over_socket(k: int, steps: int = 70):
        """The blade's hex turned to the socket's and its tip brought over the socket's axis, both by what is seen and a little a step,
        no command wound more than 2 degrees or 4 mm ahead of what it moves: the screw is stiff in its hole and will not come round to
        the blade, and the socket leaves a full hex 0.06 mm either side. Ends with the arm's command put where the arm is."""
        at, wound = wrap60(blade_yaw() - socket(k)[1]), 0.0
        xy0, sent, good = tip()[:2].copy(), np.zeros(2), 0
        for _ in range(steps):
            off = wrap60(blade_yaw() - socket(k)[1])
            miss = (socket(k)[0] - tip())[:2]
            good = good + 1 if abs(off) <= 0.7 and np.linalg.norm(miss) <= 0.0004 else 0
            if good >= 2:
                break
            if abs(off) > 0.4:
                step = -float(np.clip(0.3 * off, -0.3, 0.3))
                if abs(wound + step - (off - at)) <= 2.0:
                    turn_hand((0.0, 0.0, 1.0), step, arm=right)
                    wound += step
            if np.linalg.norm(miss) > 0.0002:
                step = np.clip(0.3 * miss, -0.0004, 0.0004)
                if np.linalg.norm(sent + step - (tip()[:2] - xy0)) <= 0.004:
                    nudge(right, (float(step[0]), float(step[1]), 0.0))
                    sent += step
            yield
        hold_still(right)
        yield from right.wait(0.15)

    def set_blade(k: int, hand: float):
        """The blade over screw k's socket with the hand turned to `hand`, its hex lined up with the socket's, then down into it. Returns
        whether it went in. The hand is turned there by the angle itself, never by the shortest way round: its wrist has one window."""
        head, _ = socket(k)
        above = head + [0.0, 0.0, g.SOCKET_DEPTH + HOVER + g.TIP_TO_TCP]
        if np.linalg.norm(right.at() - above) > 0.004:
            yield from right.move(above, t=T(1.5) if np.linalg.norm(right.at() - above) > 0.03 else 0.4, label=f"the blade over screw {NAMES[k]}")
        back = hand - hand_yaw()
        if abs(back) > 0.5:
            yield from right.turn(back, t=max(0.3, abs(back) / (2.0 * TURN_RATE)), label="turn the hand back in the air")
        yield from right.servo(socket(k)[0] + [0.0, 0.0, g.SOCKET_DEPTH + HOVER], tol=0.0006, timeout=2.0, gain=0.25, read=tip, label=f"the blade over screw {NAMES[k]}")
        hold_still(right)
        yield from over_socket(k)
        depth = lambda: float(tip()[2] - socket(k)[0][2])  # noqa: E731
        xy0, sent, lead, quiet = tip()[:2].copy(), np.zeros(2), 0.0, 0
        for _ in range(200):  # down, the tip kept toward the socket's axis: on the head's top it is drawn across to the socket, in its mouth the chamfer is helped
            h = depth()
            if h <= 0.0008:
                break
            miss = (socket(k)[0] - tip())[:2]
            if np.linalg.norm(miss) > 0.0002:
                step = np.clip(0.3 * miss, -0.0003, 0.0003)
                if np.linalg.norm(sent + step - (tip()[:2] - xy0)) <= 0.003:
                    nudge(right, (float(step[0]), float(step[1]), 0.0))
                    sent += step
            if lead < BLADE_LEAD:
                nudge(right, (0.0, 0.0, -0.00025))
                lead += 0.00025
            yield
            drop = h - depth()
            lead = max(lead - max(drop, 0.0), 0.0)
            quiet = quiet + 1 if drop < 0.00003 and lead >= BLADE_LEAD - 1e-9 else 0
            if quiet >= 30:
                break
        hold_still(right)
        right.trim = np.zeros(3)
        yield from right.wait(0.15)
        head = socket(k)
        print(f"[tslot] t {ctx.t:5.1f} blade at screw {NAMES[k]}: tip {1000 * depth():.2f} mm above the socket's floor, {1000 * float(np.linalg.norm((tip() - head[0])[:2])):.2f} mm off its axis, "
              f"hex {wrap60(blade_yaw() - head[1]):+.1f} degrees from the socket's; the command had run {lead * 1000:.1f} mm ahead", flush=True)
        return depth() <= 0.0015

    def stroke(k: int, strokes: int = 12):
        """Screw k in clockwise strokes of the wrist until its thread is wound tight, the blade lifted out and turned back between strokes."""
        for n in range(strokes):
            _, phase = socket(k)
            top = 100.0 if mode == "drive" else STROKE_FROM  # the one-stroke test starts lower, to fit its 200 steps
            start = top - ((top - (phase + hand_yaw() - blade_yaw())) % 60.0)  # the largest yaw in the window at which the hex meets the socket
            seated = yield from set_blade(k, start)
            if not seated:
                print(f"[tslot] the blade is not in screw {NAMES[k]}'s socket: lift and try again", flush=True)
                yield from right.move(right.at() + [0.0, 0.0, 0.004], t=0.3, label="lift the blade")
                continue
            sweep = start - STROKE_TO  # as far as the wrist goes: the next stroke starts where the hex meets the socket again
            before = ctx.deg("bracket", f"screw_{NAMES[k]}")
            for _ in right.turn(-sweep, t=sweep / TURN_RATE / (1.0 if pace == 1.0 else 2.0), label=f"turn screw {NAMES[k]}"):
                yield
                if lags[k] >= r.TIGHT_TURNS * g.PITCH:
                    break
            right.grip = d._as_grip(ik.tcp_pose(right.q, right.base)[1])
            yield from right.wait(0.2)
            report(env, f"screw {NAMES[k]} stroke {n + 1}, turned {before - ctx.deg('bracket', f'screw_{NAMES[k]}'):.0f}: ")
            yield from right.move(right.at() + [0.0, 0.0, 0.010], t=0.4, label="lift the blade out")
            if lags[k] >= r.TIGHT_TURNS * g.PITCH or mode == "drive":
                return True
        return False

    def driver_back():
        hole = np.array([BLOCK_XY[0], BLOCK_XY[1], TOP + g.BLOCK[2]])
        yield from right.move(hole + [0.0, 0.0, 0.02 + g.TIP_TO_TCP], grip=d.down(0.0), t=T(3.0), label="the driver over its block")
        for _ in range(3):  # a blade that stands on the block's top is lifted and brought over again
            yield from right.servo(hole + [0.0, 0.0, 0.02], tol=0.0008, timeout=2.5, gain=0.3, read=tip, label="the blade over the block's hole")
            down = yield from ease_down(right, lambda: float(tip()[2] - (TOP + g.DRIVER_START[2])), 0.002, step=0.001)
            if down:
                break
            yield from right.move(right.at() + [0.0, 0.0, 0.015], t=0.5, label="lift the blade off the block")
        yield from right.move(right.at(), finger=-(g.NECK[1] / 2 + 0.012), t=0.5, label="let go of the driver")
        right.trim = np.zeros(3)
        yield from right.move(right.at() + [0.0, 0.0, 0.07], t=T(1.2), label="up off the driver")
        yield from right.move(np.array([0.16, -0.40, TOP + 0.20]), grip=d.down(0.0), finger=FINGER_OPEN, t=T(2.0), label="clear")

    def right_hand():
        yield from take_driver()
        while not flags["seated"]:
            if flags["failed"]:
                return
            yield
        if mode == "unheld":
            yield from right.wait(2.0)
        for k in ((0,) if mode in ("one_screw", "drive") else (0, 1)):
            yield from stroke(k)
        if mode != "drive":
            yield from driver_back()
        flags["done"] = True  # the other hand lets go last

    def settle(steps: int):
        for k in range(steps):
            yield
            if k % 20 == 19:
                report(env, "settle: ")

    def both():
        yield from ctx.together(left_hand(), right_hand())

    def run(main):
        """`main` with the threads followed every step."""
        follower = watch()
        for _ in main:
            next(follower)
            yield

    if mode == "none":
        yield from settle(40)
        return
    if mode == "settle":
        yield from settle(180)
        return
    if mode == "take":
        yield from run(left_hand())
        return
    if mode == "back":  # the driver out of its block and back into it
        def back():
            yield from take_driver()
            yield from right.wait(0.3)
            yield from driver_back()
            report(env, "driver back: ")
        yield from run(back())
        return
    if mode == "seat":  # with TSLOT_START=onlips: from standing on its nuts on a lip, over the slot and down into it
        def seat():
            yield from take_bracket(lift=0.006)
            left.trim = np.zeros(3)
            seated = yield from lower()
            if seated:
                yield from to_line()
                report(env, "foot on the bar: ")
        yield from run(seat())
        return
    if mode == "drive":  # with TSLOT_START=mounted: nobody holds the bracket, which sits on the bar with its nuts in the slot
        flags["seated"] = True
        yield from run(right_hand())
        report(env, "after one stroke: ")
        return
    yield from run(both())
    yield from settle(60)
