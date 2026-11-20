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
# TSLOT_START=onlips has it start on the bar 8 mm off the slot, standing on its nuts on a lip, its arm propped level.
START = os.environ.get("TSLOT_START", "")
if START not in ("", "mounted", "askew", "onlips"):
    raise ValueError(f"TSLOT_START={START!r}")
BLOCK_XY = (0.20, -0.17) if START == "mounted" else g.BLOCK_XY
DRIVER_YAW = 75.0 if START == "mounted" else 0.0  # in the mounted test the driver stands turned, so that the hand takes it near the top of its stroke
ASKEW_DEG, ASKEW_UP = 20.0, 0.003
ONLIPS = (0.003, 0.008)  # where the foot starts in the on-the-lips test, from where it belongs
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
    return m.fixed("frame", (0.0, 0.0, TOP), parts, cols, LOOKS, friction=(0.5, 0.4))


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
    if START == "mounted":
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
    return m.mechanism("bracket", tuple(origin + UP + [0.0, 0.0, 0.0003]), links, LOOKS, frame_parts=parts, frame_colliders=cols, yaw_deg=yaw, friction=(0.7, 0.6),
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
OBJECTS = [FRAME, BLOCK, BRACKET, DRIVER] + ([PROP] if START in ("mounted", "onlips") else [] if START else [SUPPORTS])


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
BESIDE = 0.0025  # the foot is aimed this far to the inner side of the slot's line
ROLL = 8.0  # degrees the camera end is tipped up: the nuts' outer lower edges then lead
EDGE_AT = -0.001  # those edges are brought over the opening, this far across its middle line
HOOK = 0.0018  # and go this far down into it
PUSH, PUSH_STEP = 0.009, 0.0002  # then the hand's command goes this far outward, at 4 mm a second: what the foot does not follow is a light push on the outer lip
STUCK = 0.006  # a part that has not followed the command this far is held by something. Under 3 mm it may only be the arm's own friction
STROKE_FROM, STROKE_TO = 138.0, -55.0  # the driving hand's yaw window over the screws
TURN_RATE = 45.0  # degrees a second
MODES = ("full", "none", "held", "unheld", "one_screw", "off_line", "settle", "take", "drive", "seat")


def demo_script(ctx):
    """The left hand takes the bracket by its arm, carries it a quarter turn round to the marked bar, lowers the foot until both nuts are down
    in the slot, slides the notch to the line and holds. The right hand takes the driver from its block, sets the blade in each screw's socket
    and turns it in half-turn strokes of the wrist until the thread is wound half a turn past where the nut met the lips, lifting the blade out
    and turning back between strokes. It then stands the driver back in its block, and the left hand lets go last.
    Other modes, for the README's table: none does nothing; held never lets go of the arm; unheld lets go as soon as the foot is on the bar;
    one_screw tightens screw a only; off_line mounts the bracket 4 mm from the line. Mechanism tests of 200 steps or fewer: settle (any start),
    take (grip and lift the bracket), drive (with TSLOT_START=mounted: the driver into screw a and one stroke), seat (with TSLOT_START=onlips:
    the hand takes a bracket whose nuts stand on a lip, brings them over the slot and lowers them in)."""
    import demokit as d
    from yam_arena import ik
    from yam_arena.constants import FINGER_OPEN

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
    pace = {"take": 0.6, "drive": 0.45, "seat": 0.6}.get(mode, 1.0)  # the mechanism tests move faster, to fit 200 steps
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
        """Square the foot to the bar and bring its middle to 2.5 mm inside the slot's line, the nuts' undersides 3.4 mm above the bar's face."""
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
        """The middle of the outer lower edges of the two nuts' undersides: (along the bar, across the slot from its middle line, above
        the bar's face), in metres. Tipped camera end up, those edges are the nuts' lowest."""
        pos, rot = bracket()
        p = (g.NUT_LOW[g.NUT_LOW[:, 1] < 0] @ rot.T + pos).mean(axis=0)
        return np.array([p[0], p[1] - g.MARKED_Y, p[2] - TOP - g.BAR])

    def tipped() -> float:
        rot = bracket()[1]
        return math.degrees(math.atan2(rot[2, 1], rot[1, 1]))

    def lower():
        """Both nuts into the slot, by feel. The arm cannot place them to the 0.34 mm either side that the opening leaves, and nuts drawn
        across the lips turn on their screws. So the hand tips the camera end up 8 degrees, brings the nuts' outer lower edges over the
        opening, which is 8 mm wide for an edge, lets them down into it, draws the foot outward until those edges lie against the outer
        lip, lays the foot flat about that lip, lowers it, and takes its sideways push off again. Returns whether the foot is down."""
        for attempt in range(3):
            pos, _ = bracket()
            yield from left.arc(np.array([pos[0], pos[1], TOP + g.BAR]), (1.0, 0.0, 0.0), ROLL, t=T(1.0), label="tip the camera end up")
            over = np.array([line_x, g.MARKED_Y + EDGE_AT, TOP + g.BAR + 0.0015])
            edge_point = lambda: low_edge() + [0.0, g.MARKED_Y, TOP + g.BAR]  # noqa: E731
            yield from left.servo(over, tol=0.0012, timeout=T(1.5), gain=0.3, read=edge_point, label="the nuts' edges over the opening")
            e = low_edge()
            print(f"[tslot] tipped {tipped():.1f} degrees; the nuts' outer edges {e[1] * 1000:+.2f} mm across the slot, {e[2] * 1000:+.2f} mm from the bar's face", flush=True)
            lead = 0.0
            for _ in range(60):  # the edges down into the opening
                z = low_edge()[2]
                if z <= -HOOK or lead > STUCK:
                    break
                nudge(left, (0.0, 0.0, -0.0005))
                yield
                lead += 0.0005 - max(z - low_edge()[2], 0.0)
            e = low_edge()
            if e[2] > -0.0006:  # held up on a lip: away from it by 2 mm, and again
                way = 1.0 if e[1] < 0.0 else -1.0
                print(f"[tslot] the nuts' edges stay {e[2] * 1000:+.2f} mm from the bar's face, {e[1] * 1000:+.2f} mm across: on a lip. Up and 2 mm {'in' if way > 0 else 'out'}", flush=True)
                yield from left.arc(np.array([pos[0], pos[1], TOP + g.BAR]), (1.0, 0.0, 0.0), -ROLL, t=T(0.8), label="level again")
                yield from left.move(left.at() + [0.0, way * 0.002, OVER / 2], t=T(0.8), label="up and across")
                continue
            y0 = float(bracket()[0][1])
            for _ in range(round(PUSH / PUSH_STEP)):  # outward, until the edges lie against the outer lip and the arm bears on it a little
                nudge(left, (0.0, -PUSH_STEP, 0.0))
                yield
            moved = y0 - float(bracket()[0][1])
            e = low_edge()
            print(f"[tslot] drawn {moved * 1000:.1f} mm outward: the nuts' outer edges {e[1] * 1000:+.2f} mm across (the outer lip is at {-g.SLOT_OPEN / 2 * 1000:.2f}), {e[2] * 1000:+.2f} mm from the face", flush=True)
            lip = np.array([bracket()[0][0], g.MARKED_Y - g.SLOT_OPEN / 2, TOP + g.BAR])
            yield from left.arc(lip, (1.0, 0.0, 0.0), -ROLL, t=T(1.2), label="lay the foot flat about the outer lip")
            heights, lead = [], 0.0
            for _ in range(80):  # and down onto the bar
                heights.append(height())
                if heights[-1] <= 0.0008 or lead > STUCK:
                    break
                stopped = len(heights) > 5 and heights[-6] - heights[-1] < 0.0002
                lead = lead + 0.0005 if stopped else 0.0
                nudge(left, (0.0, 0.0, -0.0005))
                yield
            for _ in range(4):
                yield
            ease = max(PUSH - moved - 0.001, 0.0)  # the sideways push, less a millimetre: a nut pressed on a lip cannot swing
            for _ in range(10):
                nudge(left, (0.0, ease / 10, 0.0))
                yield
            report(env, f"try {attempt + 1}: ")
            if height() <= 0.0008:
                return True
            print(f"[tslot] the foot stays {height() * 1000:.1f} mm up: lifting it out to try again", flush=True)
            yield from left.move(left.at() + [0.0, BESIDE, OVER], t=T(1.0), label="lift the foot out")
        return False

    def to_line():
        """With the nuts in the slot the foot slides along it: bring the notch onto the line."""
        notch = lambda: point(g.NOTCH_POINT)  # noqa: E731
        for _ in range(40):
            off = line_x - float(notch()[0])
            if abs(off) <= 0.0010:
                break
            nudge(left, (float(np.clip(0.4 * off, -0.001, 0.001)), 0.0, 0.0))
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
        report(env, f"foot {'on the bar' if seated else 'NOT down'}: ")
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

    def set_blade(k: int, hand: float):
        """The blade over screw k's socket with the hand turned to `hand`, its hex lined up with the socket's, then down into it. Returns
        whether it went in. The hand is turned there by the angle itself, never by the shortest way round: its wrist has one window."""
        head, _ = socket(k)
        above = head + [0.0, 0.0, 0.008 + g.TIP_TO_TCP]
        if np.linalg.norm(right.at() - above) > 0.004:
            yield from right.move(above, t=T(1.5) if np.linalg.norm(right.at() - above) > 0.03 else 0.4, label=f"the blade over screw {NAMES[k]}")
        back = hand - hand_yaw()
        if abs(back) > 0.5:
            yield from right.turn(back, t=max(0.3, abs(back) / (2.0 * TURN_RATE)), label="turn the hand back in the air")
        for _ in range(3):  # the tip's chamfer takes up 0.7 mm and brings a free screw round to the blade: near is near enough
            off = wrap60(blade_yaw() - socket(k)[1])
            if abs(off) > 8.0:
                yield from right.turn(-off, t=0.3, label="line the hex up with the socket")
            ok = yield from right.servo(socket(k)[0] + [0.0, 0.0, 0.008], tol=0.0006, timeout=T(2.0), gain=0.3, read=tip, label=f"the blade over screw {NAMES[k]}")
            if ok:
                break
        lead = 0.0
        for _ in range(60):
            depth = float(tip()[2] - socket(k)[0][2])
            if depth <= 0.0008 or lead > STUCK:
                break
            nudge(right, (0.0, 0.0, -0.0005))
            yield
            lead = lead + 0.0005 - max(depth - float(tip()[2] - socket(k)[0][2]), 0.0)
        right.trim = np.zeros(3)
        yield from right.wait(0.15)
        head = socket(k)
        print(f"[tslot] blade at screw {NAMES[k]}: tip {1000 * float(tip()[2] - head[0][2]):.2f} mm above the socket's floor, {1000 * float(np.linalg.norm((tip() - head[0])[:2])):.2f} mm off its axis, "
              f"hex {wrap60(blade_yaw() - head[1]):+.1f} degrees from the socket's", flush=True)
        return float(tip()[2] - head[0][2]) <= 0.0015

    def stroke(k: int, strokes: int = 8):
        """Screw k in clockwise strokes of the wrist until its thread is wound tight, the blade lifted out and turned back between strokes."""
        for n in range(strokes):
            _, phase = socket(k)
            top = 100.0 if mode == "drive" else STROKE_FROM  # the one-stroke test starts lower, to fit its 200 steps
            start = top - ((top - (phase + hand_yaw() - blade_yaw())) % 60.0)  # the largest yaw in the window at which the hex meets the socket
            seated = yield from set_blade(k, start)
            if not seated:
                print(f"[tslot] the blade is not in screw {NAMES[k]}'s socket: lift and try again", flush=True)
                yield from right.move(right.at() + [0.0, 0.0, 0.010], t=0.4, label="lift the blade")
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
        yield from right.servo(hole + [0.0, 0.0, 0.02], tol=0.0008, timeout=2.5, gain=0.3, read=tip, label="the blade over the block's hole")
        lead = 0.0
        for _ in range(120):
            depth = float(tip()[2] - (TOP + g.DRIVER_START[2]))
            if depth <= 0.002 or lead > 0.006:
                break
            nudge(right, (0.0, 0.0, -0.001))
            yield
            lead = lead + 0.001 - max(depth - float(tip()[2] - (TOP + g.DRIVER_START[2])), 0.0)
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
    if mode == "seat":  # with TSLOT_START=onlips: from standing on its nuts on a lip, over the slot and down into it
        def seat():
            yield from take_bracket(lift=0.006)
            yield from over_slot()
            seated = yield from lower()
            if seated:
                yield from to_line()
            report(env, f"foot {'on the bar' if seated else 'NOT down'}: ")
        yield from run(seat())
        return
    if mode == "drive":  # with TSLOT_START=mounted: nobody holds the bracket, which sits on the bar with its nuts in the slot
        flags["seated"] = True
        yield from run(right_hand())
        report(env, "after one stroke: ")
        return
    yield from run(both())
    yield from settle(60)
