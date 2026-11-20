"""A kinematic stand-in for the simulator, to shake out the demo and the scorer's wiring without simulator time.

Not physics. The arms follow their commands with a lag. The bracket follows a hand whose fingers have closed on its arm; over the marked
bar it is pushed out of what it would cut into: a nut's underside goes below the bar's face only inside the slot's opening, whose lips
then stop it sideways, and otherwise stands on a lip. The driver follows the hand that holds its neck; its blade goes into a socket only when it is over it and its hex is lined up,
and a blade in a socket turns the screw with the hand. The nuts follow the thread rule as contact would have them: carried round with the
screw until the boss meets the lips at a quarter turn, then up the screw a pitch a turn until the flange meets the lips, then wound. An
unheld bracket that no nut holds tips over its camera end. It runs the task's real demo_script, Trial and rule, so it finds wrong calls,
unreachable poses, loops that never end, end states that do not score, and how many control steps a run takes.

    PYTHONPATH=<yam-isaac-lab-arena> python3 dryrun.py [modes] [--clear] [--quiet]
"""
import importlib.util
import math
import os
import pathlib
import sys
import types

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "1")  # many small products: threads only get in each other's way

import numpy as np  # noqa: E402

TASK_DIR = str(pathlib.Path(__file__).resolve().parent)
QUIET = "--quiet" in sys.argv
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import demokit as d  # noqa: E402
import mechkit as m  # noqa: E402
import tslot_geometry as g  # noqa: E402
import tslot_rule as r  # noqa: E402
from yam_arena import ik, station  # noqa: E402
from yam_arena.constants import FINGER_OPEN, READY  # noqa: E402

spec = importlib.util.spec_from_file_location("tslot_task", f"{TASK_DIR}/task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
TOP = task.TOP
UP = np.array([0.0, 0.0, TOP])
DT = 1.0 / d.CONTROL_HZ
LAG, FINGER_LAG = 0.35, 0.5
CAMERA = (0.050, 0.050, 0.040)  # a box round a wrist camera and its mount, in the gripper's frame (a D405 is 42 x 42 x 23 mm)
STOP_CW, STOP_CCW = g.boss_stops()
TIP_OVER = 6.0  # degrees an unheld, unfastened bracket tips over the bar's inner edge before its nuts wedge in the opening


class State:
    """Stands in for a part's state."""

    def __init__(self, pos, rot, joints=None):
        self.pos, self.rot, self.joints = np.asarray(pos, float), np.asarray(rot, float), joints or {}

    def rotation(self):
        return self.rot

    def joint(self, link):
        return self.joints[f"{link}_joint"]

    def deg(self, link):
        return math.degrees(self.joint(link))


def yaw_of(rot) -> float:
    return math.degrees(math.atan2(rot[1, 0], rot[0, 0]))


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class World:
    def __init__(self):
        self.q = {s: np.array(READY, float) for s in ("left", "right")}
        self.f = {s: FINGER_OPEN for s in ("left", "right")}
        origin, yaw = task.bracket_start()
        self.bracket = [origin + UP, g.rot_z(yaw)]
        self.screw, self.nut, self.travel = [0.0, 0.0], [math.radians(v) for v in (task.ASKEW_DEG if task.START == "askew" else 0.0, 0.0)], [0.0, 0.0]
        if task.START == "swung":
            self.screw[0], self.nut[0], self.travel[0] = math.radians(task.SWUNG[0]), math.radians(task.SWUNG[1]), g.PITCH * task.SWUNG[1] / 360.0
        self.driver = [np.array([task.BLOCK_XY[0], task.BLOCK_XY[1], g.DRIVER_START[2]]) + UP, g.rot_z(task.DRIVER_YAW)]
        self.held = {}  # name -> (side, position in the hand's frame, rotation in the hand's frame)
        self.in_socket, self.last_blade = None, None
        self.tipped, self.notes = False, []
        self._screw_was = list(self.screw)

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)
            if not QUIET:
                print(f"   [world] {text}", flush=True)

    # -- what the task reads ----------------------------------------------------------------------------------------------
    def tcp(self, side):
        return ik.tcp_pose(self.q[side], np.asarray(station.BASES[side], float))

    def bracket_state(self):
        joints = {}
        for k, name in enumerate(task.NAMES):
            joints |= {f"screw_{name}_joint": self.screw[k], f"travel_{name}_joint": self.travel[k], f"nut_{name}_joint": self.nut[k]}
        return State(self.bracket[0], self.bracket[1], joints)

    def state(self, name):
        return self.bracket_state() if name == "bracket" else State(*self.driver)

    def link_pose(self, k):
        """A nut's link: the bracket's frame, turned about its screw's axis by the screw's and the nut's angles and run up it by the travel."""
        pos, rot = self.bracket
        c = np.array([g.SCREWS[k][0], 0.0, 0.0])
        turn = g.rot_z(math.degrees(self.screw[k] + self.nut[k]))
        return rot @ turn, pos + rot @ (c - turn @ c + [0.0, 0.0, self.travel[k]])

    def nut_turn(self, k) -> float:
        """A nut's turn from lengthwise, clockwise seen from above, in degrees."""
        return -math.degrees(self.screw[k] + self.nut[k])

    def fingertips(self):
        out = []
        for side in ("left", "right"):
            pos, rot = self.tcp(side)
            origin = pos - rot @ np.array([0.0, 0.0, 0.134])
            opening = abs(self.f[side])
            for sx in (-1.0, 1.0):
                for dy in (-0.018, 0.0, 0.018):
                    for dz in (0.147, 0.117):
                        out.append(origin + rot @ np.array([sx * (opening + 0.0045), dy, dz]))
        return np.array(out)

    # -- one control step -------------------------------------------------------------------------------------------------
    def step(self, ctx):
        for side, arm in (("left", ctx.left), ("right", ctx.right)):
            self.q[side] = self.q[side] + LAG * (arm.cmd[:6] - self.q[side])
            self.f[side] = self.f[side] + FINGER_LAG * (arm.cmd[6] * FINGER_OPEN - self.f[side])
        self.grasps()
        self.carry_bracket()
        self.carry_driver()
        self.threads()
        self.unheld()

    def grasps(self):
        for name, half, spot in (("bracket", g.ARM_W / 2, g.GRIP_POINT), ("driver", g.NECK[1] / 2, np.array([0.0, 0.0, g.TIP_TO_TCP]))):
            pos, rot = self.bracket if name == "bracket" else self.driver
            if name in self.held:
                if abs(self.f[self.held[name][0]]) > half + 0.003:
                    side = self.held.pop(name)[0]
                    self.note(f"the {side} hand lets go of the {name}")
                    self.let_go(name)
                continue
            for side in ("left", "right"):
                hand_pos, hand = self.tcp(side)
                pads = hand_pos + hand[:, 2] * task.PAD_MID
                local = rot.T @ (pads - pos)
                if name == "bracket":
                    near = g.box_distance(local, (0.0, g.FOOT[1] / 2 + (g.ARM_END - g.FOOT[1] / 2) / 2, g.ARM_H / 2), (0.004, g.ARM_END - g.FOOT[1] / 2 - 0.03, g.ARM_H))[0] < 0.004
                    across = abs(float((rot.T @ hand[:, 0]) @ np.array([1.0, 0.0, 0.0])))  # the fingers close across the arm
                else:
                    near = float(np.linalg.norm(rot.T @ (hand_pos - pos) - spot)) < 0.008
                    across = abs(float((rot.T @ hand[:, 0]) @ np.array([0.0, 1.0, 0.0])))  # and across the neck's flat sides
                if abs(self.f[side]) <= half + 0.0005 and near and across > 0.95:
                    self.held[name] = (side, hand.T @ (pos - hand_pos), hand.T @ rot)
                    self.note(f"the {side} hand holds the {name}: {1000 * float(np.linalg.norm(rot.T @ (pads - pos) - spot)):.1f} mm from where it means to")
                    if name == "bracket":
                        self.tipped = False

    def over_marked_bar(self, pos) -> bool:
        return abs(pos[1] - g.MARKED_Y) < g.BAR / 2 + 0.01 and g.SIDE_SPAN[0] - 0.03 < pos[0] < g.SIDE_SPAN[1] + 0.03

    def rest_on_bar(self, pos, rot) -> np.ndarray:
        """The bracket pushed out of the marked bar: a nut's underside may be below the bar's face only inside the slot's opening, whose lips
        then stop it sideways, and the foot may not be below the face at all."""
        pos = np.array(pos, float)
        face, lo, hi = TOP + g.BAR, g.MARKED_Y - g.SLOT_OPEN / 2, g.MARKED_Y + g.SLOT_OPEN / 2
        for _ in range(4):
            pts = g.NUT_LOW @ rot.T + pos
            deep = pts[:, 2] < face - 0.0004
            if deep.any():  # down in the opening
                pos[1] += max(lo - pts[deep, 1].min(), 0.0) - max(pts[deep, 1].max() - hi, 0.0)
                pts = g.NUT_LOW @ rot.T + pos
            out = (pts[:, 1] < lo - 1e-9) | (pts[:, 1] > hi + 1e-9)
            if out.any() and pts[out, 2].min() < face:  # over a lip: it stands on it
                pos[2] += face - pts[out, 2].min()
                self.note("the nuts land on the lips")
            foot = g.FOOT_CORNERS @ rot.T + pos
            if foot[:, 2].min() < face:
                pos[2] += face - foot[:, 2].min()
        if (g.NUT_LOW @ rot.T + pos)[:, 2].max() < face - 0.002:
            self.note("both nuts pass the slot's opening")
        return pos

    def carry_bracket(self):
        if "bracket" not in self.held:
            return
        side, p, rr = self.held["bracket"]
        hand_pos, hand = self.tcp(side)
        pos, rot = hand_pos + hand @ p, hand @ rr
        if self.over_marked_bar(pos):
            pos = self.rest_on_bar(pos, rot)
        self.bracket = [pos, rot]

    def support(self, xy) -> tuple:
        """(height above the table of what a blade's tip may come down to here, whether that is a socket's floor)."""
        pos, rot = self.bracket
        for k in range(2):
            head = pos + rot @ (g.SCREWS[k] + [0.0, 0.0, g.SOCKET_FLOOR])
            off = float(np.linalg.norm(np.asarray(xy) - head[:2]))
            if off <= g.HEAD_D / 2:
                lined = off <= g.TIP_CAPTURE  # the tip's chamfer takes up the rest
                already = self.in_socket == k
                return (head[2] if lined or already else head[2] + g.SOCKET_DEPTH), k if lined or already else None
        b = np.asarray(xy) - np.asarray(task.BLOCK_XY)
        if abs(b[0]) <= g.BLOCK[0] / 2 and abs(b[1]) <= g.BLOCK[1] / 2:
            return (TOP + g.DRIVER_START[2] if max(abs(b)) <= g.HOLE / 2 - 0.0005 else TOP + g.BLOCK[2]), None
        return TOP, None

    def carry_driver(self):
        if "driver" not in self.held:
            return
        side, p, rr = self.held["driver"]
        hand_pos, hand = self.tcp(side)
        pos, rot = hand_pos + hand @ p, hand @ rr
        floor, k = self.support(pos[:2])
        before = yaw_of(self.driver[1])
        if pos[2] < floor:
            pos = np.array([pos[0], pos[1], floor])
        self.driver = [pos, rot]
        engaged = k is not None and pos[2] <= floor + g.SOCKET_DEPTH - 0.001
        if engaged and self.in_socket == k:  # the blade is in the socket: the screw turns with it
            self.screw[k] += math.radians((yaw_of(rot) - before + 180.0) % 360.0 - 180.0)
        if engaged and self.in_socket != k:  # going in, the chamfered tip brings the free screw round to the blade's hex
            self.screw[k] += math.radians((yaw_of(rot) - (yaw_of(self.bracket[1]) + math.degrees(self.screw[k])) + 30.0) % 60.0 - 30.0)
            self.note(f"the blade is in screw {task.NAMES[k]}'s socket")
        self.in_socket = k if engaged else None

    def threads(self):
        """Each nut as contact would have it: carried round with its screw until something stops it, then up the screw."""
        pos, rot = self.bracket
        seated = self.over_marked_bar(pos) and pos[2] - TOP <= g.BAR + 0.0008
        for k in range(2):
            turned = self._screw_was[k] - self.screw[k]  # clockwise positive
            if not turned:
                continue
            cw = self.nut_turn(k) - math.degrees(turned)  # where the nut was before this step's turn
            bearing = seated and self.travel[k] >= g.NUT_DROP - 1e-9 and cw > 20.0
            if not seated:
                free = math.degrees(turned)  # in the air a nut goes round with its screw
            elif bearing:
                free = 0.0  # on the lips, the ridges hold it
            else:
                free = float(np.clip(cw + math.degrees(turned), -STOP_CCW, STOP_CW)) - cw  # the boss in the opening lets it turn a quarter turn, one way
            self.nut[k] += turned - math.radians(free)  # what the screw turned and the nut did not, the nut has turned on the screw
            target = float(np.clip(g.PITCH * self.nut[k] / (2.0 * math.pi), *g.TRAVEL))
            under = seated and self.nut_turn(k) > 20.0
            if under and target > g.NUT_DROP - max(float(pos[2]) - TOP - g.BAR, 0.0):  # the nut meets the lips: from here the screw draws the foot down flat
                self.settle_flat()
                pos = self.bracket[0]
            self.travel[k] = min(target, g.NUT_DROP if under else g.TRAVEL[1])  # under the lips it rises to them; lengthwise, on up to the foot
        self._screw_was = list(self.screw)

    def settle_flat(self):
        """The bracket down flat on the marked bar, where it is; a hand that holds it keeps holding it there."""
        pos, rot = self.bracket
        pos = np.array([pos[0], pos[1], TOP + g.BAR])
        rot = g.rot_z(yaw_of(rot))
        self.bracket = [pos, rot]
        if "bracket" in self.held:
            side = self.held["bracket"][0]
            hand_pos, hand = self.tcp(side)
            self.held["bracket"] = (side, hand.T @ (pos - hand_pos), hand.T @ rot)

    def nut_grips(self, k) -> bool:
        return self.travel[k] >= g.NUT_DROP - 1e-9 and self.nut_turn(k) > 20.0 and g.PITCH * self.nut[k] / (2.0 * math.pi) - self.travel[k] >= 0.0001

    def unheld(self):
        pos, rot = self.bracket
        if "bracket" in self.held or self.tipped or task.START in ("mounted", "swung", "onlips") or not self.over_marked_bar(pos) or pos[2] - TOP > g.BAR + 0.001:
            return
        if not any(self.nut_grips(k) for k in range(2)):  # nothing holds the foot down: the camera end goes down, the foot's outer edge up
            edge = np.array([pos[0], g.MARKED_Y + g.BAR / 2, TOP + g.BAR])
            tip = rot_x(-TIP_OVER) if abs(yaw_of(rot)) < 90.0 else rot_x(TIP_OVER)
            self.bracket = [edge + tip @ (pos - edge), tip @ rot]
            self.tipped = True
            self.note(f"nobody holds the bracket and no nut grips: it tips {TIP_OVER:.0f} degrees over the bar's inner edge")

    def let_go(self, name):
        if name == "driver":
            pos, rot = self.driver
            floor, k = self.support(pos[:2])
            b = np.asarray(pos[:2]) - np.asarray(task.BLOCK_XY)
            if max(abs(b)) <= g.HOLE / 2 and pos[2] - floor < 0.01:
                self.driver = [np.array([pos[0], pos[1], floor]), rot]
                self.note("the driver stands in its block")
            elif k is not None:
                self.note(f"the driver is left standing in screw {task.NAMES[k]}")
            else:
                self.driver = [np.array([pos[0], pos[1], TOP + 0.011]), rot @ rot_x(90.0)]
                self.note("the driver is let go in the air and falls on the table")


class Ctx(d.Ctx):
    def observed(self, side):
        return np.r_[WORLD.q[side], WORLD.f[side] / FINGER_OPEN]


# -- clearances (--clear): what the kinematic world cannot feel ----------------------------------------------------------------------
def _labelled(method):
    def run(self, *a, **k):
        if "label" in k:
            self.label = k["label"]
        return (yield from method(self, *a, **k))
    return run


for _name in ("move", "servo", "turn", "arc", "open", "close"):
    setattr(d.Arm, _name, _labelled(getattr(d.Arm, _name)))

_SHELLS: dict = {}


def _shell(size, step=0.012) -> np.ndarray:
    """Points over a box's surface, in its frame."""
    key = tuple(round(float(v), 6) for v in size)
    if key not in _SHELLS:
        axes = [np.linspace(-v / 2, v / 2, max(2, int(math.ceil(v / step)) + 1)) for v in key]
        pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        _SHELLS[key] = pts[np.any(np.abs(np.abs(pts) - np.asarray(key) / 2) < 1e-9, axis=1)]
    return _SHELLS[key]


class Clearance:
    """The least distance between each arm's collision hulls and each thing in the scene, by the primitive the arm was in, and between the
    two arms. Fingertips on what they hold read about 0. Each bar is taken as a plain 40 mm box."""

    def __init__(self):
        from yam_arena import analysis
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

        self.an, self.joints, self.least, self.arms = analysis, (ARM_JOINTS, FINGER_JOINTS), {}, {}
        eye = np.eye(3)
        self.fixed = [("table", np.array([0.3, 0.0, -0.05]) + UP, eye, (2.0, 2.0, 0.1))]
        for bar, axis, centre, span in g.frame_bars():
            mid, length = (span[0] + span[1]) / 2, span[1] - span[0]
            c, size = ((mid, centre[1], g.BAR / 2), (length, g.BAR, g.BAR)) if axis == "x" else ((centre[0], mid, g.BAR / 2), (g.BAR, length, g.BAR))
            self.fixed.append((f"{bar} bar", np.array(c) + UP, eye, size))
        self.fixed.append(("block", np.array([task.BLOCK_XY[0], task.BLOCK_XY[1], g.BLOCK[2] / 2]) + UP, eye, g.BLOCK))
        if not task.START:
            self.fixed += [(f"support{k}", np.array(c) + UP, eye, size) for k, (c, size) in enumerate(g.strip_boxes())]

    def hulls(self, world, side):
        """The arm's collision hulls, and a box where its wrist camera is (the simulator's arm has no collider there; a real one has a camera)."""
        from yam_arena.constants import WRIST_CAM_POS

        q = {n: float(v) for n, v in zip(self.joints[0], world.q[side])} | {n: float(world.f[side]) for n in self.joints[1]}
        poses = self.an.fk_link_poses_world(q, station.BASES[side])
        rot, pos = poses["gripper"][:3, :3], poses["gripper"][:3, 3]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(CAMERA) / 2 + WRIST_CAM_POS
        normals = np.vstack([np.eye(3), -np.eye(3)])
        planes = np.c_[normals @ rot.T, np.r_[np.array(CAMERA) / 2 + WRIST_CAM_POS, np.array(CAMERA) / 2 - WRIST_CAM_POS] + (normals @ rot.T) @ pos]
        return self.an.world_hulls(poses) + [("wrist camera", self.an.Hull(vertices=corners @ rot.T + pos, planes=planes))]

    def things(self, world) -> list:
        """(name, centre, rotation, size) in the world."""
        pos, rot = world.bracket
        out = list(self.fixed)
        out += [(f"bracket's {name}", pos + rot @ np.array(c), rot, size) for name, c, size, _ in g.bracket_boxes()]
        pos, rot = world.driver
        out += [(f"driver's {name}", pos + rot @ np.array(c), rot, size) for name, c, size, _ in g.driver_boxes()]
        out.append(("driver's blade", pos + rot @ np.array([0.0, 0.0, g.BLADE / 2]), rot, (0.005, 0.005, g.BLADE)))
        return out

    def sample(self, world, ctx):
        an = self.an
        sets = {side: self.hulls(world, side) for side in ("left", "right")}
        labels = {side: getattr(ctx.arm(side), "label", "start") for side in sets}
        for name, centre, rot, size in self.things(world):
            size = np.asarray(size, float)
            reach = float(np.linalg.norm(size)) / 2 + 0.04
            shell = None
            for side, hulls in sets.items():
                best, link = 0.04, None
                for part, h in hulls:
                    if name == "table" and not part.startswith(("tip", "gripper")):
                        continue
                    if np.min(np.linalg.norm(h.vertices - centre, axis=1)) > reach:
                        continue
                    dist = float(an.box_signed_distance(h.vertices, np.asarray(centre), np.asarray(rot), size).min())
                    if dist < 0.04 and max(size) < 0.5:  # near: the thing's own surface against the hull's faces too
                        if shell is None:
                            shell = _shell(size) @ np.asarray(rot).T + centre
                        dist = min(dist, float(an._signed(shell, h).min()))
                    if dist < best:
                        best, link = dist, part
                key = (side, labels[side], name)
                if link is not None and best < self.least.get(key, (0.04, None))[0]:
                    self.least[key] = (best, link)
        lo, hi = ({side: np.array([f(h.vertices, axis=0) for _, h in hulls]) for side, hulls in sets.items()} for f in (np.min, np.max))
        apart = np.maximum(lo["left"][:, None, :] - hi["right"][None, :, :], lo["right"][None, :, :] - hi["left"][:, None, :]).max(axis=2)
        key = (labels["left"], labels["right"])
        for i, j in zip(*np.nonzero(apart < 0.04)):  # hulls whose bounding boxes are near each other
            (pl, hl), (pr, hr) = sets["left"][i], sets["right"][j]
            dist = min(float(an._signed(hl.vertices, hr).min()), float(an._signed(hr.vertices, hl).min()))
            if dist < self.arms.get(key, (0.04, None))[0]:
                self.arms[key] = (dist, (pl, pr))

    def worst(self, held=("bracket's arm", "driver's neck", "driver's bar")) -> tuple:
        """(the least distance of a hull to anything it is not holding, the least between the arms), in metres."""
        things = [v[0] for (side, label, name), v in self.least.items() if not ((side == "left" and name == held[0]) or (side == "right" and name in held[1:]))]
        return (min(things) if things else 0.04), (min(v[0] for v in self.arms.values()) if self.arms else 0.04)

    def table(self, under=0.012):
        for side in ("left", "right"):
            print(f"-- {side} arm: hulls within {under * 1000:.0f} mm of something, by primitive")
            for (s, label, name), (dist, link) in self.least.items():
                if s == side and dist < under:
                    print(f"   {label:44s} {name:18s} {dist * 1000:6.1f} mm  {link}")
        print(f"-- arm to arm, within {under * 1000:.0f} mm")
        for (ll, rl), (dist, pair) in self.arms.items():
            if dist < under:
                print(f"   left '{ll}' / right '{rl}': {dist * 1000:.1f} mm  {pair}")


CLEAR = Clearance() if "--clear" in sys.argv else None
CLEAR_EVERY = 2  # control steps between samples


def run(mode, max_steps=4000, hold_s=2.0):
    global WORLD, CTX
    WORLD = World()
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(step_dt=DT))
    task.states = lambda e, names: {n: WORLD.state(n) for n in names}
    task.put_back = lambda e: None
    m.link_pose = lambda e, part, link: WORLD.link_pose(task.NAMES.index(link[-1]))
    m.fingertips = lambda e: WORLD.fingertips()
    m.set_target = lambda e, part, link, value: None
    m.set_friction = lambda e, part, link, value: None
    task.TRIAL.reset(env)
    CTX = ctx = Ctx(types.SimpleNamespace(_env=env), None, mode)
    ctx.state = lambda name: WORLD.state(name)
    ctx.pos = lambda name: WORLD.state(name).pos
    ctx.deg = lambda part, link: WORLD.state(part).deg(link)
    script = task.demo_script(ctx)
    ended, steps, best = None, 0, 0.0
    while steps < max_steps:
        if ended is None:
            try:
                next(script)
            except StopIteration:
                ended = steps
        WORLD.step(ctx)
        if CLEAR and steps % CLEAR_EVERY == 0:
            CLEAR.sample(WORLD, ctx)
        task.TRIAL.step(env)
        steps += 1
        ctx.t += DT
        best = max(best, task.score(None))
        if task.success(None):
            break
        if ended is not None and steps - ended >= hold_s * d.CONTROL_HZ:
            break
    result = {"score": task.score(None), "best": best, "success": task.success(None), "steps": steps, "ended": ended,
              "ik_mm": max(ctx.left.worst[0], ctx.right.worst[0]), "ik_deg": max(ctx.left.worst[1], ctx.right.worst[1]),
              "notes": list(WORLD.notes), "now": dict(task.TRIAL.rule.now), "turns": [round(-WORLD.screw[k] / (2 * math.pi), 2) for k in range(2)]}
    print(f"RESULT {mode}: score {task.score(None)} (best on the way {best}), success {task.success(None)}, steps {steps} ({steps * DT:.1f} s), script ended at {ended}, "
          f"IK worst left {ctx.left.worst[0]:.1f} mm {ctx.left.worst[1]:.1f} deg, right {ctx.right.worst[0]:.1f} mm {ctx.right.worst[1]:.1f} deg; "
          f"screws turned {result['turns']} turns; now {task.TRIAL.rule.now}")
    return result


if __name__ == "__main__":
    for mode in (sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["full"]):
        print(f"\n===== {mode}")
        run(mode)
        if CLEAR:
            CLEAR.table()
