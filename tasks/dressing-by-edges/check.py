"""Offline check of dressing_by_edges.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/dressing-by-edges/check.py

1. Design: the room the right gripper has at the dressing on the pack and over the wound (the arm's collision
   meshes placed by IK), its distance to the key part while grasping and releasing, which sites each arm reaches,
   and whether the top camera sees the pack, the dressing, the wound and the tape past the arms at their ready pose.
2. Scores: hand-built trials (the dressing and the arms step by step, on a stand-in for the episode clock) against
   the score each must get.
3. Demo: plays the right arm's plan through the IK at 20 Hz with the dressing following the gripper that holds it,
   scores it, and prints the smallest distances from the robot to the key part, the pack, the skin, the tape, the
   saline bottle and the glove box, the worst IK error and the joint limits.
4. Nudge variant: the same with the dressing set down 3 cm off and pushed into place.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState
from yam_arena.urdf import quat_xyzw_to_matrix

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("dressing_by_edges.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ, EVERY = 20, 4
MM = 1e3
SAFE = 0.005  # the room the demo must keep to the key part and to the props it does not touch
PARKED = dict.fromkeys(("left", "right"), np.r_[READY, FINGER_OPEN])
BASE = station.BASES["right"]
WX, WY = T.WOUND_CENTRE

# the simulator's episode clock and arm joints, which the task file reads from the embodiment
NOW = {"t": 0.0, "arms": PARKED}
T.clock = lambda: NOW["t"]
T.arm_joints = lambda: NOW["arms"]


# ---- states ----
def box(size, xy, under, yaw=0.0, moving=False, pitch=0.0) -> ObjectState:
    """A box with its centre over xy and its underside at height `under`, turned yaw about the vertical and
    pitched about y."""
    qz = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    qy = np.array([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)])
    rot = T._rotation(qz) @ T._rotation(qy)
    tr = np.trace(rot)
    s = math.sqrt(max(tr + 1.0, 1e-12)) * 2
    q = np.array([(rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s, s / 4])
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.array(size) / 2
    corners = corners @ rot.T
    pos = np.array([*xy, under - corners[:, 2].min()])
    vel = np.array([0.1 if moving else 0.0, 0.0, 0.0])
    return ObjectState(pos, q, vel, np.zeros(3), pos + corners.min(0), pos + corners.max(0))


def rest_height(xy) -> float:
    """What a dressing let go over xy comes to rest on."""
    half = np.array(T.DRESSING_SIZE[:2]) / 2
    over_wound = np.all(np.abs(np.asarray(xy) - T.WOUND_CENTRE) < half + np.array(T.WOUND_SIZE[:2]) / 2)
    if over_wound:
        return T.WOUND_TOP
    if T._inside(xy, T.SKIN_CENTRE, T.SKIN_SIZE[:2]):
        return T.SKIN_TOP
    if T._inside(xy, T.PACK_CENTRE, T.PACK_SIZE[:2]):
        return T.PACK_TOP
    if T._inside(xy, T.DRAPE_CENTRE, T.DRAPE_SIZE[:2]):
        return T.DRAPE_TOP
    return T.TOP


def dressing(xy=None, under=None, yaw=0.0, moving=False, pitch=0.0) -> ObjectState:
    xy = T.PACK_CENTRE if xy is None else xy
    return box(T.DRESSING_SIZE, xy, rest_height(xy) if under is None else under, yaw, moving, pitch)


def tape() -> ObjectState:
    size = (2 * T.TAPE_RADIUS, 2 * T.TAPE_RADIUS, T.TAPE_HEIGHT)
    return box(size, T.TAPE_CENTRE, T.TOP)


START = {"dressing": dressing(), "tape": tape()}


def at(dx=0.0, dy=0.0, **kw) -> dict:
    """The dressing lying on the skin, its centre dx, dy from the wound's."""
    return {"dressing": dressing((WX + dx, WY + dy), **kw)}


def up(dx=0.0, dy=0.0) -> dict:
    """The dressing carried 5 cm over the skin."""
    return {"dressing": dressing((WX + dx, WY + dy), under=T.SKIN_TOP + 0.05)}


def gripper_at(xyz, yaw, finger=FINGER_OPEN) -> dict:
    """The arms with the right gripper's TCP at a point, pointing down, and the left arm parked."""
    q, miss, _ = ik.solve(xyz, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), READY, BASE)
    assert miss < 1e-3, f"the right gripper cannot reach {xyz} at yaw {math.degrees(yaw):.0f}"
    return {**PARKED, "right": np.r_[q, finger]}


def reaches(side, points, yaws) -> bool:
    """Whether an arm has an IK solution at every point, under 0.5 mm with every joint 3 degrees inside its limits,
    for one of the grip yaws."""
    for yaw in yaws:
        rot = ik.matrix_to_quat_xyzw(grip_rotation(yaw))
        margins = []
        for p in points:
            q, miss, _ = ik.solve(p, rot, READY, station.BASES[side])
            margins.append(-1.0 if miss >= 5e-4 else np.degrees(np.minimum(q - ik.LOWER, ik.UPPER - q)).min())
        if min(margins) > 3.0:
            return True
    return False


def touching(objects) -> dict:
    """The right gripper's closed finger tips resting on the middle of the dressing's top face."""
    d = objects["dressing"]
    return gripper_at((d.pos[0], d.pos[1], d.high[2] + PAD_TIP_PAST_TCP), T.GRIP_YAW, FINGER_CLOSED)


def holding(objects) -> dict:
    """The right gripper closed on the dressing's edge faces where it lies."""
    d = objects["dressing"]
    return gripper_at((d.pos[0], d.pos[1], d.low[2] - T.TIPS_BELOW_UNDERSIDE + PAD_TIP_PAST_TCP), T.GRIP_YAW, T.HOLD)


# ---- distances from the right arm ----
def _groups() -> dict:
    kind = {"drape": "the drape", "skin": "the skin", "wound": "the skin", "pack": "the pack", "gloves": "the glove box"}
    out = {}
    for b in T.FIXED:
        if b.name == "saline":
            size = (2 * T.SALINE_RADIUS, 2 * T.SALINE_RADIUS, T.SALINE_HEIGHT)
            out.setdefault("the saline bottle", []).append((size, box(size, b.pos[:2], T.TOP)))
        else:
            out.setdefault(kind[b.name], []).append((b.size, box(b.size, b.pos[:2], b.pos[2] - b.size[2] / 2)))
    return out


GROUPS = _groups()
TAPE_SIZE = (2 * T.TAPE_RADIUS, 2 * T.TAPE_RADIUS, T.TAPE_HEIGHT)


def gaps(arms, objects) -> dict:
    """Smallest distance (m) from the right arm (every link, its points within 15 cm of the table) to each group of
    fixed props, to the tape roll, to the dressing and to its key part."""
    points = T.arm_points(arms["right"], "right")
    points = points[points[:, 2] < T.TOP + 0.15]
    out = {what: min(T.box_gap(points, b, size) for size, b in boxes) for what, boxes in GROUPS.items()}
    out["the tape roll"] = T.box_gap(points, objects["tape"], TAPE_SIZE)
    out["the dressing"] = T.box_gap(points, objects["dressing"], T.DRESSING_SIZE)
    out["the key part"] = T.box_gap(points, objects["dressing"], T.KEY_SIZE)
    return out


def shown(g, *keys) -> str:
    return ", ".join(f"{k} {g[k] * MM:.1f}" for k in keys)


# ---- trials ----
def step(objects, arms=PARKED, hook=T.score):
    NOW["arms"] = arms
    NOW["t"] += 1 / HZ
    return hook(dict(objects))


def play(moves, restart=True) -> float:
    """The score at the end of a trial that starts from the layout and applies the moves one step at a time. A
    move changes some objects' states; its "arms" entry places the arms for that step only."""
    if restart:
        NOW["t"] = 0.0
    world, out = dict(START), step(START)
    for move in moves:
        move = dict(move)
        arms = move.pop("arms", PARKED)
        world.update(move)
        out = step(world, arms)
    play.world = world
    return out


def check_design():
    props = ("the pack", "the tape roll", "the saline bottle", "the glove box", "the drape", "the skin")
    # the grasp: fingers open coming down on the dressing, then closed on its edge faces
    px, py = T.PACK_CENTRE
    open_ = gaps(gripper_at((px, py, T.GRASP_Z), T.GRIP_YAW), START)
    closed = gaps(holding(START), START)
    print(f"  grasp, fingers open (mm): {shown(open_, *props, 'the dressing', 'the key part')}")
    print(f"  grasp, fingers closed (mm): {shown(closed, *props, 'the key part')}")
    assert all(open_[k] >= SAFE for k in (*props, "the dressing")), "the open fingers have no room at the pack"
    assert open_["the key part"] >= SAFE and closed["the key part"] >= SAFE, "the grasp touches the key part"
    assert all(closed[k] >= SAFE for k in props), "the closed fingers have no room at the pack"
    # the release: the dressing 2.5 mm over the skin, fingers closed then open
    laid = {**START, **at(under=T.SKIN_TOP + T.DROP)}
    hold = gaps(holding(laid), laid)
    let_go = gaps(gripper_at((WX, WY, T.PLACE_Z), T.GRIP_YAW), laid)
    print(f"  release, fingers closed (mm): {shown(hold, 'the skin', 'the drape', 'the key part')}")
    print(f"  release, fingers open (mm): {shown(let_go, 'the skin', 'the drape', 'the key part', 'the dressing')}")
    assert hold["the skin"] >= 0.001 and let_go["the skin"] >= 0.001, "the finger tips reach the skin at the release"
    assert hold["the key part"] >= SAFE and let_go["the key part"] >= SAFE
    # coverage margins
    margin = (np.array(T.DRESSING_SIZE[:2]) - np.array(T.WOUND_SIZE[:2])) / 2
    print(f"  wound margin under a centred dressing: x {margin[0] * 100:.2f} cm, y {margin[1] * 100:.2f} cm")
    assert margin.min() > 0.015
    # reach: the right arm at every demo site, the left arm reported
    sites = {
        "the pack": [(px, py, T.GRASP_Z), (px, py, T.TRAVEL)],
        "the wound": [(WX, WY, T.PLACE_Z), (WX, WY, T.TRAVEL)],
        "the nudge": [(WX + T.NUDGE + 0.05, WY, T.PUSH_Z), (WX + T.NUDGE, WY, T.PLACE_Z)],
        "the clear point": [(WX, WY - 0.10, T.TRAVEL)],
    }
    for side in ("right", "left"):
        reached = [what for what, points in sites.items() if reaches(side, points, (T.GRIP_YAW, math.pi / 2))]
        print(f"  {side} arm reaches {', '.join(reached) or 'nothing'}")
        assert side == "left" or len(reached) == len(sites), "the right arm cannot reach every demo site"


def check_view():
    """The top camera (320 x 176) sees the pack, the dressing, the wound and the tape roll past both arms at
    their ready pose."""
    cam = station.top_camera()
    lens = cam.lens.resized(320, 176)
    rot, pos = quat_xyzw_to_matrix(np.array(cam.rot_xyzw)), np.array(cam.pos)

    def project(points):
        pc = (np.asarray(points) - pos) @ rot  # camera frame: looks along -z, y up
        depth = -pc[:, 2]
        return np.c_[lens.cx + lens.fx * pc[:, 0] / depth, lens.cy - lens.fy * pc[:, 1] / depth, depth]

    nearest = {}
    for side in ("left", "right"):
        for u, v, d in project(T.arm_points(np.r_[READY, FINGER_OPEN], side)):
            key = (round(u), round(v))
            nearest[key] = min(nearest.get(key, np.inf), d)
    hx, hy = np.array(T.DRESSING_SIZE[:2]) / 2
    points = {
        **{
            f"dressing corner {i}": (T.PACK_CENTRE[0] + sx * hx, T.PACK_CENTRE[1] + sy * hy, T.PACK_TOP + 0.004)
            for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        },
        **{f"wound corner {i}": (*c, T.WOUND_TOP) for i, c in enumerate(T.WOUND_CORNERS)},
        "tape roll": (*T.TAPE_CENTRE, T.TOP + T.TAPE_HEIGHT),
        "glove box": (*T.GLOVES_CENTRE, T.TOP + T.GLOVES_SIZE[2]),
    }
    for label, point in points.items():
        u, v, d = project([point])[0]
        inside = 0 <= u < 320 and 0 <= v < 176
        behind = any(nearest.get((round(u) + du, round(v) + dv), np.inf) < d for du in (-1, 0, 1) for dv in (-1, 0, 1))
        print(
            f"  {label:18s} at pixel ({u:5.1f}, {v:5.1f}){'' if inside else ', OUT OF FRAME'}{', BEHIND AN ARM' if behind else ''}"
        )
        assert inside and not behind, label
    a, b = project([(WX, WY, T.SKIN_TOP), (WX + 0.01, WY + 0.01, T.SKIN_TOP)])
    print(f"  1 cm along x is {abs(b[1] - a[1]):.1f} px, 1 cm along y is {abs(b[0] - a[0]):.1f} px, at the wound")


def check_scores():
    done = at()
    off = at(dx=T.NUDGE)
    cases = {
        "start, nothing moved": ([], 0.0),
        "dressing flat over the wound, let go": ([up(), done], 1.0),
        "dressing 1.5 cm off along x: still covers": ([up(), at(dx=0.015)], 1.0),
        "dressing 2 cm off along x: a wound corner shows": ([up(), at(dx=0.02)], 0.5),
        "dressing 2.5 cm off along y: a wound corner shows": ([up(), at(dy=0.025)], 0.5),
        "dressing 5 cm off: the wound's centre shows": ([up(), at(dx=0.05)], 0.0),
        "dressing turned 30 degrees, centred": ([up(), at(yaw=math.radians(30))], 1.0),
        "perfect placement after one key-part touch": ([{**up(), "arms": touching(up())}, done], 0.5),
        "perfect placement after two key-part touches": (
            [{**up(), "arms": touching(up())}, up(), {**up(), "arms": touching(up())}, done],
            0.0,
        ),
        "one touch that lasts five steps counts once": ([*[{**up(), "arms": touching(up())}] * 5, done], 0.5),
        "set down 3 cm off, then slid into place": ([up(), off, at(dx=0.015), done], 0.5),
        "set down 3 cm off, slid into place, one touch": (
            [{**up(), "arms": touching(up())}, off, at(dx=0.015), done],
            0.25,
        ),
        "set down 5 cm off, slid to 2 cm off, one touch": ([{**up(), "arms": touching(up())}, at(dx=0.05), at(dx=0.02)], 0.125),
        "set down 1.5 cm off, slid into place (under 2 cm)": ([up(), at(dx=0.015), done], 1.0),
        "set down 1.5 cm off, lifted, set down centred": ([up(), at(dx=0.015), up(), done], 1.0),
        "dressing left on the pack": ([{"arms": holding(START)}], 0.0),
        "dressing dropped on the drape": ([up(), {"dressing": dressing((0.30, -0.28))}], 0.0),
        "dressing dropped on the table": ([up(), {"dressing": dressing((0.30, -0.42))}], 0.0),
        "dressing held 5 cm over the wound at the end": ([up(), {**up(), "arms": holding(up())}], 0.0),
        "dressing on the wound, the gripper still on it": ([up(), {**done, "arms": holding(done)}], 0.0),
        "dressing on the wound with one edge on the tape roll (8 degrees)": (
            [up(), at(dx=0.03, under=T.SKIN_TOP, pitch=math.radians(8))],
            0.0,
        ),
        "dressing on the wound but still moving": ([up(), at(moving=True)], 0.0),
        "key-part touch, then everything perfect": ([{**START, "arms": touching(START)}, up(), done], 0.5),
        "tape roll knocked over, dressing perfect": ([up(), {**done, "tape": box(TAPE_SIZE, (0.30, -0.40), T.TOP)}], 1.0),
    }
    for name, (moves, want) in cases.items():
        got = play(moves)
        print(f"  {got:.4f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"

    # success is the 1.0 state only, and holds while it lasts
    play([up(), done])
    assert all(step(play.world, hook=T.success) for _ in range(HZ))
    assert not step({**play.world, **at(moving=True)}, hook=T.success)
    assert not step(play.world, holding(play.world), hook=T.success)
    play([up(), at(dx=0.02)])
    assert not step(play.world, hook=T.success)
    play([up(), off, done])
    assert not step(play.world, hook=T.success)  # slid
    play([{**up(), "arms": touching(up())}, done])
    assert not step(play.world, hook=T.success)  # touched

    # two episodes back to back: the memory clears when the clock goes back, and only then
    assert play([{**up(), "arms": touching(up())}, up(), {**up(), "arms": touching(up())}, done]) == 0.0
    assert play([up(), done], restart=False) == 0.0  # same episode
    assert play([up(), done]) == 1.0  # new episode
    scores = {round(c * T.PENALTY**n * f, 4) for c in (0.0, T.PARTIAL, 1.0) for n in (0, 1) for f in (1.0, T.PENALTY)}
    assert scores == {0.0, 0.125, 0.25, 0.5, 1.0}, scores


def _grip_yaw(rot) -> float:
    turn = rot @ grip_rotation(0.0).T
    return math.atan2(turn[1, 0], turn[0, 0])


def run_demo(variant) -> dict:
    plan = ArmPlan(T.demo(variant)["right"], BASE)
    half = np.array(T.DRESSING_SIZE[:2]) / 2
    pose = np.array(T.DRESSING.pos)
    least = {}
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    held, grasp, held_s = False, None, 0
    NOW["t"] = 0.0

    def note(what, value, t, label):
        if value < least.get(what, (np.inf,))[0]:
            least[what] = (value, t, label)

    for n, t in enumerate(np.arange(1 / HZ, plan.duration + 3.0, 1 / HZ)):
        action, label = plan.action7(t)
        arms = {**PARKED, "right": np.r_[action[:6], action[6] * FINGER_OPEN]}
        lo, hi = np.minimum(lo, action[:6]), np.maximum(hi, action[:6])
        tcp, rot, finger, _ = plan.target(t)
        pads_on = abs(finger) <= half[1] + 0.001
        over = np.all(np.abs(tcp[:2] - pose[:2]) < half + 0.01) and tcp[2] < pose[2] + 0.03
        if not held and pads_on and over and finger != FINGER_CLOSED:
            held, grasp = True, pose - tcp
        if held and pads_on:
            pose = tcp + grasp
        elif held:
            pose = np.array([*pose[:2], rest_height(pose[:2]) + T.DRESSING_SIZE[2] / 2])
            held = False
        if label.startswith("push") and finger == FINGER_CLOSED and tcp[2] < pose[2] + 0.02:
            blade = tcp[0] - T.BLADE_HALF
            far = pose[0] + half[0]
            if blade < far and abs(tcp[1] - pose[1]) < half[1]:
                pose[0] -= far - blade
        objects = {"dressing": box(T.DRESSING_SIZE, pose[:2], pose[2] - T.DRESSING_SIZE[2] / 2, moving=held), "tape": tape()}
        final = step(objects, arms)
        held_s = held_s + 1 if T.success(objects) else 0
        note("robot to the key part (tracker)", T.TRACKER.key_gap, t, label)
        if n % EVERY == 0:
            g = gaps(arms, objects)
            for what in (*GROUPS, "the tape roll", "the key part"):
                note(f"robot to {what}", g[what], t, label)
    margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
    return {"plan": plan, "final": final, "held_s": held_s / HZ, "least": least, "margin": margin, "pose": pose}


def check_demo():
    run = run_demo("scripted")
    plan, least, tracker = run["plan"], run["least"], T.TRACKER
    print(
        f"  ideal run: score {run['final']:.2f}, success held {run['held_s']:.1f} s, touches {tracker.touches},"
        f" slid {tracker.slid * MM:.1f} mm, dragged {tracker.dragged}"
    )
    print(f"  dressing centre {(run['pose'][0] - WX) * MM:.1f} mm x, {(run['pose'][1] - WY) * MM:.1f} mm y from the wound's")
    for what, (value, t, label) in least.items():
        print(f"  {value * MM:6.1f} mm  {what}, at {t:.1f} s ({label})")
    print(f"  worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg over {plan.duration:.1f} s")
    print(f"  smallest joint-limit margin per joint (deg): {np.round(np.degrees(run['margin']), 1)}")
    assert run["final"] == 1.0 and run["held_s"] >= T.SUCCESS_HOLD_S and tracker.touches == 0 and not tracker.dragged
    assert least["robot to the key part"][0] >= SAFE
    for what in ("the tape roll", "the saline bottle", "the glove box", "the drape"):
        assert least[f"robot to {what}"][0] >= SAFE, what
    assert least["robot to the pack"][0] >= 0.003 and least["robot to the skin"][0] >= 0.001
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (run["margin"] > 0).all()
    assert plan.duration + 3.0 < 1200 / HZ


def check_nudge():
    run = run_demo("nudge")
    tracker = T.TRACKER
    print(
        f"  set down 3 cm off and pushed into place: slid {tracker.slid * MM:.1f} mm, dragged {tracker.dragged},"
        f" touches {tracker.touches}, score {run['final']:.2f}"
    )
    print(f"  dressing centre {(run['pose'][0] - WX) * MM:.1f} mm x from the wound's; key part {run['least']['robot to the key part'][0] * MM:.1f} mm")
    assert tracker.dragged and tracker.touches == 0 and run["final"] == 0.5


if __name__ == "__main__":
    print("design:")
    check_design()
    print("view:")
    check_view()
    print("scores:")
    check_scores()
    print("demo:")
    check_demo()
    print("nudge variant:")
    check_nudge()
    print("ok")
