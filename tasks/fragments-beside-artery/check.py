"""Offline check of fragments_beside_artery.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/fragments-beside-artery/check.py

1. Design: places the right arm's collision meshes at each fragment by IK and measures the room to the artery for
   the default grasp (fingers across the artery, fully open) and for the careful one (fingers along it, narrow).
2. Scores: hand-built trials (the artery, the fragments and the arms step by step, on a stand-in for the episode
   clock) against the score each must get.
3. Demo: plays the right arm's plan through the IK at 20 Hz with each fragment following the gripper that holds
   it, scores it, and prints the smallest distances from the robot to the artery, the pad, the tray and the
   fragments it is not holding, the worst IK error and the joint limits.
4. Naive variant: the same with the default grasp, the artery standing still, counting the touches.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("fragments_beside_artery.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ, EVERY = 20, 4
MM = 1e3
SAFE = T.TOUCH_CLEAR  # the room the demo must keep
PARKED = dict.fromkeys(("left", "right"), np.r_[READY, FINGER_OPEN])
BASE = station.BASES["right"]

# the simulator's episode clock and arm joints, which the task file reads from the embodiment
NOW = {"t": 0.0, "arms": PARKED}
T.clock = lambda: NOW["t"]
T.arm_joints = lambda: NOW["arms"]


# ---- states ----
def box(size, xy, under, yaw=0.0, moving=False, tilt=0.0) -> ObjectState:
    """A box with its centre over xy and its underside at height `under`, turned yaw about the vertical (and
    tilt about its own y axis, for the artery)."""
    q_yaw = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    q = q_yaw if not tilt else _mul(q_yaw, np.array([0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2)]))
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.array(size) / 2
    corners = corners @ T._rotation(q).T
    pos = np.array([*xy, under - corners[:, 2].min()])
    vel = np.array([0.1 if moving else 0.0, 0.0, 0.0])
    return ObjectState(pos, q, vel, np.zeros(3), pos + corners.min(0), pos + corners.max(0))


def _mul(a, b) -> np.ndarray:
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def fragment(name, xy=None, under=T.WOUND_TOP, **kwargs) -> ObjectState:
    start = T.FRAGMENTS[name]
    return box(T.SIZES[name], start.pos[:2] if xy is None else xy, under, kwargs.pop("yaw", start.yaw), **kwargs)


def artery(dx=0.0, dy=0.0, up=0.0, yaw=0.0, tilt=0.0) -> ObjectState:
    x, y = T.CENTRE
    return box(T.ARTERY_SIZE, (x + dx, y + dy), T.WOUND_TOP + up, yaw, tilt=tilt)


START = {"artery": artery(), **{name: fragment(name) for name in T.VALUES}}
A, B, C, D = T.ORDER
BESIDE_DRAPE = (T.CENTRE[0] - 0.16, T.CENTRE[1])  # on the drape, outside the skin pad


def up(name, under=0.05, **kwargs) -> dict:
    """The fragment lifted over where it lay, its underside `under` above the wound."""
    return {name: fragment(name, under=T.WOUND_TOP + under, **kwargs)}


def tray(name, under=0.0, **kwargs) -> dict:
    """The fragment over its place in the tray, its underside `under` above the tray's floor."""
    return {name: fragment(name, T.TRAY_SPOTS[name], T.TRAY_FLOOR_TOP + under, **kwargs)}


def over_artery(name, above=0.001) -> dict:
    """The fragment carried across the artery, its underside `above` the artery's top."""
    return {name: fragment(name, (T.FRAGMENTS[name].pos[0], T.CENTRE[1]), T.WOUND_TOP + T.ARTERY_SIZE[2] + above)}


def taken(*names) -> list:
    """Each fragment lifted out and set down in the tray, one after the other."""
    return [move for name in names for move in (up(name), tray(name))]


def gripper_at(xyz, yaw, finger=FINGER_OPEN) -> dict:
    """The arms with the right gripper's TCP at a point, pointing down, and the left arm parked."""
    q, miss, _ = ik.solve(xyz, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), READY, BASE)
    assert miss < 1e-3, f"the right gripper cannot reach {xyz} at yaw {math.degrees(yaw):.0f}"
    return {**PARKED, "right": np.r_[q, finger]}


def fingers_beside_artery(gap) -> dict:
    """The right gripper, fingers along the artery and 3 cm open, with a fingertip `gap` from the artery's side."""

    def arms(back):  # the TCP this far from the artery's side, on the right arm's side of it
        y = T.CENTRE[1] - T.ARTERY_SIZE[1] / 2 - back
        return gripper_at((0.31, y, T.WOUND_TOP + 0.004 + PAD_TIP_PAST_TCP), math.pi / 2, -0.015)

    near, far = 0.0, 0.04
    for _ in range(16):
        mid = (near + far) / 2
        near, far = (mid, far) if T.robot_gap(START, arms(mid)) < gap else (near, mid)
    return {"arms": arms(far)}


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
    def room(name, yaw, opening, tips_up) -> float:
        x, y, _ = T.FRAGMENTS[name].pos
        arms = gripper_at((x, y, T.WOUND_TOP + tips_up + PAD_TIP_PAST_TCP), yaw, -opening / 2)
        return T.robot_gap(START, arms)

    wide, careful = -2 * FINGER_OPEN, {}
    for name in T.ORDER:
        size, yaw = T.SIZES[name], T.FRAGMENTS[name].yaw
        mid = size[2] / 2
        default = room(name, 0.0, wide, mid)
        closing = min(room(name, 0.0, w, mid) for w in np.linspace(wide, size[1], 16))
        any_yaw = min(room(name, math.radians(a), wide, mid) for a in range(-90, 91, 15))
        careful[name] = room(name, yaw + math.pi / 2, size[0] + T.APPROACH, T.PADS_ABOVE_UNDERSIDE)
        print(
            f"  {name}: default grasp {default * MM:4.1f} mm from the artery coming down, {closing * MM:4.1f} mm"
            f" while closing; fully open at any yaw {any_yaw * MM:4.1f} mm; careful grasp {careful[name] * MM:4.1f} mm"
        )
        if name == A:  # R1: safe for any grasp
            assert min(default, closing, any_yaw) >= SAFE, f"{name} is not safe for every grasp"
        else:  # R2, R3: the default grasp touches, the careful one does not
            assert min(default, closing) < T.TOUCH, f"the default grasp does not touch the artery at {name}"
            assert careful[name] >= SAFE, f"the careful grasp of {name} is too close to the artery"
    assert careful[D] < careful[C] < careful[B] < careful[A], "the room for the careful grasp is not graded"


def check_scores():
    brush = fingers_beside_artery(0.001)
    assert T.robot_gap(START, brush["arms"]) < T.TOUCH < T.robot_gap(START, fingers_beside_artery(0.0035)["arms"])
    lingers = [brush, fingers_beside_artery(0.0035), brush]  # never 5 mm clear in between: one touch
    twice = [brush, {}, brush]
    holding_d = gripper_at((*T.TRAY_SPOTS[D], T.TRAY_FLOOR_TOP + 0.004 + PAD_TIP_PAST_TCP), 0.0, -0.005)
    over_tray = gripper_at((*T.TRAY_SPOTS[D], T.TRAY_FLOOR_TOP + 0.044 + PAD_TIP_PAST_TCP), 0.0, -0.005)
    pushed = [{"artery": artery(dy=-0.02)}, {"artery": artery()}]
    cases = {
        "start, nothing moved": ([], 0.0),
        "all four in the tray": (taken(A, B, C, D), 1.0),
        "A and B in the tray, C and D left in the wound": (taken(A, B), 0.3),
        "A, B and C in the tray, D left": (taken(A, B, C), 0.6),
        "only D in the tray": (taken(D), 0.4),
        "all four, in the opposite order": (taken(D, C, B, A), 1.0),
        "all four, D carried 1 mm over the artery (one touch)": (
            [*taken(A, B, C), up(D), over_artery(D), tray(D)],
            0.5,
        ),
        "all four, C carried 1 mm over the artery (one touch)": ([up(C), over_artery(C), *taken(A, B, C, D)], 0.5),
        "all four, a finger 1 mm from the artery with nothing in it (one touch)": ([brush, *taken(A, B, C, D)], 0.5),
        "all four, the finger lingers 1 to 3.5 mm from the artery (still one touch)": (
            [*lingers, *taken(A, B, C, D)],
            0.5,
        ),
        "all four, two touches": ([*twice, *taken(A, B, C, D)], 0.25),
        "all four, three touches": ([*twice, {}, brush, *taken(A, B, C, D)], 0.0),
        "A and B, one touch": ([brush, *taken(A, B)], 0.15),
        "A dropped on the drape outside the pad, B in the tray": (
            [up(A), {A: fragment(A, BESIDE_DRAPE, T.TOP + T.DRAPE[2])}, *taken(B)],
            0.25,
        ),
        "A dropped on the table, B in the tray": ([up(A), {A: fragment(A, (0.10, -0.20), T.TOP)}, *taken(B)], 0.25),
        "D in the tray on top of C": (
            [*taken(A, B, C), up(D), {D: fragment(D, T.TRAY_SPOTS[C], T.TRAY_FLOOR_TOP + T.SIZES[C][2])}],
            0.8,
        ),
        "D in the tray, the gripper still on it": ([*taken(A, B, D), {"arms": holding_d}], 0.3),
        "all four in the tray, the gripper still on D and 1.4 cm from C": (
            [*taken(A, B, C, D), {"arms": holding_d}],
            0.3,
        ),
        "D held 4 cm over the tray": ([*taken(A, B, C), up(D), {**tray(D, under=0.04), "arms": over_tray}], 0.6),
        "D in the tray, still moving": ([*taken(A, B, C), up(D), tray(D, moving=True)], 0.6),
        "B slid 4 cm across the wound, then lifted into the tray": (
            [{B: fragment(B, np.add(T.FRAGMENTS[B].pos[:2], (-0.04, 0.0)))}, *taken(A, B)],
            0.1,
        ),
        "A pushed off the pad, never lifted": ([{A: fragment(A, BESIDE_DRAPE, T.TOP + T.DRAPE[2])}], 0.0),
        "A lifted and put back on the wound": ([up(A), {A: fragment(A)}], 0.0),
        "A lifted and left on the skin pad": (
            [up(A), {A: fragment(A, (T.CENTRE[0] - 0.115, T.CENTRE[1]), T.TOP + T.DRAPE[2] + T.SKIN[2])}],
            0.0,
        ),
        "A in the tray, then taken out and left on the wound": ([*taken(A), up(A), {A: fragment(A)}], 0.0),
        "A dropped off the table": ([up(A), {A: fragment(A, (0.30, -0.80), 0.0)}], 0.0),
        "artery pushed 2 cm and pushed back, then all four": ([*pushed, *taken(A, B, C, D)], 0.0),
        "artery pushed 8 mm and pushed back, then all four": (
            [{"artery": artery(dy=0.008)}, {"artery": artery()}, *taken(A, B, C, D)],
            1.0,
        ),
        "artery turned 15 degrees and turned back, then all four": (
            [{"artery": artery(yaw=math.radians(15))}, {"artery": artery()}, *taken(A, B, C, D)],
            0.0,
        ),
        "artery lifted 1 cm and put back, then all four": (
            [{"artery": artery(up=0.01)}, {"artery": artery()}, *taken(A, B, C, D)],
            0.0,
        ),
        "one end of the artery raised 1 cm and put back, then all four": (
            [{"artery": artery(tilt=math.asin(0.01 / T.ARTERY_SIZE[0]))}, {"artery": artery()}, *taken(A, B, C, D)],
            0.0,
        ),
        "one touch, everything back at its start, then all four": ([brush, dict(START), *taken(A, B, C, D)], 0.5),
        "artery pushed, everything back at its start, then all four": ([*pushed, dict(START), *taken(A, B, C, D)], 0.0),
    }
    for name, (moves, want) in cases.items():
        got = play(moves)
        print(f"  {got:.4f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"

    # success is the 1.0 state only, and holds while it lasts
    play(taken(A, B, C, D))
    assert all(step(play.world, hook=T.success) for _ in range(HZ))
    assert not step({**play.world, **tray(D, moving=True)}, hook=T.success)
    assert not step(play.world, holding_d, hook=T.success)
    play([brush, *taken(A, B, C, D)])
    assert not step(play.world, hook=T.success)  # a touch: 0.5, and the episode runs on
    play(taken(A, B, C))
    assert not step(play.world, hook=T.success)

    # two episodes back to back: the memory clears when the clock goes back, and only then
    assert play([*twice, *taken(A, B, C, D)]) == 0.25
    assert play(taken(A, B, C, D), restart=False) == 0.25  # same episode, everything put back first
    assert play(taken(A, B, C, D)) == 1.0  # new episode
    # every sum of full, half or no shares, halved up to twice, is a multiple of 0.0125
    values = np.array(list(T.VALUES.values()))
    for shares in np.ndindex(3, 3, 3, 3):
        for touches in range(T.MAX_TOUCHES):
            steps = values @ (np.array(shares) / 2) * T.TOUCH_FACTOR**touches / 0.0125
            assert abs(steps - round(steps)) < 1e-9


def _grip_yaw(rot) -> float:
    turn = rot @ grip_rotation(0.0).T
    return math.atan2(turn[1, 0], turn[0, 0])


def _rest_height(xy) -> float:
    """Height of what a fragment let go over xy comes to rest on (the artery aside)."""
    if T._inside(xy, T.TRAY_CENTRE, T.TRAY_INSIDE):
        return T.TRAY_FLOOR_TOP
    for size, top in ((T.WOUND, T.WOUND_TOP), (T.SKIN, T.TOP + T.DRAPE[2] + T.SKIN[2]), (T.DRAPE, T.TOP + T.DRAPE[2])):
        if T._inside(xy, T.CENTRE, size[:2]):
            return top
    return T.TOP


def run_demo(variant) -> dict:
    plan = ArmPlan(T.demo(variant)["right"], BASE)
    width = {name: T.SIZES[name][0 if variant == "scripted" else 1] for name in T.ORDER}
    pose = {name: (np.array(T.FRAGMENTS[name].pos), T.FRAGMENTS[name].yaw) for name in T.ORDER}
    parts = [(b.size, box(b.size, b.pos[:2], b.pos[2] - b.size[2] / 2)) for b in T.FIXED]  # none is turned
    fixed = {"the tray": parts[3:], "the pad and the drape": parts[:3]}
    least = {}
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    held, grasp, name, held_s = None, None, T.ORDER[0], 0
    NOW["t"] = 0.0

    def note(what, value, t, label):
        if value < least.get(what, (np.inf,))[0]:
            least[what] = (value, t, label)

    for n, t in enumerate(np.arange(1 / HZ, plan.duration + 3.0, 1 / HZ)):
        action, label = plan.action7(t)
        arms = {**PARKED, "right": np.r_[action[:6], action[6] * FINGER_OPEN]}
        lo, hi = np.minimum(lo, action[:6]), np.maximum(hi, action[:6])
        tcp, rot, finger, _ = plan.target(t)
        yaw = _grip_yaw(rot)
        if label != "done":
            name = f"fragment_{label[0]}"
        if abs(finger) <= width[name] / 2 + 0.001:  # the pads are on the fragment
            if held is None:
                held, grasp = name, (pose[name][0] - tcp, pose[name][1], yaw)
            offset, yaw0, grip0 = grasp
            c, s = math.cos(yaw - grip0), math.sin(yaw - grip0)
            pose[held] = (tcp + np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ offset, yaw0 + yaw - grip0)
        elif held is not None:  # let go: the fragment comes to rest on what is under it
            xyz, turned = pose[held]
            pose[held] = (np.array([*xyz[:2], _rest_height(xyz[:2]) + T.SIZES[held][2] / 2]), turned)
            held = None
        objects = {"artery": artery()}
        for other, (xyz, turned) in pose.items():
            objects[other] = fragment(other, xyz[:2], xyz[2] - T.SIZES[other][2] / 2, yaw=turned, moving=other == held)
        final = step(objects, arms)
        held_s = held_s + 1 if T.success(objects) else 0
        note("robot to the artery", T.TRACKER.gaps[0], t, label)
        note("fragments to the artery", T.TRACKER.gaps[1], t, label)
        for other in T.ORDER:
            if other != name:
                note("gripper to a fragment it is not taking", T.gripper_gap(objects, other, arms), t, label)
        if n % EVERY == 0:
            points = T.arm_points(arms["right"], "right")
            tips = T.arm_points(arms["right"], "right", ("tip_left", "tip_right"))
            for what, boxes in fixed.items():
                note(f"robot to {what}", min(T.box_gap(points, b, size) for size, b in boxes), t, label)
            note("fingers to the tray's walls", min(T.box_gap(tips, b, size) for size, b in parts[4:]), t, label)
    margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
    return {
        "plan": plan,
        "final": final,
        "held_s": held_s / HZ,
        "least": least,
        "margin": margin,
        "touches": T.TRACKER.touches,
    }


def check_demo():
    run = run_demo("scripted")
    plan, least = run["plan"], run["least"]
    print(f"  ideal run: score {run['final']:.2f}, success held {run['held_s']:.1f} s, {run['touches']} touches")
    for what, (value, t, label) in least.items():
        shown = "over 50.0" if value >= T.NEAR else f"{value * MM:9.1f}"
        print(f"  {shown} mm  {what}, at {t:.1f} s ({label})")
    print(f"  worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg over {plan.duration:.1f} s")
    print(f"  smallest joint-limit margin per joint (deg): {np.round(np.degrees(run['margin']), 1)}")
    assert run["final"] == 1.0 and run["held_s"] >= T.SUCCESS_HOLD_S and run["touches"] == 0
    assert least["robot to the artery"][0] >= SAFE and least["fragments to the artery"][0] >= SAFE
    assert least["robot to the tray"][0] >= SAFE and least["gripper to a fragment it is not taking"][0] >= SAFE
    assert least["robot to the pad and the drape"][0] >= T.PADS_ABOVE_UNDERSIDE - 1e-4  # the fingertips, at a grasp
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (run["margin"] > 0).all()
    assert plan.duration + 3.0 < 2400 / HZ


def check_naive():
    run = run_demo("naive")
    print(f"  default grasp, artery standing still: {run['touches']} touches, score {run['final']:.2f}")
    assert run["touches"] >= 2


if __name__ == "__main__":
    print("design:")
    check_design()
    print("scores:")
    check_scores()
    print("demo:")
    check_demo()
    print("naive variant:")
    check_naive()
    print("ok")
