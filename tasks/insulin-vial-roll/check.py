"""Offline check of insulin_vial_roll.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/insulin-vial-roll/check.py

1. Design: how far the right arm's grip can turn at the grasp pose before a joint reaches its limit (this sets the
   turns per grasp and the number of grasps), and the room between the robot's collision meshes and the clear
   vial, the mat, the needle box, the meter and the sharps container at the grasp pose, at every grip angle the
   demo uses.
2. Scores: hand-built trials (both vials and the arms step by step, on a stand-in for the episode clock) against
   the score each must get.
3. Demo: plays the right arm's plan through the IK at 20 Hz with the vial following the grip while it is pinched,
   scores it, and prints the turns, the smallest distances, the worst IK error and the joint-limit margins.
4. Lift variant: the same plan with the vial raised 3 cm during each twist, which must score 0.
"""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("insulin_vial_roll.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ, EVERY = 20, 4
MM = 1e3
TURN = 2 * math.pi
PARKED = dict.fromkeys(("left", "right"), np.r_[READY, FINGER_OPEN])
BASE = station.BASES["right"]
MARGIN = math.radians(3.0)  # the joint-limit margin the design check asks for at every grip angle
ROOM = 0.02  # the room the demo must keep from everything but the vial it turns

# the simulator's episode clock and arm joints, which the task file reads from the embodiment
NOW = {"t": 0.0, "arms": PARKED}
T.clock = lambda: NOW["t"]
T.arm_joints = lambda: NOW["arms"]


# ---- states ----
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


def vial(xy=T.SQUARE_CENTRE, yaw=0.0, rise=0.0, tilt=0.0, moving=False) -> ObjectState:
    """A vial standing over xy with its base `rise` above the mat, turned `yaw` about the vertical and leaning
    `tilt` (rad) about its own y axis."""
    q = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    if tilt:
        q = _mul(q, np.array([0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2)]))
    r, h = T.VIAL_RADIUS, T.VIAL_HEIGHT
    rim = np.array([[r * math.cos(a), r * math.sin(a), z] for a in np.linspace(0, TURN, 24) for z in (-h / 2, h / 2)])
    rim = rim @ T._rotation(q).T
    pos = np.array([*xy, T.MAT_TOP + rise - rim[:, 2].min()])
    vel = np.array([0.1 if moving else 0.0, 0.0, 0.0])
    return ObjectState(pos, q, vel, np.zeros(3), pos + rim.min(0), pos + rim.max(0))


def clear(dx=0.0, dy=0.0, yaw=0.0) -> ObjectState:
    x, y = T.CLEAR_CENTRE
    return vial((x + dx, y + dy), yaw)


def box_state(b) -> ObjectState:
    half = np.array(b.size) / 2
    pos = np.array(b.pos)
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), pos - half, pos + half)


START = {"cloudy_vial": vial(), "clear_vial": clear(), **{b.name: box_state(b) for b in T.DRESSING}}


def gripper_at(xyz, yaw, finger=FINGER_OPEN, q0=READY) -> dict:
    """The arms with the right gripper's TCP at a point, pointing down, and the left arm parked."""
    q, miss, _ = ik.solve(xyz, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), q0, BASE)
    assert miss < 1e-3, f"the right gripper cannot reach {xyz} at yaw {math.degrees(yaw):.0f}"
    return {**PARKED, "right": np.r_[q, finger]}


GRASP = (*T.SQUARE_CENTRE, T.GRASP_Z)
PINCH = -(T.VIAL_RADIUS - T.GRIP_IN)
HOLDING = gripper_at(GRASP, T.YAW_LO, PINCH)
HOVERING = gripper_at((*T.SQUARE_CENTRE, T.HOVER), T.YAW_LO)


# ---- trials ----
def step(objects, arms=PARKED, hook=T.score):
    NOW["arms"] = arms
    NOW["t"] += 1 / HZ
    return hook(dict(objects))


STEP = math.radians(30)  # the vial's turn per control step in the hand-built trials


def turning(turns, rise=0.0, start=0.0, per_step=STEP, xy=T.SQUARE_CENTRE, arms=None) -> list:
    """Moves that turn the cloudy vial `turns` times (negative: the other way) in steps of `per_step`, standing
    with its base `rise` above the mat, held by the gripper unless `arms` says otherwise."""
    total, out, yaw = abs(turns) * TURN, [], start
    while total > 1e-9:
        d = min(per_step, total)
        yaw, total = yaw + math.copysign(d, turns), total - d
        out.append({"cloudy_vial": vial(xy, yaw, rise), "arms": HOLDING if arms is None else arms})
    return out


def let_go(yaw=0.0, xy=T.SQUARE_CENTRE) -> dict:
    return {"cloudy_vial": vial(xy, yaw), "arms": HOVERING}


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


# ---- checks ----
def yaw_span() -> tuple:
    """How far the grip turns each way from 0 at the grasp pose before the IK misses or a joint comes within
    MARGIN of a limit, the IK warm-started from the previous angle as the plan is."""
    q0 = gripper_at(GRASP, 0.0)["right"][:6]
    out = []
    for sign in (-1, 1):
        q, last = q0, 0.0
        for deg in range(5, 181, 5):
            yaw = sign * math.radians(deg)
            q, pe, re = ik.solve(GRASP, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), q, BASE, iters=120)
            if pe > 1e-3 or re > math.radians(0.5) or (q - ik.LOWER < MARGIN).any() or (ik.UPPER - q < MARGIN).any():
                break
            last = deg
        out.append(sign * last)
    return tuple(out)


def box_gap(points, b) -> float:
    half, pos = np.array(b.size) / 2, np.array(b.pos)
    return float(np.linalg.norm(np.maximum(np.abs(points - pos) - half, 0.0), axis=1).min())


def room_at(xyz, yaw, finger) -> dict:
    """Smallest distances from the right arm's meshes to the scene around the cloudy vial, at one grip."""
    arms = gripper_at(xyz, yaw, finger)
    points = T.arm_points(arms["right"], "right")
    tips = T.arm_points(arms["right"], "right", ("tip_left", "tip_right"))
    others = [b for b in T.FIXED if b.name != "mat"]
    return {
        "clear vial": T.cylinder_gap(points, START["clear_vial"]),
        "mat (fingertips)": float((tips[:, 2] - T.MAT_TOP).min()),
        "needle box and meter": min(box_gap(points, b) for b in T.DRESSING),
        "tape and sharps container": min(box_gap(points, b) for b in others),
    }


def check_design():
    lo, hi = yaw_span()
    print(
        print(
            f"  grip turns from {lo:+d} to {hi:+d} degrees at the grasp pose"
            f" with every joint {math.degrees(MARGIN):.0f} deg inside its limit"
        )
    )
    print(
        print(
            f"  the demo uses {math.degrees(T.YAW_LO):+.0f} to {math.degrees(T.YAW_HI):+.0f}"
            f" ({math.degrees(T.SWING):.0f} per grasp), {T.CYCLES} grasps,"
            f" {T.CYCLES * math.degrees(T.SWING):.0f} degrees in all"
        )
    )
    assert math.radians(lo) <= T.YAW_LO and math.radians(hi) >= T.YAW_HI, (
        "the demo's swing is outside the reachable span"
    )
    assert T.CYCLES * T.SWING >= T.TURNS_FULL * TURN, "the demo does not turn the vial ten times"
    least = {}
    yaws = np.linspace(T.YAW_LO, T.YAW_HI, 8)
    for yaw in yaws:
        for finger, z in ((PINCH, T.GRASP_Z), (-T.APPROACH / 2, T.GRASP_Z), (-T.APPROACH / 2, T.RETURN_Z)):
            for what, value in room_at((*T.SQUARE_CENTRE, z), yaw, finger).items():
                least[what] = min(least.get(what, math.inf), value)
    for what, value in least.items():
        print(f"  {value * MM:6.1f} mm  robot to the {what} over the demo's grips")
    assert least["clear vial"] >= ROOM and least["needle box and meter"] >= ROOM
    assert least["tape and sharps container"] >= ROOM
    assert least["mat (fingertips)"] >= 0.015  # the pad tips sit at the vial's mid-height, well above the mat
    # the pinch: pads 2.5 mm into a 25 mm vial, tips at its mid-height
    assert abs(2 * PINCH + T.VIAL_RADIUS * 2 - 2 * T.GRIP_IN) < 1e-9
    assert abs((T.GRASP_Z - PAD_TIP_PAST_TCP) - (T.MAT_TOP + T.VIAL_HEIGHT / 2)) < 1e-9


def check_scores():
    ten = [*turning(10), let_go(10 * TURN)]
    lifted_once = [{"cloudy_vial": vial(rise=0.03), "arms": HOLDING}, {"cloudy_vial": vial(), "arms": HOLDING}]
    off_centre = (T.SQUARE_CENTRE[0] + 0.04, T.SQUARE_CENTRE[1])
    cases = {
        "start, nothing moved": ([], 0.0),
        "ten turns, let go, in the square": (ten, 1.0),
        "three full turns, let go": ([*turning(3), let_go(3 * TURN)], 0.3),
        "nine and a half turns, let go": ([*turning(9.5), let_go(9.5 * TURN)], 0.9),
        "six turns one way, then four the other": ([*turning(6), *turning(-4, start=6 * TURN), let_go(2 * TURN)], 1.0),
        "twelve turns, let go": ([*turning(12), let_go(12 * TURN)], 1.0),
        "ten turns, the vial still pinched at the end": (turning(10), 0.0),
        "ten turns, the closed fingertips 1.5 cm above the vial at the end": (
            [
                *turning(10),
                {"arms": gripper_at((*T.SQUARE_CENTRE, T.GRASP_Z + T.VIAL_HEIGHT / 2 + 0.015), T.YAW_LO, PINCH)},
            ],
            0.0,
        ),
        "ten turns, the closed fingertips 2.5 cm above the vial at the end": (
            [
                *turning(10),
                {"arms": gripper_at((*T.SQUARE_CENTRE, T.GRASP_Z + T.VIAL_HEIGHT / 2 + 0.025), T.YAW_LO, PINCH)},
            ],
            1.0,
        ),
        "ten turns, the vial set down 4 cm from the square's centre": (
            [*turning(10), let_go(10 * TURN, off_centre)],
            0.0,
        ),
        "ten turns made 4 cm from the square's centre, then put back": (
            [*turning(10, xy=off_centre), let_go(10 * TURN)],
            0.0,
        ),
        "lifted 3 cm once, put back, then ten turns": ([*lifted_once, *ten], 0.0),
        "lifted 1 cm while turning five times, then five more on the mat": (
            [*turning(5, rise=0.01), *turning(5, start=5 * TURN), let_go(10 * TURN)],
            0.5,
        ),
        "lifted 4 mm while turning ten times": ([*turning(10, rise=0.004), let_go(10 * TURN)], 1.0),
        "tilted 20 degrees once, stood up, then ten turns": (
            [
                {"cloudy_vial": vial(tilt=math.radians(20)), "arms": HOLDING},
                {"cloudy_vial": vial(), "arms": HOLDING},
                *ten,
            ],
            0.0,
        ),
        "tilted 10 degrees while turning, then stood up": (
            [
                *turning(10, per_step=math.radians(30)),
                {"cloudy_vial": vial(yaw=10 * TURN, tilt=math.radians(10)), "arms": HOLDING},
                let_go(10 * TURN),
            ],
            1.0,
        ),
        "ten turns, then the vial left tilted 20 degrees": (
            [*turning(10), {"cloudy_vial": vial(yaw=10 * TURN, tilt=math.radians(20)), "arms": HOVERING}],
            0.0,
        ),
        "ten turns, the vial still moving at the end": (
            [*turning(10), {"cloudy_vial": vial(yaw=10 * TURN, moving=True), "arms": HOVERING}],
            1.0,
        ),
        "clear vial nudged 1.5 cm and put back, then ten turns": (
            [{"clear_vial": clear(dy=0.015)}, {"clear_vial": clear()}, *ten],
            0.0,
        ),
        "clear vial nudged 5 mm, then ten turns": ([{"clear_vial": clear(dy=0.005)}, *ten], 1.0),
        "clear vial turned 30 degrees, then ten turns": ([{"clear_vial": clear(yaw=math.radians(30))}, *ten], 0.0),
        "clear vial turned 15 degrees, then ten turns": ([{"clear_vial": clear(yaw=math.radians(15))}, *ten], 1.0),
        "needle box knocked off the mat, then ten turns": (
            [{"pen_needles": box_state(replace(T.DRESSING[0], pos=(0.50, -0.30, T.TOP + 0.015)))}, *ten],
            1.0,
        ),
        "the vial spun free after letting go (ten turns in all)": (
            [*turning(8), let_go(8 * TURN), *turning(2, start=8 * TURN, arms=HOVERING)],
            1.0,
        ),
        "ten turns in big steps of 170 degrees": ([*turning(10, per_step=math.radians(170)), let_go(10 * TURN)], 1.0),
    }
    for name, (moves, want) in cases.items():
        got = play(moves)
        print(f"  {got:.4f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"

    # success is the 1.0 state only, and holds while it lasts
    play(ten)
    assert all(step(play.world, HOVERING, hook=T.success) for _ in range(HZ))
    assert not step({**play.world, "cloudy_vial": vial(yaw=10 * TURN, moving=True)}, HOVERING, hook=T.success)
    assert not step(play.world, HOLDING, hook=T.success)
    play([*turning(9), let_go(9 * TURN)])
    assert not step(play.world, HOVERING, hook=T.success)

    # two episodes back to back: the memory clears when the clock goes back, and only then
    assert play([*lifted_once, *ten]) == 0.0
    assert play(ten, restart=False) == 0.0  # same episode, the vial back at its start first
    assert play(ten) == 1.0  # new episode
    # the yaw step the tracker can tell apart: under half a turn per control step
    assert play([*turning(10, per_step=math.radians(179)), let_go(10 * TURN)]) == 1.0


def _grip_yaw(rot) -> float:
    turn = rot @ grip_rotation(0.0).T
    return math.atan2(turn[1, 0], turn[0, 0])


def run_demo(variant) -> dict:
    plan = ArmPlan(T.demo(variant)["right"], BASE)
    least = {}
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    pinched, grip_yaw0, vial_yaw0 = False, 0.0, 0.0
    vial_yaw, vial_xy, vial_rise = 0.0, np.array(T.SQUARE_CENTRE), 0.0
    held_s, final = 0, 0.0
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
        if abs(finger) <= T.VIAL_RADIUS + 0.001:  # the pads are on the vial: it turns and moves with the grip
            if not pinched:
                pinched, grip_yaw0, vial_yaw0 = True, yaw, vial_yaw
            vial_yaw = (
                vial_yaw0 + T.wrap(yaw - grip_yaw0) if abs(yaw - grip_yaw0) < math.pi else vial_yaw0 + (yaw - grip_yaw0)
            )
            vial_xy, vial_rise = tcp[:2], max(tcp[2] - T.GRASP_Z, 0.0)
        else:
            pinched, vial_rise = False, 0.0
        objects = {**START, "cloudy_vial": vial(vial_xy, vial_yaw, vial_rise, moving=pinched)}
        final = step(objects, arms)
        held_s = held_s + 1 if T.success(objects) else 0
        if n % EVERY == 0:
            points = T.arm_points(arms["right"], "right")
            note("robot to the clear vial", T.cylinder_gap(points, START["clear_vial"]), t, label)
            note("robot to the needle box and meter", min(box_gap(points, b) for b in T.DRESSING), t, label)
            note(
                "robot to the sharps container",
                min(box_gap(points, b) for b in T.FIXED if b.name.startswith("sharps")),
                t,
                label,
            )
            note("fingertips above the mat", float((points[:, 2] - T.MAT_TOP).min()), t, label)
            if not pinched:
                note("open gripper to the cloudy vial", T.cylinder_gap(points, objects["cloudy_vial"]), t, label)
    margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
    return {
        "plan": plan,
        "final": final,
        "held_s": held_s / HZ,
        "least": least,
        "margin": margin,
        "turns": T.TRACKER.total_rot / TURN,
        "spoiled": T.TRACKER.spoiled,
    }


def check_demo():
    run = run_demo("scripted")
    plan, least = run["plan"], run["least"]
    print(f"  ideal run: {run['turns']:.2f} turns, score {run['final']:.2f}, success held {run['held_s']:.1f} s")
    for what, (value, t, label) in least.items():
        print(f"  {value * MM:7.1f} mm  {what}, at {t:.1f} s ({label})")
    print(
        print(
            f"  worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg"
            f" over {plan.duration:.1f} s ({plan.duration * HZ:.0f} steps)"
        )
    )
    print(f"  smallest joint-limit margin per joint (deg): {np.round(np.degrees(run['margin']), 1)}")
    assert run["final"] == 1.0 and run["held_s"] >= T.SUCCESS_HOLD_S and not run["spoiled"]
    assert run["turns"] >= T.TURNS_FULL * T.SLACK - 0.1
    assert least["robot to the clear vial"][0] >= ROOM and least["robot to the needle box and meter"][0] >= ROOM
    assert least["robot to the sharps container"][0] >= ROOM and least["fingertips above the mat"][0] >= 0.015
    assert least["open gripper to the cloudy vial"][0] >= 0.005
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (run["margin"] > 0).all()
    assert plan.duration + 3.0 < 2400 / HZ


def check_lift():
    run = run_demo("lift")
    print(
        print(
            f"  vial raised {T.LIFT_SPOIL * 100:.0f} cm during each twist: {run['turns']:.2f} turns counted,"
            f" spoiled {run['spoiled']}, score {run['final']:.2f}"
        )
    )
    assert run["spoiled"] and run["final"] == 0.0


if __name__ == "__main__":
    print("design:")
    check_design()
    print("scores:")
    check_scores()
    print("demo:")
    check_demo()
    print("lift variant:")
    check_lift()
    print("ok")
