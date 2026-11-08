"""Offline check of pivot_through_port.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/pivot-through-port/check.py

1. Design: the rod's room to the posts along the swing to each peg, and the room a sideways slide leaves (none).
2. Scores: hand-built trials (the rod, the posts, the pegs and the arms step by step, on a stand-in for the
   episode clock) against the score each must get.
3. Demo: plays the right arm's plan through the IK at 20 Hz with the rod following the gripper and each peg
   falling when the rod's tip reaches it, scores it, and prints the smallest distances from the rod and the robot
   to the posts, the wall and the lintel, the worst IK error and the joint limits.
4. Naive variant: the same with the gripper sliding sideways instead of swinging, counting the touches.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("pivot_through_port.py"))
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


def box(size, xy, under, yaw=0.0, tilt=0.0, moving=False) -> ObjectState:
    """A box with its centre over xy and its lowest point at height `under`, turned yaw about the vertical and
    tilted about its own y axis."""
    q = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    if tilt:
        q = _mul(q, np.array([0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2)]))
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.array(size) / 2
    corners = corners @ T._rotation(q).T
    pos = np.array([*xy, under - corners[:, 2].min()])
    vel = np.array([0.1 if moving else 0.0, 0.0, 0.0])
    return ObjectState(pos, q, vel, np.zeros(3), pos + corners.min(0), pos + corners.max(0))


def rod(xy=None, under=T.REST_TOP, yaw=0.0, **kwargs) -> dict:
    """The rod with its centre over xy (default: on its blocks), lowest point at `under`, turned yaw."""
    xy = T.ROD.pos[:2] if xy is None else xy
    return {"rod": box(T.ROD_SIZE, xy, under, yaw, **kwargs)}


def rod_through(angle, tip_past, under=T.TOP + T.PORT_ROD_BOTTOM, **kwargs) -> dict:
    """The rod with its axis through the slot's centre at `angle`, its tip `tip_past` beyond the centre."""
    half = T.ROD_SIZE[0] / 2
    xy = (T.SLOT_X + (tip_past - half) * math.cos(angle), T.Y0 + (tip_past - half) * math.sin(angle))
    return rod(xy, under, angle, **kwargs)


def post(name, dy=0.0, dx=0.0, tilt_deg=0.0, moving=False) -> dict:
    x, y = T.POSTS[name].pos[:2]
    return {name: box(T.POST_SIZE, (x + dx, y + dy), T.TOP, tilt=math.radians(tilt_deg), moving=moving)}


def peg(name, down=False, dx=0.0) -> dict:
    x, y = T.PEGS[name].pos[:2]
    if down:  # lying flat beyond its mark, pushed over away from the wall
        return {name: box(T.PEG_SIZE, (x + T.PEG_SIZE[2] / 2 + dx, y), T.TOP, tilt=math.pi / 2)}
    return {name: box(T.PEG_SIZE, (x + dx, y), T.TOP)}


def pegs_down(*names) -> dict:
    out = {}
    for name in names:
        out.update(peg(name, down=True))
    return out


START = {**rod(), **{k: v for n in T.POSTS for k, v in post(n).items()}, **{k: v for n in T.PEGS for k, v in peg(n).items()}}
AXIS, NEAR, FAR = "peg_axis", "peg_near", "peg_far"


def gripper_at(xyz, yaw=0.0, finger=FINGER_OPEN) -> dict:
    """The arms with the right gripper's TCP at a point, pointing down, and the left arm parked."""
    q, miss, _ = ik.solve(np.array(xyz), ik.matrix_to_quat_xyzw(grip_rotation(yaw)), READY, BASE)
    assert miss < 1e-3, f"the right gripper cannot reach {xyz} at yaw {math.degrees(yaw):.0f}"
    return {**PARKED, "right": np.r_[q, finger]}


def fingers_before_post(gap) -> dict:
    """The right gripper, fingers closing along y and 2 cm open, at the left post's mid-height, with a fingertip `gap`
    from the post's near face."""

    def arms(back):
        return gripper_at((T.WALL_X[0] - back, T.Y0 + T.SLOT_W / 2 + T.POST_SIZE[1] / 2, T.TOP + T.POST_SIZE[2] / 2 + 0.013), 0.0, -0.01)

    near, far = 0.0, 0.05
    for _ in range(16):
        mid = (near + far) / 2
        near, far = (mid, far) if T.post_gap(T.robot_points(arms(mid)), START) < gap else (near, mid)
    return {"arms": arms(far)}


def holding(angle, tip_past) -> dict:
    """The right gripper on the rod that rod_through(angle, tip_past) describes."""
    return gripper_at(T._on_ray(angle, T.GRIP_FROM_TIP - tip_past, T.PORT_Z), angle, T.GRIP)


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


def wall_boxes() -> list:
    return [(b.size, box(b.size, b.pos[:2], b.pos[2] - b.size[2] / 2)) for b in T.FIXED[:3]]


def check_design():
    walls = wall_boxes()
    for name, (angle, reach) in T.PEG_ANGLES.items():
        least = math.inf
        for tip_past in np.linspace(-T.BACK_OFF - T.WALL_T / 2, reach + T.PUSH_PAST, 40):
            points = T.rod_points(rod_through(angle, tip_past)["rod"])
            least = min(least, T.post_gap(points, START), *(T.box_gap(points, b, size) for size, b in walls))
        pushed = min(
            T.post_gap(T.rod_points(rod((T.WALL_X[1] + T.NAIVE_TIP_PAST - T.ROD_SIZE[0] / 2, T.Y0 + dy), T.TOP + T.PORT_ROD_BOTTOM)["rod"]), START)
            for dy in np.linspace(0.0, T.PEG_SPECS[name][0], 40)
        )
        print(f"  {name}: {math.degrees(angle):5.1f} degrees, rod to the posts and wall along its ray {least * MM:4.1f} mm; slid sideways instead {pushed * MM:4.1f} mm")
        assert least >= SAFE, f"the ray to {name} brings the rod within {least * MM:.1f} mm of a post or the wall"
        if name != AXIS:
            assert pushed < T.TOUCH, f"sliding sideways to {name} does not push a post"
    assert T.PEG_ANGLES[FAR][1] + T.PUSH_PAST + T.BACK_OFF < T.GRIP_FROM_TIP
    # the arms at their ready pose are not in the cavity
    assert not T.in_cavity(T.robot_points(PARKED))
    tips = T.robot_points(PARKED)
    print(f"  ready pose: robot points reach x {tips[:, 0].max() * 100:.1f} cm, lowest point past x 30 cm at {(tips[tips[:, 0] > 0.30][:, 2].min() - T.TOP) * 100:.1f} cm")


def check_scores():
    # the rod in the slot, shifted sideways until it is 1 mm from the left post (a touch), then back to the axis
    def shifted(dy) -> dict:
        return rod((T.SLOT_X + 0.15 - T.ROD_SIZE[0] / 2, T.Y0 + dy), T.TOP + T.PORT_ROD_BOTTOM)

    touch_dy = next(dy for dy in np.linspace(0.0, 0.03, 600) if T.post_gap(T.rod_points(shifted(dy)["rod"]), START) < 0.001)
    graze = shifted(touch_dy)
    assert T.post_gap(T.rod_points(graze["rod"]), START) < T.TOUCH
    home = rod_through(0.0, 0.06)
    out = rod((T.WALL_X[0] - T.BACK_OFF - T.ROD_SIZE[0] / 2, T.Y0), T.TOP + T.PORT_ROD_BOTTOM)
    over = rod((T.SLOT_X, -0.30), T.TOP + T.WALL_H + 0.01)  # the rod lifted over the right wall half
    round_end = rod((T.WALL_X[1] + 0.05, -0.33), T.TOP + 0.04, yaw=math.radians(60))  # round the wall's end, aimed at the far peg
    held_parked = gripper_at((T.ROD_X[1] - T.GRIP_FROM_TIP, T.ROD_Y, T.GRASP_Z), 0.0, T.GRIP)
    cases = {
        "start, nothing moved": ([], 0.0),
        "all three pegs down, rod back on its blocks": ([home, pegs_down(AXIS, NEAR, FAR), out, rod()], 1.0),
        "axis peg only": ([home, pegs_down(AXIS), out, rod()], 0.2),
        "axis and near pegs": ([home, pegs_down(AXIS, NEAR), out, rod()], 0.5),
        "far peg only": ([home, pegs_down(FAR), out, rod()], 0.5),
        "all three, one touch": ([home, graze, home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.5),
        "all three, the rod lingers 1 to 4 mm from the post (still one touch)": (
            [home, graze, shifted(touch_dy - 0.003), graze, home, pegs_down(AXIS, NEAR, FAR), out, rod()],
            0.5,
        ),
        "all three, two touches": ([home, graze, home, graze, home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.25),
        "all three, three touches": ([home, graze, home, graze, home, graze, home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.0),
        "all three, rod left in the slot": ([home, pegs_down(AXIS, NEAR, FAR)], 0.5),
        "all three, rod withdrawn and left on the table in front of the wall": ([home, pegs_down(AXIS, NEAR, FAR), rod((0.20, -0.15), T.TOP)], 0.5),
        "all three, rod on its blocks but still held": ([home, pegs_down(AXIS, NEAR, FAR), out, {**rod(), "arms": held_parked}], 0.5),
        "all three, rod held 2 cm above its blocks": ([home, pegs_down(AXIS, NEAR, FAR), out, rod(under=T.REST_TOP + 0.02)], 0.5),
        "all three, rod on its blocks but still moving": ([home, pegs_down(AXIS, NEAR, FAR), out, rod(moving=True)], 0.5),
        "all three, rod on one block only": ([home, pegs_down(AXIS, NEAR, FAR), out, rod((T.ROD.pos[0] + 0.10, T.ROD_Y))], 0.5),
        "axis and near pegs, one touch, rod on its blocks": ([home, graze, home, pegs_down(AXIS, NEAR), out, rod()], 0.25),
        "post nudged 5 mm and left there, all three pegs (one touch)": (
            [home, {**post("post_left", dy=0.005, moving=True)}, post("post_left", dy=0.005), home, pegs_down(AXIS, NEAR, FAR), out, rod()],
            0.5,
        ),
        "post tilted 5 degrees and stood back, all three (one touch)": (
            [home, post("post_left", tilt_deg=5.0, moving=True), post("post_left"), home, pegs_down(AXIS, NEAR, FAR), out, rod()],
            0.5,
        ),
        "post pushed 2 cm and stood back on its mark before the end": ([home, post("post_left", dy=0.02, moving=True), post("post_left"), home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.0),
        "post tilted 20 degrees and stood back": ([home, post("post_right", tilt_deg=20.0, moving=True), post("post_right"), home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.0),
        "post knocked over": ([home, post("post_right", tilt_deg=90.0, dy=-0.05), home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.0),
        "post lifted out and put back": ([{**post("post_right", dx=-0.03)}, post("post_right"), home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.0),
        "pegs in the opposite order": ([home, pegs_down(FAR), pegs_down(FAR, NEAR), pegs_down(FAR, NEAR, AXIS), out, rod()], 1.0),
        "peg knocked down, then knocked upright again": ([home, pegs_down(AXIS), peg(AXIS, dx=0.08), out, rod()], 0.2),
        "peg pushed 2 cm but still standing": ([home, peg(AXIS, dx=0.02), out, rod()], 0.0),
        "rod passed over the right wall half": ([over, pegs_down(FAR), rod()], 0.0),
        "rod round the wall's end": ([round_end, pegs_down(FAR), rod()], 0.0),
        "rod round the end, then properly through the slot for the rest": ([round_end, pegs_down(FAR), home, pegs_down(FAR, NEAR, AXIS), out, rod()], 0.0),
        "fingertip 1 mm from a post with the rod not in the slot (one touch)": (
            [fingers_before_post(0.001), home, pegs_down(AXIS, NEAR, FAR), out, rod()],
            0.5,
        ),
        "fingertip 4 mm from a post (no touch)": ([fingers_before_post(0.004), home, pegs_down(AXIS, NEAR, FAR), out, rod()], 1.0),
        "one touch, everything back at its start, then all three": ([home, graze, home, out, dict(START), home, pegs_down(AXIS, NEAR, FAR), out, rod()], 0.5),
    }
    for name, (moves, want) in cases.items():
        got = play(moves)
        print(f"  {got:.4f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"

    # the robot behind the wall (a straight-down grip cannot reach there, so the points are placed by hand): the
    # cavity starts behind the wall's far face and ends 2 cm above the wall top
    original = T.robot_points
    for point, want in (
        ((T.WALL_X[1] + 0.01, -0.30, T.TOP + T.CAVITY_Z - 0.01), 0.0),  # a fingertip over the wall and down behind it
        ((T.WALL_X[1] + 0.03, T.WALL_Y[0] - 0.03, T.TOP + 0.05), 0.0),  # round the wall's end at peg height
        ((T.WALL_X[1] + 0.01, -0.30, T.TOP + T.CAVITY_Z + 0.01), 1.0),  # passing high over the wall
        ((T.WALL_X[1] - 0.03, -0.30, T.TOP + 0.05), 1.0),  # in front of the wall, low
    ):
        T.robot_points = lambda arms, p=point: np.array([p])
        play([{}])
        T.robot_points = original
        got = play([home, pegs_down(AXIS, NEAR, FAR), out, rod()], restart=False)
        print(f"  {got:.4f}  robot point at x {point[0] * 100:.0f} y {point[1] * 100:.0f} height {(point[2] - T.TOP) * 100:.0f} cm")
        assert got == want

    # success is the 1.0 state only, and holds while it lasts
    play([home, pegs_down(AXIS, NEAR, FAR), out, rod()])
    assert all(step(play.world, hook=T.success) for _ in range(HZ))
    assert not step({**play.world, **rod(moving=True)}, hook=T.success)
    assert not step(play.world, held_parked, hook=T.success)
    assert not step({**play.world, **post("post_left", moving=True)}, hook=T.success)
    play([home, graze, home, pegs_down(AXIS, NEAR, FAR), out, rod()])
    assert not step(play.world, hook=T.success)  # a touch: 0.5, and the episode runs on
    play([home, pegs_down(AXIS, NEAR), out, rod()])
    assert not step(play.world, hook=T.success)

    # two episodes back to back: the memory clears when the clock goes back, and only then
    assert play([home, graze, home, graze, home, pegs_down(AXIS, NEAR, FAR), out, rod()]) == 0.25
    assert play([home, pegs_down(AXIS, NEAR, FAR), out, rod()], restart=False) == 0.25
    assert play([home, pegs_down(AXIS, NEAR, FAR), out, rod()]) == 1.0
    # every sum of values, halved up to three times, is a multiple of 0.0125
    values = np.array(list(T.VALUES.values()))
    for down in np.ndindex(2, 2, 2):
        for halvings in range(3):
            steps = values @ np.array(down) * 0.5**halvings / 0.0125
            assert abs(steps - round(steps)) < 1e-9


def _grip_yaw(rot) -> float:
    turn = rot @ grip_rotation(0.0).T
    return math.atan2(turn[1, 0], turn[0, 0])


def run_demo(variant) -> dict:
    plan = ArmPlan(T.demo(variant)["right"], BASE)
    walls = wall_boxes()
    rests = [(b.size, box(b.size, b.pos[:2], b.pos[2] - b.size[2] / 2)) for b in T.FIXED[3:]]
    least = {}
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    grasp, pose = None, (np.array(T.ROD.pos), 0.0)
    standing = dict.fromkeys(T.PEGS, True)
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
        if abs(finger) <= T.ROD_SIDE / 2 + 0.001:  # the pads are on the rod
            if grasp is None:
                grasp = (pose[0] - tcp, pose[1], yaw)
            offset, yaw0, grip0 = grasp
            c, s = math.cos(yaw - grip0), math.sin(yaw - grip0)
            pose = (tcp + np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ offset, yaw0 + yaw - grip0)
        elif grasp is not None:  # let go: the rod settles on what is under it
            grasp = None
            pose = (np.array([*pose[0][:2], T.REST_TOP + T.ROD_SIDE / 2]), pose[1])
        rod_state = rod(pose[0][:2], pose[0][2] - T.ROD_SIDE / 2, pose[1], moving=grasp is not None)["rod"]
        objects = {**START, "rod": rod_state}
        points = T.rod_points(rod_state)
        for name in T.PEGS:
            if standing[name] and T.box_gap(points, START[name], T.PEG_SIZE) < 0.001:
                standing[name] = False
            if not standing[name]:
                objects.update(peg(name, down=True))
        final = step(objects, arms)
        held_s = held_s + 1 if T.success(objects) else 0
        note("rod to a post", T.TRACKER.gaps[0], t, label)
        note("robot to a post", T.TRACKER.gaps[1], t, label)
        if n % EVERY == 0:
            robot = T.robot_points(arms)
            note("rod to the wall and lintel", min(T.box_gap(points, b, size) for size, b in walls), t, label)
            note("robot to the wall and lintel", min(T.box_gap(robot, b, size) for size, b in walls), t, label)
            note("robot to the blocks", min(T.box_gap(robot, b, size) for size, b in rests), t, label)
            note(f"robot behind the wall, height above {T.CAVITY_Z * 100:.0f} cm", float((robot[robot[:, 0] > T.CAVITY_X][:, 2] - T.TOP - T.CAVITY_Z).min()) if (robot[:, 0] > T.CAVITY_X).any() else T.NEAR, t, label)
    margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
    return {"plan": plan, "final": final, "held_s": held_s / HZ, "least": least, "margin": margin, "touches": T.TRACKER.touches, "tracker": T.TRACKER}


def check_demo():
    run = run_demo("scripted")
    plan, least, tracker = run["plan"], run["least"], run["tracker"]
    print(f"  ideal run: score {run['final']:.2f}, success held {run['held_s']:.1f} s, {run['touches']} touches, pegs down {sum(tracker.fallen.values())}")
    for what, (value, t, label) in least.items():
        shown = "over 50.0" if value >= T.NEAR else f"{value * MM:9.1f}"
        print(f"  {shown} mm  {what}, at {t:.1f} s ({label})")
    print(f"  worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg over {plan.duration:.1f} s")
    print(f"  smallest joint-limit margin per joint (deg): {np.round(np.degrees(run['margin']), 1)}")
    assert run["final"] == 1.0 and run["held_s"] >= T.SUCCESS_HOLD_S and run["touches"] == 0
    assert not tracker.cavity and not tracker.round_wall and not tracker.pushed
    assert least["rod to a post"][0] >= SAFE and least["robot to a post"][0] >= SAFE
    assert least["rod to the wall and lintel"][0] >= SAFE and least["robot to the wall and lintel"][0] >= 0.015  # the wrist over the wall top
    assert least["robot to the blocks"][0] >= SAFE
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (run["margin"] > 0).all()
    assert plan.duration + 3.0 < 2400 / HZ


def check_naive():
    run = run_demo("naive")
    print(f"  slid sideways instead of swung: {run['touches']} touches, pushed {run['tracker'].pushed}, score {run['final']:.2f}")
    assert run["touches"] >= 1


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
