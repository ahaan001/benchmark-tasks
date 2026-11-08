"""Offline check of port_instrument_swap.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/port-instrument-swap/check.py

1. Design: the room a rod has to the flanges on the hole's axis and off it, that the hole takes one rod, the
   gripper's room to the flanges at the grasp, and what the top camera sees past the arms at their ready pose.
2. Scores: hand-built trials (the rods, the flanges and the arms step by step, on a stand-in for the episode
   clock) against the score each must get.
3. Demo: plays the right arm's plan through the IK at 20 Hz with the held rod following the gripper, scores it,
   and prints the smallest distances from the rods and the robot to the flanges, the worst IK error and the
   joint limits.
4. Angled variant: the same with the strokes off the hole's axis, counting the touches.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.ik import quat_xyzw_to_matrix
from yam_arena.task_file import ObjectState

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("port_instrument_swap.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ = 20
MM = 1e3
SAFE_ROD = T.TOUCH + 0.0015  # the room a rod must keep to the flanges in the demo: the touch rule plus the IK error
SAFE_ROBOT = 0.015  # and the robot
PARKED = dict.fromkeys(("left", "right"), np.r_[READY, FINGER_OPEN])
BASE = station.BASES["right"]
GREY, BLUE, A, B = "grey_inst", "blue_inst", "post_a", "post_b"

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


def rod(name, xy, under, tilt_deg=0.0, moving=False) -> dict:
    return {name: box(T.ROD_SIZE, xy, under, tilt=math.radians(tilt_deg), moving=moving)}


def post(name, dy=0.0, dx=0.0, tilt_deg=0.0, moving=False) -> dict:
    x, y = T.POSTS[name].pos[:2]
    return {name: box(T.POST_SIZE, (x + dx, y + dy), T.SKIN_TOP, tilt=math.radians(tilt_deg), moving=moving)}


def grey(where, **kwargs) -> dict:
    """The grey rod: "port" (seated), "tray" (lying flat on the tray's floor), "skin" (standing on the slab),
    "lifted" (held 3 cm up the hole), "out" (straight above the hole, clear of the flanges)."""
    places = {
        "port": (T.PORT, T.WELL_FLOOR, 0.0),
        "tray": (T.TRAY_XY, T.TRAY_FLOOR, 90.0),
        "skin": ((T.PORT[0] + 0.08, T.PORT[1]), T.SKIN_TOP, 0.0),
        "lifted": (T.PORT, T.WELL_FLOOR + 0.03, 0.0),
        "out": (T.PORT, T.SKIN_TOP + T.POST_SIZE[2] + 0.01, 0.0),
    }
    xy, under, tilt = places[where]
    return rod(GREY, xy, under, tilt, **kwargs)


def blue(where, **kwargs) -> dict:
    """The blue rod: "stand", "port" (seated), "short" (1 cm above the hole's floor), "leaning" (in the hole,
    tilted 10 degrees), "on_grey" (standing on the seated grey rod), "over" (above the hole, clear of the
    flanges), "skin"."""
    places = {
        "stand": (T.STAND_XY, T.STAND_FLOOR, 0.0),
        "port": (T.PORT, T.WELL_FLOOR, 0.0),
        "short": (T.PORT, T.WELL_FLOOR + 0.01, 0.0),
        "leaning": (T.PORT, T.WELL_FLOOR, 10.0),
        "on_grey": (T.PORT, T.WELL_FLOOR + T.ROD_LEN, 0.0),
        "over": (T.PORT, T.SKIN_TOP + T.POST_SIZE[2] + 0.01, 0.0),
        "skin": ((T.PORT[0] - 0.08, T.PORT[1]), T.SKIN_TOP, 0.0),
    }
    xy, under, tilt = places[where]
    return rod(BLUE, xy, under, tilt, **kwargs)


START = {**grey("port"), **blue("stand"), **post(A), **post(B)}
DONE = [grey("lifted"), grey("out"), grey("tray"), blue("over"), blue("short"), blue("port")]


def gripper_at(xyz, yaw=0.0, finger=FINGER_OPEN) -> dict:
    """The arms with the right gripper's TCP at a point, pointing down, and the left arm parked."""
    q, miss, _ = ik.solve(np.array(xyz), ik.matrix_to_quat_xyzw(grip_rotation(yaw)), READY, BASE)
    assert miss < 1e-3, f"the right gripper cannot reach {xyz} at yaw {math.degrees(yaw):.0f}"
    return {**PARKED, "right": np.r_[q, finger]}


def holding(name, xyz) -> dict:
    """The right gripper closed on the rod whose top is TCP_ABOVE_BASE - ROD_LEN below the TCP at xyz."""
    return {"arms": gripper_at(xyz, 0.0, T.GRIP)}


def fingers_before_post(gap) -> dict:
    """The right gripper, 2 cm open, beside flange A at its mid-height with a fingertip `gap` from its outer face,
    and the rods at their start."""

    def arms(out):
        y = T.POST_Y[A] - T.POST_SIZE[1] / 2 - out
        return gripper_at((T.PORT[0], y, T.SKIN_TOP + T.POST_SIZE[2] / 2 + 0.013), 0.0, -0.01)

    near, far = 0.0, 0.05
    for _ in range(16):
        mid = (near + far) / 2
        near, far = (mid, far) if T.post_gap(T.robot_points(arms(mid)), START) < gap else (near, mid)
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
    gaps = {}
    for dy in (0.0, 0.002, 0.003, 0.004):
        gaps[dy] = min(
            T.post_gap(T.rod_points(rod(GREY, (T.PORT[0], T.PORT[1] + dy), under)[GREY]), START)
            for under in np.linspace(T.WELL_FLOOR, T.SKIN_TOP + T.POST_SIZE[2] + 0.005, 30)
        )
        print(f"  rod {dy * MM:.0f} mm off the hole's axis: {gaps[dy] * MM:.1f} mm from the nearer flange")
    assert gaps[0.0] >= SAFE_ROD and gaps[0.0] > T.TOUCH_CLEAR, "a rod on the axis must be clear of the flanges"
    assert gaps[0.002] >= T.TOUCH, "2 mm off the axis is not a touch"
    assert gaps[0.003] < T.TOUCH, "3 mm off the axis is a touch"
    assert gaps[0.004] < 1e-6, "4 mm off the axis hits the flange"
    assert 2 * T.ROD_SIDE > T.WELL_W, "the hole must take one rod only"
    assert T.POST_Y[B] - T.POST_SIZE[1] / 2 == T.PORT[1] + T.WELL_W / 2, "the flanges' inner faces are flush with the hole's walls"
    for finger in (T.GRIP, FINGER_OPEN):
        arms = gripper_at((*T.PORT, T.GRASP_Z), 0.0, finger)
        gap = T.post_gap(T.robot_points(arms), START)
        print(f"  gripper at the grasp, fingers {'closed on a rod' if finger == T.GRIP else 'open'}: {gap * MM:.1f} mm from the flanges")
        assert gap >= SAFE_ROBOT
    assert T.post_gap(T.robot_points(PARKED), START) >= T.NEAR, "the arms at their ready pose are nowhere near the flanges"
    assert T.CARRY_Z - T.TCP_ABOVE_BASE > T.SKIN_TOP + T.POST_SIZE[2], "a carried rod clears the flange tops"


def check_view():
    """The top camera (320 x 176) sees the port, both rods, the stand and the tray past the arms at their ready
    pose."""
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
    points = {
        "flange A": (T.PORT[0], T.POST_Y[A], T.SKIN_TOP + T.POST_SIZE[2]),
        "flange B": (T.PORT[0], T.POST_Y[B], T.SKIN_TOP + T.POST_SIZE[2]),
        "grey rod top": (*T.PORT, T.WELL_FLOOR + T.ROD_LEN),
        "blue rod top": (*T.STAND_XY, T.STAND_FLOOR + T.ROD_LEN),
        "stand": (*T.STAND_XY, T.SKIN_TOP),
        "scope top": (*T.SCOPE_XY, T.SKIN_TOP + T.SCOPE_LEN),
        **{
            f"tray corner {i}": (T.TRAY_XY[0] + sx * T.TRAY_INNER[0] / 2, T.TRAY_XY[1] + sy * T.TRAY_INNER[1] / 2, T.TRAY_FLOOR)
            for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        },
    }
    for label, point in points.items():
        u, v, d = project([point])[0]
        inside = 0 <= u < 320 and 0 <= v < 176
        behind = any(nearest.get((round(u) + du, round(v) + dv), np.inf) < d for du in (-1, 0, 1) for dv in (-1, 0, 1))
        print(f"  {label:14s} at pixel ({u:5.1f}, {v:5.1f}){'' if inside else ', OUT OF FRAME'}{', BEHIND AN ARM' if behind else ''}")
        assert inside and not behind, label
    a, b = project([(*T.PORT, T.SKIN_TOP), (T.PORT[0] + 0.01, T.PORT[1] + 0.01, T.SKIN_TOP)])
    print(f"  at the port, 1 cm along x is {abs(b[1] - a[1]):.1f} px and 1 cm along y is {abs(b[0] - a[0]):.1f} px")


def check_scores():
    # the grey rod drawn up the hole 2.5 mm off the axis: a touch; and the right gripper holding it there
    graze = rod(GREY, (T.PORT[0], T.PORT[1] + 0.0025), T.WELL_FLOOR + 0.02)
    assert T.post_gap(T.rod_points(graze[GREY]), START) < T.TOUCH
    held_blue = {**blue("port"), **holding(BLUE, (*T.PORT, T.GRASP_Z))}
    held_grey = {**rod(GREY, T.TRAY_XY, T.TRAY_FLOOR + 0.01), **holding(GREY, (*T.TRAY_XY, T.DROP_Z))}  # standing over the tray's floor, in the pads
    cases = {
        "start, nothing moved": ([], 0.0),
        "grey in the tray, blue seated": (DONE, 1.0),
        "grey in the tray, blue still in the stand": (DONE[:3], 0.3),
        "grey in the tray, blue in the port 1 cm short of the floor": (DONE[:5], 0.65),
        "grey in the tray, blue leaning in the port": ([*DONE[:4], blue("leaning")], 0.65),
        "grey in the tray, blue seated but still moving": ([*DONE[:5], blue("port", moving=True)], 0.65),
        "grey laid on the skin, blue seated": ([grey("lifted"), grey("out"), grey("skin"), blue("over"), blue("port")], 0.7),
        "grey in the tray but still moving, blue seated": ([*DONE, grey("tray", moving=True)], 0.7),
        "grey left in the port, blue in the stand": ([], 0.0),
        "grey left in the port, blue standing on top of it": ([blue("over"), blue("on_grey")], 0.0),
        "grey on the skin, blue on the skin": ([grey("lifted"), grey("out"), grey("skin"), blue("over"), blue("skin")], 0.0),
        "done, one touch": ([graze, *DONE], 0.5),
        "done, the rod lingers 1 to 3 mm from a flange (still one touch)": ([graze, rod(GREY, (T.PORT[0], T.PORT[1] + 0.002), T.WELL_FLOOR + 0.025), graze, *DONE], 0.5),
        "done, two touches": ([graze, grey("lifted"), graze, *DONE], 0.25),
        "done, three touches": ([graze, grey("lifted"), graze, grey("lifted"), graze, *DONE], 0.0),
        "blue in the port but still held at the end": ([*DONE[:4], held_blue], 0.3),
        "grey in the tray but still held, blue seated": ([*DONE, held_grey], 0.7),
        "flange nudged 5 mm and left there (one touch)": ([post(A, dy=-0.005, moving=True), post(A, dy=-0.005), *DONE], 0.5),
        "flange tilted 5 degrees and stood back (one touch)": ([post(B, tilt_deg=5.0, moving=True), post(B), *DONE], 0.5),
        "flange pushed 2 cm and stood back on its mark before the end": ([post(A, dy=-0.02, moving=True), post(A), *DONE], 0.0),
        "flange tilted 20 degrees and stood back": ([post(B, tilt_deg=20.0, moving=True), post(B), *DONE], 0.0),
        "flange knocked over": ([post(A, tilt_deg=90.0, dy=-0.03), *DONE], 0.0),
        "flange lifted off and put back": ([post(B, dx=0.04), post(B), *DONE], 0.0),
        "fingertip 1 mm from a flange with no rod involved (one touch)": ([fingers_before_post(0.001), *DONE], 0.5),
        "fingertip 4 mm from a flange (no touch)": ([fingers_before_post(0.004), *DONE], 1.0),
        "one touch, everything back at its start, then done": ([graze, dict(START), *DONE], 0.5),
    }
    for name, (moves, want) in cases.items():
        got = play(moves)
        print(f"  {got:.4f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"

    # success is the 1.0 state only, and holds while it lasts
    play(DONE)
    assert all(step(play.world, hook=T.success) for _ in range(HZ))
    assert not step({**play.world, **blue("port", moving=True)}, hook=T.success)
    assert not step({**play.world, **grey("tray", moving=True)}, hook=T.success)
    assert not step({**play.world, **post(A, moving=True)}, hook=T.success)
    assert not step({**play.world, **post(B, dy=0.004)}, hook=T.success)
    assert not step(play.world, held_blue["arms"], hook=T.success)
    play([graze, *DONE])
    assert not step(play.world, hook=T.success)  # a touch: 0.5, and the episode runs on
    play(DONE[:5])
    assert not step(play.world, hook=T.success)

    # two episodes back to back: the memory clears when the clock goes back, and only then
    assert play([graze, grey("lifted"), graze, *DONE]) == 0.25
    assert play(DONE, restart=False) == 0.25
    assert play(DONE) == 1.0
    # the possible scores
    sums = {0.0, T.GREY_VALUE, T.BLUE_VALUE * T.BLUE_SHORT, T.GREY_VALUE + T.BLUE_VALUE * T.BLUE_SHORT, T.BLUE_VALUE, T.GREY_VALUE + T.BLUE_VALUE}
    scores = sorted({round(s * T.TOUCH_FACTOR**k, 4) for s in sums for k in range(T.MAX_TOUCHES)})
    print(f"  possible scores: {scores}")


def run_demo(variant) -> dict:
    plan = ArmPlan(T.demo(variant)["right"], BASE)
    least = {}
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    world = dict(START)
    held, offset = None, None
    held_s, final = 0, 0.0
    NOW["t"] = 0.0

    def note(what, value, t, label):
        if value < least.get(what, (np.inf,))[0]:
            least[what] = (value, t, label)

    for t in np.arange(1 / HZ, plan.duration + 3.0, 1 / HZ):
        action, label = plan.action7(t)
        arms = {**PARKED, "right": np.r_[action[:6], action[6] * FINGER_OPEN]}
        lo, hi = np.minimum(lo, action[:6]), np.maximum(hi, action[:6])
        tcp, _, finger, _ = plan.target(t)
        closed = abs(finger) <= T.ROD_SIDE / 2 + 0.001
        if closed and held is None:  # the pads close on the rod whose top is under the TCP
            for name in (GREY, BLUE):
                top = world[name].pos + np.array([0.0, 0.0, T.ROD_LEN / 2])
                if np.linalg.norm(top[:2] - tcp[:2]) < 0.01 and abs(top[2] - (tcp - T.TCP_ABOVE_BASE + T.ROD_LEN)[2]) < 0.01:
                    held, offset = name, world[name].pos - tcp
        if held is not None and closed:
            centre = tcp + offset
            world.update(rod(held, centre[:2], centre[2] - T.ROD_LEN / 2, moving=True))
        elif held is not None:  # let go: the grey rod falls over in the tray, the blue rod drops onto what is under it
            centre = world[held].pos
            if held == GREY:
                world.update(rod(GREY, centre[:2], T.TRAY_FLOOR, 90.0))
            else:
                under = T.WELL_FLOOR if T.over_well(world[BLUE]) else T.SKIN_TOP
                world.update(rod(BLUE, centre[:2], under))
            held = None
        final = step(world, arms)
        held_s = held_s + 1 if T.success(world) else 0
        note("a rod to a flange", T.TRACKER.gaps[0], t, label)
        note("the robot to a flange", T.TRACKER.gaps[1], t, label)
    margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
    return {"plan": plan, "final": final, "held_s": held_s / HZ, "least": least, "margin": margin, "touches": T.TRACKER.touches, "world": world}


def check_demo():
    run = run_demo("scripted")
    plan, least, world = run["plan"], run["least"], run["world"]
    print(f"  ideal run: score {run['final']:.2f}, success held {run['held_s']:.1f} s, {run['touches']} touches")
    for what, (value, t, label) in least.items():
        shown = "over 50.0" if value >= T.NEAR else f"{value * MM:9.1f}"
        print(f"  {shown} mm  {what}, at {t:.1f} s ({label})")
    print(f"  worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg over {plan.duration:.1f} s")
    print(f"  smallest joint-limit margin per joint (deg): {np.round(np.degrees(run['margin']), 1)}")
    assert run["final"] == 1.0 and run["held_s"] >= T.SUCCESS_HOLD_S and run["touches"] == 0
    assert T.in_tray(world[GREY]) and T.seated(world[BLUE])
    assert least["a rod to a flange"][0] >= SAFE_ROD and least["the robot to a flange"][0] >= SAFE_ROBOT
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (run["margin"] > 0).all()
    assert plan.duration + 3.0 < 1200 / HZ


def check_angled():
    run = run_demo("angled")
    print(f"  strokes off the hole's axis: {run['touches']} touches, score {run['final']:.2f}")
    assert run["touches"] >= 1


if __name__ == "__main__":
    print("design:")
    check_design()
    print("view:")
    check_view()
    print("scores:")
    check_scores()
    print("demo:")
    check_demo()
    print("angled variant:")
    check_angled()
    print("ok")
