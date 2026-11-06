"""Offline check of rod_through_wall.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/rod-through-wall/check.py

It tests the scoring rules on hand-built trials: paths of the rod, with the arms parked or placed by IK, on a
stand-in for the episode clock. Then it plays the demo of both arms through the IK at 20 Hz with the rod
following the gripper that holds it, and prints the score, the rod's clearance in the window, the worst IK error,
the joint limits and each gripper's distance from the wall.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("rod_through_wall.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ = 20
WINDOW = {name: (sum(w["u"]) / 2, T.SILL + w["side"] / 2) for name, w in T.WINDOWS.items()}  # centre (u, height)
ON_STAND = (*T.STAND_UV, T.STAND_SIZE[2] + T.ROD_SIDE / 2)
ON_TABLE = T.ROD_SIDE / 2
START = (*T.ROD_START, ON_TABLE)
PARKED = {side: np.r_[READY, FINGER_OPEN] for side in ("left", "right")}

# the simulator's episode clock and arm joints, which the task file reads from the embodiment
NOW = {"t": 0.0, "arms": PARKED}
T.clock = lambda: NOW["t"]
T.arm_joints = lambda: NOW["arms"]


def rod(u, v, height, moving=False, vz=0.0, arms=PARKED) -> tuple:
    """A step of a trial: the rod square to the wall and level, its centre at (u, v) in the wall frame and height
    above the table, and the arms' joints."""
    q = np.array([0.0, 0.0, math.sin(T.WALL_YAW / 2), math.cos(T.WALL_YAW / 2)])
    pos = np.array([*T.to_world(u, v), T.TOP + height])
    corners = T.ROD_CORNERS @ T._rotation(q).T
    vel = np.array([0.1 if moving else 0.0, 0.0, vz])
    return {"rod": ObjectState(pos, q, vel, np.zeros(3), pos + corners.min(0), pos + corners.max(0))}, arms


def step(state, hook=T.score):
    objects, NOW["arms"] = state
    NOW["t"] += 1 / HZ
    return hook(objects)


def play(path) -> float:
    """The score at the end of a new trial (the clock goes back to 0) that takes the rod through the states."""
    NOW["t"] = 0.0
    return [step(state) for state in [rod(*START), *path]][-1]


def through(name, du=0.0, **kwargs) -> list:
    """Raise the rod in front of a window and pass it through, du off the window's centre line."""
    u, h = WINDOW[name]
    return [rod(u + du, v, h, **kwargs) for v in (0.20, 0.10, 0.01, -0.01, -0.10, -0.20)]


def gripper_at(side, u, v, height, finger=FINGER_OPEN) -> dict:
    """The arms with one gripper's TCP at a point of the wall frame, pointing down, and the other arm parked."""
    target = (*T.to_world(u, v), T.TOP + height)
    q, miss, _ = ik.solve(target, ik.matrix_to_quat_xyzw(grip_rotation(T.GRIP_YAW)), READY, station.BASES[side])
    assert miss < 1e-3, f"{side} gripper cannot reach {target}"
    return {**PARKED, side: np.r_[q, finger]}


def check_scores():
    u_s, h_s = WINDOW["small"]
    stand, table = rod(*ON_STAND), rod(u_s, -0.30, ON_TABLE)
    graze = T.WINDOWS["small"]["side"] / 2 - T.ROD_SIDE / 2 - 0.001  # 1 mm from the pillar
    over = [rod(u_s, v, T.WALL_H + 0.03) for v in (0.20, 0.01, -0.01, -0.20)]
    around = [rod(-T.WALL_LENGTH / 2 - 0.05, v, 0.05) for v in (0.20, 0.01, -0.01, -0.20)]
    fingers_on_wall = gripper_at("left", u_s, 0.02, h_s + 0.01)  # the left fingers reach through the wall's plane
    still_held = gripper_at("right", T.STAND_UV[0], T.STAND_UV[1], ON_STAND[2] + 0.007, T.GRIP)
    cases = {
        "start, on the table": ([], 0.0),
        "lifted in front of the small window": ([rod(u_s, 0.20, h_s)], 0.0),
        "small window, on the stand": ([*through("small"), stand], 1.0),
        "medium window, on the stand": ([*through("medium"), stand], 0.7),
        "large window, on the stand": ([*through("large"), stand], 0.4),
        "large window, set down on the table behind the wall": ([*through("large"), table], 0.1),
        "small window, held half way through": (through("small")[:4], 0.25),
        "small window, held over the stand, still moving": ([*through("small"), rod(*ON_STAND, moving=True)], 0.25),
        "small window, on the stand, gripper still on the rod": (
            [*through("small"), rod(*ON_STAND, arms=still_held)],
            0.25,
        ),
        "small window, 1 mm from the pillar, on the stand": ([*through("small", du=graze), stand], 0.0),
        "small window, dropped onto the stand": ([*through("small"), rod(*ON_STAND, vz=-0.6), stand], 0.0),
        "small window, pulled back out": ([*through("small"), *through("small")[::-1]], 0.0),
        "over the top, on the stand": ([*over, stand], 0.0),
        "round the near end, on the stand": ([*around, stand], 0.0),
        "round the end, back and through the small window": ([*around, *around[::-1], *through("small"), stand], 1.0),
        "large window, back and through the small window": (
            [*through("large"), *through("large")[::-1], *through("small"), stand],
            1.0,
        ),
        "small window, back and through the large window": (
            [*through("small"), *through("small")[::-1], *through("large"), stand],
            0.4,
        ),
        "touch, rod put back at its start, then the small window": (
            [*through("small", du=graze)[:3], rod(*START), *through("small"), stand],
            0.0,
        ),
        "drop, rod put back at its start, then the small window": (
            [rod(u_s, 0.20, h_s, vz=-0.6), rod(*START), *through("small"), stand],
            0.0,
        ),
        "robot's fingers on the wall, then the small window": (
            [rod(u_s, 0.20, h_s, arms=fingers_on_wall), *through("small"), stand],
            0.0,
        ),
    }
    for name, (path, want) in cases.items():
        got = play(path)
        print(f"  {got:.3f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"
    play([*through("small"), stand])
    assert step(stand, T.success) and not step(rod(*ON_STAND, moving=True), T.success)
    assert not step(rod(*ON_STAND, arms=still_held), T.success)
    play([*through("medium"), stand])
    assert not step(stand, T.success)  # only the small window ends the episode with 1.0


def check_demo():
    plans = {arm: ArmPlan(waypoints, station.BASES[arm]) for arm, waypoints in T.demo().items()}
    end = max(plan.duration for plan in plans.values())
    lo = {arm: np.full(6, np.inf) for arm in plans}
    hi = {arm: np.full(6, -np.inf) for arm in plans}
    side = {"left": 1.0, "right": -1.0}
    keep_out = dict.fromkeys(plans, np.inf)
    pos, offset, clearance, robot, held_s, prev, arms = np.array(T.ROD.pos), {}, np.inf, np.inf, 0, None, {}
    NOW["t"] = 0.0
    for t in np.arange(1 / HZ, end + 3.0, 1 / HZ):
        for arm, plan in plans.items():
            a, _ = plan.action7(t)
            arms[arm] = np.r_[a[:6], a[6] * FINGER_OPEN]
            lo[arm], hi[arm] = np.minimum(lo[arm], a[:6]), np.maximum(hi[arm], a[:6])
            tcp, _, finger, _ = plan.target(t)
            keep_out[arm] = min(keep_out[arm], side[arm] * T.to_wall(tcp[:2])[1])
            if abs(finger) <= T.ROD_SIDE / 2 + 0.001:  # the pads are on the rod
                offset.setdefault(arm, pos - tcp)
            else:
                offset.pop(arm, None)
        holder = next(iter(offset), None)
        if holder is not None:
            pos = np.asarray(plans[holder].target(t)[0]) + offset[holder]
        else:  # let go: the rod comes to rest on what is under it
            over_stand = np.all(np.abs(T.to_wall(pos[:2]) - T.STAND_UV) < np.array(T.STAND_SIZE[:2]) / 2)
            pos = np.array([*pos[:2], (T.STAND_TOP if over_stand else T.TOP) + T.ROD_SIDE / 2])
        hangs = {"left": T.LEFT_SAG, "right": T.RIGHT_TAKE_UP}.get(holder, 0.0) if pos[2] > T.ROD.pos[2] else 0.0
        seen = pos - (0.0, 0.0, hangs)  # in the simulator the rod hangs this far below the plan
        state = rod(*T.to_wall(seen[:2]), seen[2] - T.TOP, vz=0.0 if prev is None else (seen[2] - prev[2]) * HZ)[0]
        prev = seen
        final = step((state, dict(arms)))
        clearance, robot = min(clearance, T.wall_clearance(state)), min(robot, T.robot_clearance(arms))
        held_s = held_s + 1 if T.success(state) else 0
    print(f"  ideal run: score {final:.2f}, success held {held_s / HZ:.1f} s, window {T.TRACKER.window}")
    print(f"  smallest clearance from the wall: rod {clearance * 1000:.1f} mm, robot {robot * 1000:.1f} mm")
    assert final == 1.0 and held_s >= T.SUCCESS_HOLD_S * HZ and clearance > T.CONTACT and robot > T.ROBOT_CONTACT
    for arm, plan in plans.items():
        margin = np.minimum(lo[arm] - ik.LOWER, ik.UPPER - hi[arm])
        print(f"  {arm}: worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg over {plan.duration:.1f} s")
        print(f"  {arm}: smallest joint-limit margin per joint (deg): {np.round(np.degrees(margin), 1)}")
        print(f"  {arm}: TCP stays {keep_out[arm] * 100:.1f} cm or more from the wall, on its own side")
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (margin > 0).all()
        assert keep_out[arm] >= T.HANDOFF_V - 1e-6


if __name__ == "__main__":
    print("scores:")
    check_scores()
    print("demo:")
    check_demo()
    print("ok")
