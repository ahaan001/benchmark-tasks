"""Offline check of the room around the arms in the demo of rod_through_wall.py (no simulator). Run from the
repository root with yam_arena importable:

    python tasks/rod-through-wall/arm_clearance.py

It plays both arms' plans through the IK and places the URDF's collision meshes at every fourth control step. It
prints the smallest distance from each arm to the wall, to the stand and to its own base, and between the two
wrists. The wall height (16 cm) and the hand-off distance (8.5 cm) come from this check: in a straight-down grip
the wrist link reaches 9.8 cm ahead of the TCP, from 14 cm above it.
"""

import importlib.util
from pathlib import Path

import numpy as np

from yam_arena import station
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import ArmPlan

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("rod_through_wall.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ, EVERY = 20, 4
MIN_CLEARANCE = 0.02
WRIST = ("link4", "link5", *T.GRIPPER_LINKS)
BASE_HALF = np.array([0.036, 0.101, 0.074])  # the fixed base link's box: half x, half y, height


def to_box(points, low, high) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0.0), axis=1).min())


def between(a, b) -> float:
    return min(float(np.sqrt(((a[i : i + 400, None] - b[None]) ** 2).sum(-1)).min()) for i in range(0, len(a), 400))


def main():
    plans = {arm: ArmPlan(waypoints, station.BASES[arm]) for arm, waypoints in T.demo().items()}
    stand_low = np.array([*(np.array(T.STAND_UV) - np.array(T.STAND_SIZE[:2]) / 2), 0.0])
    stand_high = np.array([*(np.array(T.STAND_UV) + np.array(T.STAND_SIZE[:2]) / 2), T.STAND_SIZE[2]])
    worst = {}

    def note(what, value, t, label):
        if value < worst.get(what, (np.inf,))[0]:
            worst[what] = (value, t, label)

    end = max(plan.duration for plan in plans.values())
    for step, t in enumerate(np.arange(0.0, end + 1 / HZ, 1 / HZ)):
        actions = {arm: plan.action7(t) for arm, plan in plans.items()}
        if step % EVERY:
            continue
        joints = {arm: np.r_[a[:6], a[6] * FINGER_OPEN] for arm, (a, _) in actions.items()}
        wrists = {}
        for arm, q in joints.items():
            label, base = actions[arm][1], np.array(station.BASES[arm])
            whole = T.in_wall_frame(T.arm_points(q, arm))
            note(f"{arm} arm to the wall", T.to_wall_parts(whole), t, label)
            note(f"{arm} arm to the stand", to_box(whole, stand_low, stand_high), t, label)
            hand = T.arm_points(q, arm, T.GRIPPER_LINKS)
            note(f"{arm} gripper to its base", to_box(hand, base - (*BASE_HALF[:2], 0.0), base + BASE_HALF), t, label)
            wrists[arm] = T.arm_points(q, arm, WRIST)[::3]
        if np.linalg.norm(wrists["left"].mean(0) - wrists["right"].mean(0)) < 0.35:
            labels = " / ".join(label for _, label in actions.values())
            note("left wrist to right wrist", between(wrists["left"], wrists["right"]), t, labels)
    for what, (value, t, label) in worst.items():
        print(f"  {value * 1000:6.1f} mm  {what}, at {t:.1f} s ({label})")
        assert value > MIN_CLEARANCE, f"{what}: {value * 1000:.1f} mm"
    print("ok")


if __name__ == "__main__":
    main()
