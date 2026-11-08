"""The pivot_through_port.py scene, scoring and demo, with a line printed every 5 control steps: the demo's
waypoint, how far the right gripper's TCP (from its measured joints) is from where the demo wants it (mm), the
finger opening, where the rod's tip is, the smallest distances from the rod and from the robot to the posts, each
post's shift and tilt, what the tracker holds and the score. For tuning the demo in the simulator: run it in place
of pivot_through_port.py (-E task=tasks/pivot-through-port/trace.py) and filter the output with grep trace.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.demo import ArmPlan, grip_rotation

_spec = importlib.util.spec_from_file_location("pivot_through_port", Path(__file__).with_name("pivot_through_port.py"))
_task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task)

SCENE = _task.SCENE
OBJECTS = _task.OBJECTS
SCORE_OVER = _task.SCORE_OVER
SUCCESS_HOLD_S = _task.SUCCESS_HOLD_S
success = _task.success
demo = _task.demo

_BASE = station.BASES["right"]
_plan = ArmPlan(_task.demo()["right"], _BASE)
_steps = 0
_closest = [math.inf, math.inf]  # rod to a post, robot to a post


def score(objects):
    global _steps
    value = _task.score(objects)
    _steps += 1
    tracker = _task.TRACKER
    _closest[:] = np.minimum(_closest, tracker.gaps)
    if _steps % 5 == 0:
        joints = _task.arm_joints()["right"]
        tcp, _ = ik.tcp_pose(joints[:6], _BASE)
        want, rot, finger, label = _plan.target(_task.clock())
        turn = rot @ grip_rotation(0.0).T
        want_yaw = math.degrees(math.atan2(turn[1, 0], turn[0, 0]))
        off = (tcp - want) * 1e3
        rod = objects["rod"]
        axis = _task._rotation(rod.quat_xyzw)[:, 0]
        tip = rod.pos + axis * _task.ROD_SIZE[0] / 2 * (1 if axis[0] >= 0 else -1)
        rod_yaw = math.degrees(math.atan2(axis[1], axis[0]) if axis[0] >= 0 else math.atan2(-axis[1], -axis[0]))
        posts = "  ".join(
            f"{name[5]} shift {_task.post_moved(objects, name) * 1e3:4.1f} tilt {_task.tilt_deg(objects[name]):4.1f}"
            for name in _task.POSTS
        )
        pegs = "".join("D" if tracker.fallen[name] else "-" for name in _task.VALUES)
        flags = f"{'P' if tracker.pushed else '-'}{'C' if tracker.cavity else '-'}{'R' if tracker.round_wall else '-'}{'H' if tracker.held else '-'}"
        print(
            f"[trace] step {_steps:4d}  {label:22s} tcp off x {off[0]:5.1f} y {off[1]:5.1f} z {off[2]:5.1f}"
            f"  fingers {-2e3 * joints[6]:5.1f} (want {-2e3 * finger:5.1f})"
            f"  tip x {tip[0] * 100:5.1f} y {tip[1] * 100:5.1f} z {(tip[2] - _task.TOP) * 100:4.1f} yaw off {rod_yaw - want_yaw:5.1f}"
            f"  to posts: rod {tracker.gaps[0] * 1e3:4.1f} (closest {_closest[0] * 1e3:4.1f})"
            f" robot {tracker.gaps[1] * 1e3:4.1f} (closest {_closest[1] * 1e3:4.1f})"
            f"  {posts}  touches {tracker.touches} pegs {pegs} flags {flags} parked {_task.parked(objects)}  score {value:.4f}",
            flush=True,
        )
    return value
