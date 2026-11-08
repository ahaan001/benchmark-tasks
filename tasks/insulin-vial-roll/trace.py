"""The insulin_vial_roll.py scene, scoring and demo, with a line printed every 5 control steps: the demo's
waypoint, how far the right gripper's TCP (from its measured joints) is from where the demo wants it (mm), the
finger opening, the cloudy vial's turn this step and in all (degrees and turns), its base height above the mat,
tilt and offset from the square's centre, the clear vial's shift and turn, and what the tracker holds. For tuning
the demo in the simulator: run it in place of insulin_vial_roll.py (-E task=tasks/insulin-vial-roll/trace.py)
and filter the output with grep trace.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.demo import ArmPlan

_spec = importlib.util.spec_from_file_location("insulin_vial_roll", Path(__file__).with_name("insulin_vial_roll.py"))
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
_last = {"yaw": None, "rot": 0.0}


def score(objects):
    global _steps
    value = _task.score(objects)
    _steps += 1
    tracker, vial, clear = _task.TRACKER, objects["cloudy_vial"], objects["clear_vial"]
    yaw = _task.yaw_of(vial)
    step_turn = 0.0 if _last["yaw"] is None else math.degrees(_task.wrap(yaw - _last["yaw"]))
    _last["yaw"] = yaw
    if _steps % 5 == 0:
        joints = _task.arm_joints()["right"]
        tcp, _ = ik.tcp_pose(joints[:6], _BASE)
        want, _, finger, label = _plan.target(_task.clock())
        off = (tcp - want) * 1e3
        offset = (vial.pos[:2] - _task.SQUARE_CENTRE) * 1e3
        shift = np.linalg.norm(clear.pos - _task.CLEAR_VIAL.pos) * 1e3
        turned = 0.0 if tracker.clear_yaw is None else math.degrees(_task.wrap(_task.yaw_of(clear) - tracker.clear_yaw))
        print(
            f"[trace] step {_steps:4d}  {label:14s} tcp off x {off[0]:5.1f} y {off[1]:5.1f} z {off[2]:5.1f}"
            f"  fingers {-2e3 * joints[6]:5.1f} (want {-2e3 * finger:5.1f})"
            f"  vial yaw {math.degrees(yaw):6.1f} step {step_turn:5.1f}"
            f" total {math.degrees(tracker.total_rot):7.1f} = {tracker.total_rot / (2 * math.pi):5.2f} turns"
            f"  rise {tracker.rise * 1e3:5.1f} tilt {_task.tilt_deg(vial):4.1f}"
            f" off x {offset[0]:5.1f} y {offset[1]:5.1f}"
            f"  clear shift {shift:4.1f} turn {turned:5.1f}"
            f"  held {tracker.held} spoiled {tracker.spoiled}  score {value:.2f}",
            flush=True,
        )
    return value
