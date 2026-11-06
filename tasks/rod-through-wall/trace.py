"""The rod_through_wall.py scene, scoring and demo, with a line printed every 5 control steps: the rod's centre
and its far end in the wall frame (u along the wall, v out of its near face, h above the table, in mm), how far it
is turned from square to the wall (yaw) and from level (tilt), the rod's and the robot's clearance from the wall,
and what the tracker holds. For tuning the demo in the simulator: run it in place of rod_through_wall.py
(-E task=tasks/rod-through-wall/trace.py) and filter the output with grep trace.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("rod_through_wall", Path(__file__).with_name("rod_through_wall.py"))
_task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task)

SCENE = _task.SCENE
OBJECTS = _task.OBJECTS
SUCCESS_HOLD_S = _task.SUCCESS_HOLD_S
success = _task.success
distance = _task.distance
demo = _task.demo

_steps = 0
_closest = math.inf


def score(objects):
    global _steps, _closest
    value = _task.score(objects)
    _steps += 1
    clearance = _task.wall_clearance(objects)
    _closest = min(_closest, clearance)
    if _steps % 5 == 0:
        rod, tracker = objects["rod"], _task.TRACKER
        axis = _task._rotation(rod.quat_xyzw)[:, 1]  # the rod's long axis, from its far end to its near end
        along_u, along_v = axis[:2] @ _task._AXES.T
        half = np.array([0.0, _task.ROD_LENGTH / 2, 0.0])
        ends = _task.rod_points(objects, np.array([-half, half]))
        tip = ends[ends[:, 1].argmin()] * 1e3
        centre = np.r_[_task.to_wall(rod.pos[:2]), rod.pos[2] - _task.TOP] * 1e3
        print(
            f"[trace] step {_steps:4d}  centre u {centre[0]:6.1f} v {centre[1]:6.1f} h {centre[2]:5.1f}"
            f"  far end u {tip[0]:6.1f} v {tip[1]:6.1f} h {tip[2]:5.1f}"
            f"  yaw {math.degrees(math.atan2(along_u, along_v)):5.1f}  tilt {math.degrees(math.asin(axis[2])):5.1f}"
            f"  clearance {clearance * 1e3:5.1f} (closest {_closest * 1e3:4.1f})"
            f" robot {_task.robot_clearance(_task.arm_joints()) * 1e3:4.1f}"
            f"  touched {tracker.touched}  fell {tracker.fell}  released {tracker.released}"
            f"  window {tracker.window}  score {value:.2f}",
            flush=True,
        )
    return value
