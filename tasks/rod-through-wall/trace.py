"""The rod_through_wall.py scene, scoring and demo, with a line printed every 10 control steps: the rod's centre
in the wall frame (u along the wall, v out of its near face, height above the table, in mm), its tilt from level,
its clearance from the wall and what the tracker holds. For tuning the demo in the simulator: run it in place of
rod_through_wall.py (-E task=tasks/rod-through-wall/trace.py) and filter the output with grep trace.
"""

import importlib.util
import math
from pathlib import Path

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


def score(objects):
    global _steps
    value = _task.score(objects)
    _steps = 0 if _task.at_start(objects) else _steps + 1
    if _steps and _steps % 10 == 0:
        rod, tracker = objects["rod"], _task.TRACKER
        u, v = _task.to_wall(rod.pos[:2]) * 1000
        tilt = math.degrees(math.asin(min(abs(_task._rotation(rod.quat_xyzw)[2, 1]), 1.0)))
        print(
            f"[trace] step {_steps:4d}  u {u:7.1f}  v {v:7.1f}  h {(rod.pos[2] - _task.TOP) * 1000:6.1f}"
            f"  tilt {tilt:4.1f} deg  clearance {_task.wall_clearance(objects) * 1000:6.1f} mm"
            f"  touched {tracker.touched}  fell {tracker.fell}  window {tracker.window}  score {value:.2f}",
            flush=True,
        )
    return value
