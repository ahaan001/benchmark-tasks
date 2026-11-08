"""The port_instrument_swap.py scene, scoring and demo, with a line printed every 5 control steps: the demo's
waypoint, how far the right gripper's TCP (from its measured joints) is from where the demo wants it (mm), the
finger opening, each rod's position, the smallest distances from the rods and from the robot to the flanges, each
flange's shift and tilt, what the tracker holds and the score. For tuning the demo in the simulator: run it in
place of port_instrument_swap.py (-E task=tasks/port-instrument-swap/trace.py) and filter the output with grep trace.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.demo import ArmPlan

_spec = importlib.util.spec_from_file_location("port_instrument_swap", Path(__file__).with_name("port_instrument_swap.py"))
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
_closest = [math.inf, math.inf]  # a rod to a flange, the robot to a flange


def _rod(objects, name) -> str:
    rod = objects[name]
    return f"x {rod.pos[0] * 100:5.1f} y {rod.pos[1] * 100:5.1f} base {(rod.low[2] - _task.SKIN_TOP) * 100:5.1f} tilt {_task.tilt_deg(rod):4.1f}"


def score(objects):
    global _steps
    value = _task.score(objects)
    _steps += 1
    tracker = _task.TRACKER
    _closest[:] = np.minimum(_closest, tracker.gaps)
    if _steps % 5 == 0:
        joints = _task.arm_joints()["right"]
        tcp, _ = ik.tcp_pose(joints[:6], _BASE)
        want, _, finger, label = _plan.target(_task.clock())
        off = (tcp - want) * 1e3
        posts = "  ".join(
            f"{name[-1]} shift {_task.post_moved(objects, name) * 1e3:4.1f} tilt {_task.tilt_deg(objects[name]):4.1f}" for name in _task.POSTS
        )
        where = f"grey {'T' if _task.in_tray(objects['grey_inst']) else '-'}{'H' if tracker.held['grey_inst'] else '-'} blue {'S' if _task.seated(objects['blue_inst']) else 'P' if _task.in_port(objects['blue_inst']) else '-'}{'H' if tracker.held['blue_inst'] else '-'}"
        print(
            f"[trace] step {_steps:4d}  {label:18s} tcp off x {off[0]:5.1f} y {off[1]:5.1f} z {off[2]:5.1f}"
            f"  fingers {-2e3 * joints[6]:5.1f} (want {-2e3 * finger:5.1f})"
            f"  grey {_rod(objects, 'grey_inst')}  blue {_rod(objects, 'blue_inst')}"
            f"  to flanges: rods {tracker.gaps[0] * 1e3:4.1f} (closest {_closest[0] * 1e3:4.1f})"
            f" robot {tracker.gaps[1] * 1e3:4.1f} (closest {_closest[1] * 1e3:4.1f})"
            f"  {posts}  touches {tracker.touches} {'PUSHED' if tracker.pushed else ''} {where}  score {value:.4f}",
            flush=True,
        )
    return value
