"""The fragments_beside_artery.py scene, scoring and demo, with a line printed every 5 control steps: the demo's
waypoint, how far the right gripper's TCP (from its measured joints) is from where the demo wants it (mm), the
finger opening, the smallest distances from the robot and from the fragments to the artery, the artery's shift,
turn and rise, what the tracker holds, and each fragment's height above the table with its share of its value.
For tuning the demo in the simulator: run it in place of fragments_beside_artery.py
(-E task=tasks/fragments-beside-artery/trace.py) and filter the output with grep trace.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.demo import ArmPlan

_spec = importlib.util.spec_from_file_location(
    "fragments_beside_artery", Path(__file__).with_name("fragments_beside_artery.py")
)
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
_closest = [math.inf, math.inf]  # robot to artery, fragments to artery


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
        artery = objects["artery"]
        shift = np.linalg.norm(artery.pos[:2] - _task.ARTERY.pos[:2]) * 1e3
        turn = math.degrees(math.acos(min(abs(_task._rotation(artery.quat_xyzw)[0, 0]), 1.0)))
        rise = (artery.high[2] - _task.WOUND_TOP - _task.ARTERY_SIZE[2]) * 1e3
        fragments = "  ".join(
            f"{name[-1]} h {(objects[name].low[2] - _task.TOP) * 1e3:5.1f}"
            f" {'L' if tracker.lifted[name] else '-'}{'D' if tracker.dragged[name] else '-'}"
            f"{'H' if tracker.held[name] else '-'} {_task.share(objects, name):.1f}"
            for name in _task.VALUES
        )
        print(
            f"[trace] step {_steps:4d}  {label:18s} tcp off x {off[0]:5.1f} y {off[1]:5.1f} z {off[2]:5.1f}"
            f"  fingers {-2e3 * joints[6]:5.1f} (want {-2e3 * finger:5.1f})"
            f"  to artery: robot {tracker.gaps[0] * 1e3:4.1f} (closest {_closest[0] * 1e3:4.1f})"
            f" fragments {tracker.gaps[1] * 1e3:4.1f} (closest {_closest[1] * 1e3:4.1f})"
            f"  artery shift {shift:4.1f} turn {turn:4.1f} rise {rise:4.1f}"
            f"  touches {tracker.touches} injured {tracker.injured}  {fragments}  score {value:.4f}",
            flush=True,
        )
    return value
