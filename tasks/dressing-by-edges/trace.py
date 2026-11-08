"""The dressing_by_edges.py scene, scoring and demo, with a line printed every 5 control steps: the demo's
waypoint, how far the right gripper's TCP (from its measured joints) is from where the demo wants it (mm), the
finger opening, the dressing's centre relative to the wound's and its height above the skin (mm), its tilt, the
robot's distance to the key part, what the tracker holds, the coverage and the score. For tuning the demo in the
simulator: run it in place of dressing_by_edges.py (-E task=tasks/dressing-by-edges/trace.py) and filter the
output with grep trace.
"""

import importlib.util
from pathlib import Path

from yam_arena import ik, station
from yam_arena.demo import ArmPlan

_spec = importlib.util.spec_from_file_location("dressing_by_edges", Path(__file__).with_name("dressing_by_edges.py"))
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
_least_gap = [float("inf")]


def score(objects):
    global _steps
    value = _task.score(objects)
    _steps += 1
    tracker = _task.TRACKER
    _least_gap[0] = min(_least_gap[0], tracker.key_gap)
    if _steps % 5 == 0:
        joints = _task.arm_joints()["right"]
        tcp, _ = ik.tcp_pose(joints[:6], _BASE)
        want, _, finger, label = _plan.target(_task.clock())
        off = (tcp - want) * 1e3
        d = objects["dressing"]
        dx, dy = (d.pos[:2] - _task.WOUND_CENTRE) * 1e3
        print(
            f"[trace] step {_steps:4d}  {label:26s} tcp off x {off[0]:5.1f} y {off[1]:5.1f} z {off[2]:5.1f}"
            f"  fingers {-2e3 * joints[6]:5.1f} (want {-2e3 * finger:5.1f})"
            f"  dressing dx {dx:6.1f} dy {dy:6.1f} up {(d.low[2] - _task.SKIN_TOP) * 1e3:6.1f} tilt {_task.tilt_deg(d):4.1f}"
            f"  key gap {tracker.key_gap * 1e3:5.1f} (least {_least_gap[0] * 1e3:5.1f}) touches {tracker.touches}"
            f"  on skin {_task.on_skin(d)} slid {tracker.slid * 1e3:4.1f} dragged {tracker.dragged} held {tracker.held}"
            f"  cover {_task.coverage(d):.1f}  score {value:.4f}",
            flush=True,
        )
    return value
