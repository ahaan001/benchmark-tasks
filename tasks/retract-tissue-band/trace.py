"""The retract_tissue_band.py scene, scoring and demo, with a line printed every 5 control steps: the demo's
waypoint, how far the right gripper's TCP (from its measured joints) is from where the demo wants it (mm), the
finger opening, the band's near and far end and what is left beyond the bridge (cm), the band's turn and height
above the strips, what the tracker holds, each fragment's height above the pad with its share of its value, whether
the band counts as laid back, and the score. For tuning the demo in the simulator: run it in place of
retract_tissue_band.py (-E task=tasks/retract-tissue-band/trace.py) and filter the output with grep trace.
"""

import importlib.util
from pathlib import Path

from yam_arena import ik, station
from yam_arena.demo import ArmPlan

_spec = importlib.util.spec_from_file_location(
    "retract_tissue_band", Path(__file__).with_name("retract_tissue_band.py")
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
_least_slack = [_task.SLACK]


def score(objects):
    global _steps
    value = _task.score(objects)
    _steps += 1
    tracker = _task.TRACKER
    _least_slack[0] = min(_least_slack[0], tracker.slack)
    if _steps % 5 == 0:
        joints = _task.arm_joints()["right"]
        tcp, _ = ik.tcp_pose(joints[:6], _BASE)
        want, _, finger, label = _plan.target(_task.clock())
        off = (tcp - want) * 1e3
        band = objects["band"]
        near, far = _task.band_ends(band)
        fragments = "  ".join(
            f"{name.split('_')[1][0]} h {(objects[name].low[2] - _task.PAD_TOP) * 1e3:5.1f}"
            f" {'L' if tracker.lifted[name] else '-'}{'D' if tracker.dragged[name] else '-'}"
            f"{'H' if tracker.held[name] else '-'} {_task.share(objects, name):.1f}"
            for name in _task.VALUES
        )
        print(
            f"[trace] step {_steps:4d}  {label:24s} tcp off x {off[0]:5.1f} y {off[1]:5.1f} z {off[2]:5.1f}"
            f"  fingers {-2e3 * joints[6]:5.1f} (want {-2e3 * finger:5.1f})"
            f"  band x {near * 100:5.1f}..{far * 100:5.1f} slack {tracker.slack * 100:4.1f} (least {_least_slack[0] * 100:4.1f})"
            f" turn {_task.band_askew(band):4.1f} up {(band.low[2] - _task.STRIP_TOP) * 1e3:4.1f}"
            f"  out {tracker.out} warned {tracker.warned} held {tracker.band_held} back {_task.band_back(band)}"
            f"  {fragments}  score {value:.4f}",
            flush=True,
        )
    return value
