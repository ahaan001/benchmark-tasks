"""The retract_tissue_band.py scene and scoring with an empty plan: with --policy yam_scripted -P hold_s=120, both
arms hold their READY pose for the run limit and do nothing. This records the zero-action video that stands in for
the Dream-1 run when the Dream API refuses the instruction (DA-4000, unsafe instruction) and sends no actions.
"""

import importlib.util
from pathlib import Path

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
score = _task.score


def demo():
    return {}
