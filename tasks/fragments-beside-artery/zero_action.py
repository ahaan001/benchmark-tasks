"""The fragments_beside_artery.py scene and scoring with an empty plan: with --policy yam_scripted -P hold_s=120,
both arms hold their READY pose for the run limit and do nothing. This records the zero-action video that stands in
for the Dream-1 run, because the Dream API refused the instruction (DA-4000, unsafe instruction) and sent no actions.
"""

import importlib.util
from pathlib import Path

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
score = _task.score


def demo():
    return {}
