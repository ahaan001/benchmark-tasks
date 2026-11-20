"""On the station table, on a felt cloth: one deck of 52 poker-size playing cards lying face down, squared, 20 cm in
front of the midpoint between the arms, and four card boxes taped on the felt. Deal four cards off the top of the
deck, face down, one into each box.

A card is 0.29 mm thick: it cannot be pinched off the pack. It is slid off the top by friction and steered across
the felt. The top LOOSE_N cards are separate bodies; the rest of the pack is one block.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
DEAL_MODE (environment variable) picks what the scripted demo does: full (default), one, light, edge, rest
(see deal_demo.py).
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cardkit  # noqa: E402  (hooks yam_arena on import)
import layout as L  # noqa: E402
import scoring  # noqa: E402
from cardkit import ON_RESET, Plate, quat_matrix  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Deal four cards from the deck, face down, one at a time: one card into each white box. "
               "Leave the rest of the deck where it is.")

# -- the scene ---------------------------------------------------------------------------------------------------------
FELT_TOP = TOP + L.FELT_T
BLOCK_T = (L.DECK_N - L.LOOSE_N) * L.CARD_T
CARDS = [f"card_{i:02d}" for i in range(1, L.LOOSE_N + 1)]  # card_01 is the top card


def _card_z(from_top: int) -> float:
    below = L.LOOSE_N - 1 - from_top  # loose cards under this one
    return FELT_TOP + L.CARD_GAP + BLOCK_T + (below + 1) * L.CARD_GAP + below * L.CARD_T + L.CARD_T / 2


DECK_TOP = _card_z(0) + L.CARD_T / 2


def _objects() -> list:
    fx, fy = L.FELT_X, L.FELT_Y
    out = [Plate("felt", pos=((fx[0] + fx[1]) / 2, (fy[0] + fy[1]) / 2, TOP + L.FELT_T / 2),
                 size=(fx[1] - fx[0], fy[1] - fy[0], L.FELT_T), color=L.FELT_GREEN, friction=L.FELT_MU,
                 contact_offset=0.0005)]
    for i, (bx, by) in enumerate(L.BOXES, start=1):
        out.append(Plate(f"box{i}_tape", pos=(bx, by, FELT_TOP + 2e-5), solid=False, color=L.TAPE_WHITE,
                         size=(L.BOX_L + 2 * L.TAPE, L.BOX_W + 2 * L.TAPE, 2e-5)))
        out.append(Plate(f"box{i}_felt", pos=(bx, by, FELT_TOP + 5e-5), solid=False, color=L.FELT_GREEN,
                         size=(L.BOX_L, L.BOX_W, 2e-5)))
    out.append(Plate("pack", pos=(*L.DECK_XY, FELT_TOP + L.CARD_GAP + BLOCK_T / 2), size=(L.CARD_L, L.CARD_W, BLOCK_T),
                     mass=(L.DECK_N - L.LOOSE_N) * L.CARD_MASS, color=L.CARD_WHITE, friction=L.CARD_MU))
    for k, name in enumerate(CARDS):
        out.append(Plate(name, pos=(*L.DECK_XY, _card_z(k)), size=(L.CARD_L, L.CARD_W, L.CARD_T), mass=L.CARD_MASS,
                         color=L.CARD_WHITE, back=L.CARD_RED, friction=L.CARD_MU))
    return out


OBJECTS = _objects()

# -- scoring -----------------------------------------------------------------------------------------------------------
TRIAL = scoring.Trial()
_SEEN = {"tick": None, "value": 0.0}


def _reset() -> None:
    TRIAL.reset()
    _SEEN.update(tick=None, value=0.0)


ON_RESET.append(_reset)


def _cards(objects) -> dict[str, scoring.Card]:
    return {n: scoring.Card(np.asarray(objects[n].pos, float), quat_matrix(objects[n].quat_xyzw)) for n in CARDS}


def _evaluate(objects) -> float:
    tick = cardkit.tick()
    if tick and tick == _SEEN["tick"]:
        return _SEEN["value"]
    cards, pack, dt = _cards(objects), np.asarray(objects["pack"].pos, float), cardkit.step_dt()
    TRIAL.update(cards, pack, TOP, dt)
    value = scoring.score(TRIAL, cards, pack, FELT_TOP, cardkit.fingertips(), dt)
    _SEEN.update(tick=tick, value=value)
    if os.environ.get("TASK_DEBUG") and tick % 10 == 0:
        _report(tick, cards, pack, value)
    return value


def _report(tick, cards, pack, value) -> None:
    start = np.array([*L.DECK_XY, 0.0])
    moved = " ".join(f"{n[-2:]}:{1000 * (c.pos[0] - start[0]):+.1f},{1000 * (c.pos[1] - start[1]):+.1f},"
                     f"{1000 * (c.pos[2] - FELT_TOP):.2f}/{c.tilt_deg:.0f}/{c.yaw_deg:+.0f}"
                     for n, c in list(cards.items())[:5])
    tips = cardkit.fingertips()
    tip = f"{1000 * (tips[::2, 2].min() - DECK_TOP):+.1f}" if len(tips) > 1 else cardkit._ENV.get("finger_error", "n/a")
    t = scoring.tally(TRIAL, cards, pack, FELT_TOP)
    print(f"[deal] step {tick:4d} score {value:.2f} pack {1000 * (pack[0] - start[0]):+.1f},"
          f"{1000 * (pack[1] - start[1]):+.1f} tip-z {tip} cards(dx,dy,z/tilt/yaw) {moved} "
          f"dealt {[d[-2:] if d else '-' for d in t['dealt']]} strays {len(t['strays'])} {TRIAL.spoiled}", flush=True)


def score(objects) -> float:
    return _evaluate(objects)


def success(objects) -> bool:
    return _evaluate(objects) >= 1.0


# -- scripted demo -----------------------------------------------------------------------------------------------------
def demo_controller(bases, dt):
    """The closed-loop demo in deal_demo.py (cardkit makes yam_scripted run it)."""
    import deal_demo

    return deal_demo.Dealer(bases, dt, CARDS, "pack")


def demo() -> dict:
    """Nothing is played open loop. With DEAL_MODE=rest the arms stay at their ready pose."""
    return {}


if os.environ.get("DEAL_MODE") == "rest":
    del demo_controller
