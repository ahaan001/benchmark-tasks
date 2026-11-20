"""Offline tests (numpy only): the scoring rules on hand-built card poses, and the scripted demo's reach.

Run:  PYTHONPATH=<yam-isaac-lab-arena> python3 test_task.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import scoring  # noqa: E402
import task  # noqa: E402

DT = 0.05
FELT = task.FELT_TOP
PACK = np.array([*L.DECK_XY, FELT + task.BLOCK_T / 2])
FLAT = np.eye(3)
FLIP = np.diag([1.0, -1.0, -1.0])
FAR = np.array([[0.2, 0.0, 1.2]])


def yaw(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def start() -> dict:
    return {n: scoring.Card(np.array([*L.DECK_XY, task._card_z(k)]), FLAT.copy()) for k, n in enumerate(task.CARDS)}


def put(cards, name, xy, rot=FLAT, z=None) -> None:
    cards[name] = scoring.Card(np.array([xy[0], xy[1], FELT + L.CARD_T / 2 if z is None else z]), rot.copy())


def run(cards, fingers=FAR, pack=PACK, steps=25, trial=None) -> float:
    trial = trial or scoring.Trial()
    for _ in range(steps):
        trial.update(cards, pack, task.TOP, DT)
    return scoring.score(trial, cards, pack, FELT, fingers, DT)


def dealt(n: int) -> dict:
    cards = start()
    for i in range(n):
        put(cards, task.CARDS[i], L.BOXES[i])
    return cards


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def nothing_scores_zero():
    assert run(start()) == 0.0


@case
def ladder():
    assert [run(dealt(n)) for n in range(5)] == [0.0, 0.25, 0.5, 0.75, 1.0]


@case
def any_order_any_box():
    cards = start()
    put(cards, "card_01", L.BOXES[3])
    put(cards, "card_02", L.BOXES[0])
    assert run(cards) == 0.5


@case
def held_or_moving_is_three_quarters():
    c = dealt(4)
    near = np.array([[*L.BOXES[2], FELT + 0.02]])
    assert run(c, fingers=near) == 0.75
    assert run(c, steps=5) == 0.75  # not yet a full second at rest
    trial = scoring.Trial()
    run(c, trial=trial)
    put(c, "card_01", (L.BOXES[0][0] + 0.004, L.BOXES[0][1]))
    assert run(c, steps=3, trial=trial) == 0.75  # moved 4 mm within the last second


@case
def finger_just_clear():
    c = dealt(4)
    assert run(c, fingers=np.array([[*L.BOXES[1], FELT + L.CARD_T + 0.031]])) == 1.0
    assert run(c, fingers=np.array([[*L.BOXES[1], FELT + L.CARD_T + 0.029]])) == 0.75


@case
def card_over_the_tape_is_a_stray():
    c = dealt(4)
    put(c, "card_04", (L.BOXES[3][0] + 0.020, L.BOXES[3][1]))  # 2 cm off centre: 1.4 mm over the tape's outer edge
    assert run(c) == 0.5  # three dealt, one stray
    put(c, "card_04", (L.BOXES[3][0] + 0.018, L.BOXES[3][1]))
    assert run(c) == 1.0
    put(c, "card_04", L.BOXES[3], rot=yaw(20))
    assert run(c) == 1.0  # askew but wholly inside
    put(c, "card_04", L.BOXES[3], rot=yaw(40))
    assert run(c) == 0.5


@case
def two_in_a_box():
    c = dealt(3)
    put(c, "card_04", L.BOXES[2], z=FELT + 1.5 * L.CARD_T)
    assert run(c) == 0.25  # two boxes dealt, the shared box nothing, its second card a stray


@case
def face_up_never_counts():
    c = dealt(4)
    put(c, "card_02", L.BOXES[1], rot=FLIP)
    assert run(c) == 0.5
    trial = scoring.Trial()
    run(c, trial=trial)
    put(c, "card_02", L.BOXES[1])  # turned back over: it has been seen
    assert run(c, trial=trial) == 0.5


@case
def stray_cards_cost():
    c = dealt(2)
    put(c, "card_03", (0.33, 0.0))
    assert run(c) == 0.25
    put(c, "card_04", (0.33, 0.12))
    assert run(c) == 0.0


@case
def fanned_pack_counts_as_strays():
    c = dealt(4)
    c["card_05"].pos[0] += 0.009
    assert run(c) == 1.0  # within 1 cm of the pack
    c["card_05"].pos[0] += 0.003
    assert run(c) == 0.75


@case
def shoved_pack_spoils_the_trial_for_good():
    c = dealt(4)
    trial = scoring.Trial()
    assert run(start(), trial=trial) == 0.0
    shoved = PACK + [0.021, 0.0, 0.0]
    for n in task.CARDS[4:]:
        c[n].pos[0] += 0.021
    assert run(c, pack=shoved, trial=trial) == 0.0
    for n in task.CARDS[4:]:
        c[n].pos[0] -= 0.021
    assert run(c, trial=trial) == 0.0  # pushed back: still 0
    assert run(dealt(4), pack=PACK + [0.019, 0.0, 0.0]) < 1.0  # 1.9 cm: allowed, but its cards are now off the pack


@case
def card_off_the_table():
    c = dealt(4)
    put(c, "card_04", (0.95, 0.0), z=task.TOP - 0.3)
    assert run(c) == 0.0


@case
def whole_pack_in_a_box_scores_zero():
    c = start()
    pack = np.array([*L.BOXES[1], PACK[2]])
    for n in task.CARDS:
        c[n].pos[:2] = L.BOXES[1]
    assert run(c, pack=pack) == 0.0


@case
def reset_clears_only_on_reset():
    c = dealt(4)
    trial = scoring.Trial()
    run(start(), trial=trial)
    assert run(c, pack=PACK + [0.03, 0.0, 0.0], trial=trial) == 0.0
    assert trial.spoiled == "pack moved" and run(dealt(4), trial=trial) == 0.0
    trial.reset()
    assert run(dealt(4), trial=trial) == 1.0


@case
def card_leaning_on_the_pack_is_not_dealt():
    c = dealt(3)
    tilt = np.array([[math.cos(0.17), 0, math.sin(0.17)], [0, 1, 0], [-math.sin(0.17), 0, math.cos(0.17)]])
    put(c, "card_04", (0.27, 0.0), rot=tilt, z=FELT + 0.008)
    assert run(c) == 0.5


@case
def instruction_and_limits():
    assert len(task.INSTRUCTION) <= 1000
    for word in ("slide", "fingertip", "friction", "press"):
        assert word not in task.INSTRUCTION.lower()


@case
def scene_is_consistent():
    names = [o.name for o in task.OBJECTS]
    assert len(names) == len(set(names)) and len(task.CARDS) == L.LOOSE_N
    zs = [o.pos[2] for o in task.OBJECTS if o.name in task.CARDS]
    assert zs == sorted(zs, reverse=True) and abs((task.DECK_TOP - FELT) - L.DECK_N * L.CARD_T) < 2e-4
    for bx, by in L.BOXES:  # boxes clear of the pack and of each other, on the felt
        assert abs(bx - L.DECK_XY[0]) > (L.BOX_L + L.CARD_L) / 2 or abs(by - L.DECK_XY[1]) > (L.BOX_W + L.CARD_W) / 2
        assert L.FELT_X[0] < bx - L.BOX_L / 2 and bx + L.BOX_L / 2 < L.FELT_X[1]


class StandIn:
    """Kinematics only: a card follows the fingertips while they are on it, tips over the pack's edge when let go
    and drops flat once its back end is clear of the pack. It checks the controller, not the physics."""

    def __init__(self):
        self.pos = {n: np.array([*L.DECK_XY, task._card_z(k)]) for k, n in enumerate(task.CARDS)}
        self.rot = {n: np.eye(3) for n in task.CARDS}
        self.offset = {}

    def read(self, names):
        return {n: ((PACK, np.eye(3)) if n == "pack" else (self.pos[n].copy(), self.rot[n].copy())) for n in names}

    def step(self, d) -> None:
        if not d.todo:
            return
        arm, card, _ = d.todo[0]
        tip = d.tcp[arm] - [0.0, 0.0, 0.013]
        edge, h = PACK[0] + L.CARD_L / 2, task.DECK_TOP - FELT - L.CARD_T
        if d.phase == "push":
            self.pos[card][0] = max(self.pos[card][0], tip[0] + demo_module().BACK)
        elif d.phase in ("let go", "over the front") and self.rot[card][2, 0] == 0.0:
            over = self.pos[card][0] + L.CARD_L / 2 - edge
            if over > L.CARD_L / 2:
                c, s = math.cos(math.asin(h / over)), h / over
                self.rot[card] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
                front = np.array([edge + over * c, self.pos[card][1], FELT])
                self.pos[card] = front - self.rot[card][:, 0] * L.CARD_L / 2 + self.rot[card][:, 2] * L.CARD_T / 2
        elif d.phase == "slide":
            self.offset.setdefault(card, self.pos[card][:2] - tip[:2])
            self.pos[card][:2] = tip[:2] + self.offset[card]
            if self.pos[card][0] - L.CARD_L / 2 > edge - 0.002:
                self.rot[card], self.pos[card][2] = np.eye(3), FELT + L.CARD_T / 2


def demo_module():
    import deal_demo

    return deal_demo


def play(mode: str):
    from yam_arena import ik, station

    dd = demo_module()
    dd.MODE = mode
    world = StandIn()
    d = dd.Dealer(station.BASES, DT, task.CARDS, "pack", read=world.read)
    margin, steps = 1e9, 0
    while steps < L.RUN_LIMIT:
        _, done = d.act(None)
        steps += 1
        for a in d.engaged:
            margin = min(margin, float(np.degrees(np.minimum(d.q[a] - ik.LOWER, ik.UPPER - d.q[a])).min()))
        world.step(d)
        if done:
            break
    cards = {n: scoring.Card(world.pos[n], world.rot[n]) for n in task.CARDS}
    return run(cards), steps, d.worst_ik, margin


@case
def demo_on_the_stand_in():
    value, steps, ik_error, margin = play("full")
    assert value == 1.0, value
    assert steps < L.RUN_LIMIT / 2 and ik_error < 5e-4 and margin > 10.0, (steps, ik_error, margin)
    print(f"     stand-in: full demo {steps} steps, IK error {1000 * ik_error:.3f} mm, joint margin {margin:.1f} deg")


@case
def task_json_agrees_with_the_code():
    import json

    path = pathlib.Path(__file__).resolve().parent / "task.json"
    d = json.loads(path.read_text())
    limits = {"title": 120, "instruction": 1000, "context": 1000, "fidelity": 500, "scores": 300, "run_limit": 200}
    assert all(len(d[k]) <= n for k, n in limits.items())
    assert d["instruction"] == task.INSTRUCTION and d["category"] == "Societal relatability"
    for x, y in L.BOXES:
        assert f"x {100 * x:g}, y {100 * y:g}" in d["setup"]
    assert f"{1000 * L.CARD_T:g} mm" in d["setup"] and f"{L.RUN_LIMIT:,} steps" in d["run_limit"]
    assert f"more than {100 * L.PACK_MOVED:g} cm" in d["scoring"] and f"at least {100 * L.CLEAR:g} cm" in d["scoring"]
    assert "\u2014" not in path.read_text() and chr(0x2014) not in path.read_text()


if __name__ == "__main__":
    failed = 0
    for fn in CASES:
        try:
            fn()
            print("ok  ", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, e)
    print(f"{len(CASES) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
