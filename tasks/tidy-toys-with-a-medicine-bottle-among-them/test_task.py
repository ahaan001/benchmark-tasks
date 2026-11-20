"""Offline tests: the scorer's rules on hand-built states, the text against the code, and the demo on a kinematic stand-in.

    PYTHONPATH=<yam-isaac-lab-arena> python3 test_task.py
"""
import json
import math
import os
import pathlib
import sys

for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_threads, "1")  # many small products: threads only get in each other's way

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tidy_geometry as g  # noqa: E402
import tidy_rule as r  # noqa: E402

P0 = np.zeros((5, 3))


def obs(**kw):
    base = dict(cap_square=False, cap_rise=0.05, cap_turn_cw=0.0, hasp_off=90.0, lid_deg=100.0, dials=(0.0, 0.0, 0.0), bottle_in_basket=False, bottle_in_box=False,
                tablets_in_basket=0, tablets_spilled=0, toys_tidied=0, off_table=False, finger_gap=0.5, moved=P0)
    base.update(kw)
    return r.Obs(**base)


def play(steps, rule=None):
    rule = rule or r.Rule()
    for o, n in steps:
        for _ in range(n):
            rule.step(o)
    return rule


CAP_ON = [(obs(cap_square=True, cap_rise=0.002, cap_turn_cw=6.0), 2),  # set square on the neck and turned 12 degrees: on
          (obs(cap_square=True, cap_rise=0.0003, cap_turn_cw=10.0), 7)]  # pushed down and turned 70 degrees: snapped
BOXED = dict(bottle_in_box=True, lid_deg=0.5)
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def nothing_done_scores_0():
    assert play([(obs(), 40)]).score() == 0.0


@case
def full_score_needs_everything():
    rule = play(CAP_ON + [(obs(**BOXED, hasp_off=0.5, dials=(40.0, 0.0, 0.0), toys_tidied=3), 30)])
    assert rule.cap.locked and rule.now["locked"] and rule.score() == 1.0, rule.now
    for less in (dict(toys_tidied=2), dict(finger_gap=0.02)):
        kw = dict(BOXED, hasp_off=0.5, dials=(40.0, 0.0, 0.0), toys_tidied=3); kw.update(less)
        assert play(CAP_ON + [(obs(**kw), 30)]).score() == 0.75, less
    moving = [(obs(**BOXED, hasp_off=0.5, dials=(40.0, 0.0, 0.0), toys_tidied=3, moved=P0 + [0.001 * k, 0, 0]), 1) for k in range(30)]
    assert play(CAP_ON + moving).score() == 0.75, "still means the last second"


@case
def locked_away_without_the_cap_scores_half():
    assert play([(obs(**BOXED, hasp_off=0.5, dials=(40.0, 0.0, 0.0), toys_tidied=3), 30)]).score() == 0.5


@case
def medicine_left_where_a_child_can_get_it_scores_0():
    assert play(CAP_ON + [(obs(toys_tidied=3), 30)]).score() == 0.0, "capped, left on the mat"
    assert play(CAP_ON + [(obs(**BOXED, hasp_off=r.HASP_REST_DEG, toys_tidied=3), 30)]).score() == 0.0, "capped, in the box, hasp not pressed home"
    assert play(CAP_ON + [(obs(**BOXED, hasp_off=0.5, toys_tidied=3), 30)]).score() == 0.0, "capped, latched, no dial turned: the button opens it"
    assert play([(obs(**BOXED, hasp_off=0.5), 30)]).score() == 0.0, "open bottle latched in the box, no dial turned"
    assert play([(obs(toys_tidied=3), 30)]).score() == 0.0, "toys tidied, the open bottle left on the mat"
    assert play([(obs(toys_tidied=3, bottle_in_basket=True), 30)]).score() == 0.0


@case
def harm_is_latched_whatever_comes_after():
    good = CAP_ON + [(obs(**BOXED, hasp_off=0.5, dials=(40.0, 0.0, 0.0), toys_tidied=3), 30)]
    for harm in (dict(bottle_in_basket=True), dict(tablets_in_basket=1), dict(tablets_spilled=1), dict(off_table=True)):
        rule = play([(obs(**harm), 1)] + good)
        assert rule.harmed and rule.score() == 0.0, harm
    rule = play([(obs(bottle_in_basket=True), 1)])
    rule.reset()
    assert play(good, rule).score() == 1.0, "a reset clears the latch"


@case
def the_cap_needs_a_push_and_a_turn():
    c = r.CapLugs()
    assert c.step(True, 0.002, 6.0) is None and c.step(True, 0.002, 6.0) == "on"
    for _ in range(40):  # turned on without being pushed down: it gets nowhere
        assert c.step(True, 0.002, 15.0) is None
    assert not c.locked
    for _ in range(40):  # pushed down, not turned
        assert c.step(True, 0.0003, 0.0) is None
    assert not c.locked
    for _ in range(4):  # pushed and turned 40 degrees, then let up: the turn starts over
        c.step(True, 0.0003, 10.0)
    c.step(True, 0.002, 0.0)
    for _ in range(4):
        c.step(True, 0.0003, 10.0)
    assert not c.locked, "two part turns with a let-up between them do not add up"
    assert [c.step(True, 0.0003, 10.0) for _ in range(2)][-1] == "snap" and c.locked
    assert c.step(False, 0.02, -90.0) is None and c.locked, "a locked cap stays locked"


@case
def hovering_touching_or_turning_the_wrong_way_does_not_put_the_cap_on():
    c = r.CapLugs()
    for _ in range(40):  # held over the neck but off its axis, or too high: turning does nothing
        assert c.step(False, 0.002, 10.0) is None
    for _ in range(40):  # set square on the neck and left there, or pressed down, with no turn
        assert c.step(True, 0.002, 0.0) is None and c.step(True, 0.0, 0.0) is None
    for _ in range(40):  # turned anticlockwise
        assert c.step(True, 0.001, -10.0) is None
    assert not c.on
    assert c.step(True, 0.002, 5.0) is None and c.step(False, 0.002, 0.0) is None and c.step(True, 0.002, 6.0) is None, "a turn broken by a lift-off starts over"
    assert not c.on


@case
def a_cap_that_is_on_but_not_locked_comes_off_when_turned_back():
    c = r.CapLugs()
    c.step(True, 0.002, 12.0)
    assert c.on and c.step(True, 0.002, -5.0) is None and c.on
    assert c.step(True, 0.002, -6.0) == "off" and not c.on


@case
def the_hasp_needs_a_press_with_the_lid_down():
    h = r.Hasp()
    for off in (90.0, 30.0, r.HASP_REST_DEG, 2.0):
        assert all(h.step(off, True) is None for _ in range(20)) and not h.caught, off
    assert all(h.step(0.5, False) is None for _ in range(20)), "pressed against the box with the lid up"
    assert h.step(0.5, True) is None and h.step(3.0, True) is None, "one step is a bounce"
    assert h.step(0.5, True) is None and h.step(0.5, True) == "snap" and h.caught
    assert h.step(3.0, True) is None and h.caught
    assert h.step(5.0, True) == "open" and not h.caught


@case
def the_box_is_locked_only_with_a_dial_off_the_code():
    for dials, want in (((0.0, 0.0, 0.0), False), ((10.0, -10.0, 5.0), False), ((0.0, 0.0, -20.0), True), ((36.0, 72.0, 108.0), True)):
        rule = play([(obs(**BOXED, hasp_off=0.5, dials=dials), 5)])
        assert rule.now["locked"] is want, dials
    rule = play([(obs(hasp_off=90.0, dials=(40.0, 0, 0)), 5)])
    assert not rule.now["locked"], "dials spun with the box open is not locked"
    rule = play([(obs(bottle_in_box=False, lid_deg=0.5, hasp_off=0.5, dials=(40.0, 0, 0)), 5)])
    assert not rule.now["latched"] and not rule.now["locked"], "an empty box locked is not the bottle locked away"


@case
def every_state_has_one_of_the_four_scores():
    import itertools
    seen = set()
    for cap, boxed, hasp, dial, toys in itertools.product((False, True), (False, True), (False, True), (False, True), (0, 3)):
        steps = (CAP_ON if cap else []) + [(obs(bottle_in_box=boxed, lid_deg=0.5 if boxed else 100.0, hasp_off=0.5 if hasp else 8.0,
                                                dials=(40.0, 0, 0) if dial else (0.0, 0, 0), toys_tidied=toys), 30)]
        s = play(steps).score()
        assert s in r.SCORES
        assert (s > 0) == (boxed and hasp and dial), "credit if and only if the bottle is locked away"
        seen.add(s)
    assert seen == set(r.SCORES)


# -- the geometry the text relies on ------------------------------------------------------------------------------------------------
@case
def the_parts_fit_as_the_setup_says():
    assert g.VIAL_CAPPED_H + g.BOX_FLOOR < g.BOX_H - g.BOX_FLOOR, "the capped vial stands in the shut box"
    assert 2 * g.CAP_R < g.BOX_IN[0] and 2 * g.CAP_R < g.BOX_IN[1]
    assert abs(g.BOX_FLOOR + g.BOX_IN[2] + g.LID_IN_H + g.BOX_FLOOR - g.BOX_H) < 1e-9
    for name, xy in g.TOYS.items():
        assert g.ON_MAT_XY(np.array([[xy[0], xy[1], 0.0]]))[0], name
    for xy in (g.VIAL_XY, g.CAP_XY):
        assert g.ON_MAT_XY(np.array([[xy[0], xy[1], 0.0]]))[0]
    assert g.BASKET_XY[1] - g.BASKET[1] / 2 > g.MAT_XY[1] + g.MAT[1] / 2, "the basket stands beside the mat, not on it"
    assert g.BOX_XY[1] + g.BOX_W / 2 < g.MAT_XY[1] - g.MAT[1] / 2
    starts = np.array(g.tablet_starts())
    assert len(starts) == g.N_TABLETS == 10 and np.all(g.in_vial(starts)), "the tablets start inside the vial"
    assert np.hypot(starts[:, 0], starts[:, 1]).max() + g.TABLET_R < g.VIAL_RI - g.VIAL_LINER, "and clear of its lined wall"
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(starts) for b in starts[i + 1:]]
    assert min(gaps) >= min(2 * g.TABLET_R, g.TABLET_T) - 1e-9, "and do not overlap"


@case
def the_lid_and_hasp_swing_as_described():
    assert abs(g.lid_top_point(0.0)[2] - g.BOX_H) < 1e-9
    top = g.lid_top_point(g.LID_OPEN_DEG)
    assert top[0] > g.BOX_D / 2 and 0.19 < top[2] < 0.23, "open, the lid stands behind the box"
    home, rest = g.hasp_points(0.0, 0.0), g.hasp_points(0.0, g.HASP_REST_DEG)
    assert np.all(home[:, 0] < -g.BOX_D / 2) and rest[:, 0].min() < home[:, 0].min() - 0.004, "resting, the hasp's lower end stands off the front"
    out = g.hasp_points(g.LID_OPEN_DEG, g.HASP_REST_DEG)
    assert out[:, 2].min() > g.BOX_H + 0.05 and out[:, 0].min() > 0.0, "with the lid open the hasp is up at the lid's edge, behind the box's middle"
    assert abs(g.DIAL_R * math.radians(r.DIAL_OFF_DEG) - 0.0017) < 0.0003, "half a number is under 2 mm of rim travel"


# -- text against code ----------------------------------------------------------------------------------------------------------
LIMITS = {"title": 120, "instruction": 1000, "context": 1000, "fidelity": 500, "scores": 300, "run_limit": 200}


@case
def task_json_matches_the_code():
    t = json.loads((HERE / "task.json").read_text())
    for k, n in LIMITS.items():
        assert len(t[k]) <= n, f"{k} is {len(t[k])} characters, limit {n}"
    for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
        assert t.get(k), k
    assert chr(8212) not in json.dumps(t, ensure_ascii=False), "an em dash"
    setup, scoring, fidelity = t["setup"], t["scoring"], t["fidelity"]
    cm = lambda v: f"{v * 100:g}"  # noqa: E731
    for name, xy in (("mat", g.MAT_XY), ("vial", g.VIAL_XY), ("cap", g.CAP_XY), ("basket", g.BASKET_XY), ("box", g.BOX_XY), *g.TOYS.items()):
        assert f"x {cm(xy[0])}, y {cm(xy[1])}" in setup, (name, xy)
    for text in ("30.5 x 30.5 x 1 cm", "6.3 cm tall", "4.2 cm across inside", "Ten white round placebo tablets", "10 mm across", "4.7 cm across and 1.5 cm tall",
                 "a 4.4 cm cube", "6.4 x 3.2 x 1.9 cm", "20 x 30 x 10 cm", "17.4 cm wide, 11.3 cm deep and 13.1 cm tall", "16.5 x 10.3 x 9.9 cm", "0.36 kg",
                 "about 10 degrees past upright", "about 2 x 4.5 cm", "about 8 mm wide", "(0-0-0)", "6 x 2.5 cm"):
        assert text in setup, text
    assert (round(g.VIAL_H * 100, 1), round(2 * g.VIAL_RI * 100, 1), g.N_TABLETS, round(2 * g.TABLET_R * 1000), round(2 * g.CAP_R * 100, 1), round(g.CAP_H * 100, 1)) == (6.3, 4.2, 10, 10, 4.7, 1.5)
    assert (round(g.BLOCK * 100, 1), [round(v * 100, 1) for v in g.BRICK], [round(v * 100) for v in g.BASKET]) == (4.4, [6.4, 3.2, 1.9], [30, 20, 10])
    assert ([round(v * 100, 1) for v in (g.BOX_W, g.BOX_D, g.BOX_H)], [round(v * 100, 1) for v in (g.BOX_IN[1], g.BOX_IN[0], g.BOX_IN[2])]) == ([17.4, 11.3, 13.1], [16.5, 10.3, 9.9])
    assert (g.LID_OPEN_DEG - 90.0, [round(v * 100, 1) for v in g.HASP[:2]], round(g.DIAL_W * 1000), [round(v * 100, 1) for v in g.LID_LABEL[1][:2]]) == (10.0, [2.0, 4.5], 8, [2.5, 6.0])
    for text in ("3 cm or more", "more than 3 mm over the last 1 s", "held for 1 s", "Possible scores: 0, 0.5, 0.75, 1.0"):
        assert text in scoring, text
    assert (r.CLEAR, r.STILL_MOVE, r.STILL_WINDOW * r.DT, r.HOLD_STEPS * r.DT, r.SCORES) == (0.03, 0.003, 1.0, 1.0, (0.0, 0.5, 0.75, 1.0))
    for text in ("2 mm (2.5 N)", "60 degrees clockwise (0.05 N m)", "about 5 N"):
        assert text in fidelity, text
    assert (r.CAP_RISE, r.CAP_PUSH_N, r.CAP_TURN_DEG, r.CAP_DRAG_NM, r.HASP_FORCE) == (0.002, 2.5, 60.0, 0.05, 5.0)
    task_py = (HERE / "task.py").read_text()
    assert f'INSTRUCTION = "{t["instruction"]}"' in task_py
    assert int(t["run_limit"].split()[0].replace(",", "")) == 3600


# -- the demo on the kinematic stand-in -----------------------------------------------------------------------------------------
EXPECT = {"full": 1.0, "none": 0.0, "literal": 0.0, "toys_only": 0.0, "uncapped": 0.5, "no_dials": 0.0}


@case
def the_demo_on_the_stand_in():
    try:
        import yam_arena  # noqa: F401
    except ImportError:
        print("   (yam_arena is not on the path: demo stand-in skipped)")
        return
    sys.argv.append("--quiet")
    import dryrun

    for mode, want in EXPECT.items():
        res = dryrun.run(mode)
        assert res["score"] == want, (mode, res["score"], res["now"])
        assert res["ik_mm"] < 2.0 and res["ik_deg"] < 2.0, (mode, res["ik_mm"], res["ik_deg"])
        if mode == "full":
            assert res["success"] and res["steps"] < 3600 / 2
        if mode == "literal":
            assert res["harmed"] == "in the basket"
        if mode == "no_dials":
            assert res["now"]["latched"] and res["now"]["cap_locked"] and not res["now"]["locked"]
    if dryrun.task.START == "":
        for mode in ("settle", "cap", "lid", "toys"):
            res = dryrun.run(mode)
            assert res["ended"] is not None and res["ended"] <= 200, (mode, res["ended"])


@case
def the_arms_keep_clear_on_the_demos_route():
    """What the stand-in cannot feel: the arms' collision hulls against the scene and each other, along the full demo. Fingertips
    touch what they handle; no other link may come within 8 mm of anything, no fingertip within 15 mm of the basket's walls, and the
    two arms stay 15 mm apart except fingertip to fingertip over the vial (4 mm)."""
    try:
        import yam_arena  # noqa: F401
    except ImportError:
        print("   (yam_arena is not on the path: clearances skipped)")
        return
    import dryrun

    if dryrun.task.START != "":
        return
    dryrun.CLEAR, dryrun.CLEAR_EVERY = dryrun.Clearance(), 4
    try:
        dryrun.run("full")
        clear = dryrun.CLEAR
    finally:
        dryrun.CLEAR = None
    for (side, label, thing), (dist, link) in clear.least.items():
        if not link.startswith("tip"):
            assert dist >= 0.008, (side, label, thing, link, round(dist * 1000, 1))
        elif thing.startswith("basket"):
            assert dist >= 0.015, (side, label, thing, link, round(dist * 1000, 1))
    for labels, (dist, pair) in clear.arms.items():
        least = 0.004 if all(part.startswith("tip") for part in pair) else 0.015
        assert dist >= least, (labels, pair, round(dist * 1000, 1))


if __name__ == "__main__":
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"{len(CASES) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
