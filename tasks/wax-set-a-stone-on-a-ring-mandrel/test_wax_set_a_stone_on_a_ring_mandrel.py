"""Offline checks, no simulator: every scoring outcome and loophole on hand-built poses, the wax rule's red-team cases,
the bench's geometry, the props' USD, the demo's reach, and the task's text against its code.

    PYTHONPATH=<yam-isaac-lab-arena> python3 test_wax_set_a_stone_on_a_ring_mandrel.py      (or pytest)

Exits non-zero if any check fails.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import mechkit as m  # noqa: E402
import ring_geometry as g  # noqa: E402
import wax_rule as w  # noqa: E402

_spec = importlib.util.spec_from_file_location("wax_task", HERE / "wax_set_a_stone_on_a_ring_mandrel.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)

DT = 0.05
BAD: list[str] = []


def check(label, got, want, tol=1e-9):
    ok = (got == want) if isinstance(want, (bool, str, dict)) else abs(got - want) <= tol
    if not ok:
        BAD.append(label)
    print(f"{'ok ' if ok else 'BAD'} {label}: {got} (want {want})")
    return ok


def at_least(label, got, floor):
    ok = got >= floor
    if not ok:
        BAD.append(label)
    print(f"{'ok ' if ok else 'BAD'} {label}: {got:.4g} (at least {floor:.4g})")
    return ok


def at_most(label, got, ceiling):
    ok = got <= ceiling
    if not ok:
        BAD.append(label)
    print(f"{'ok ' if ok else 'BAD'} {label}: {got:.4g} (at most {ceiling:.4g})")
    return ok


# -- hand-built scenes ----------------------------------------------------------------------------------------------------


def finger_pair(centre, half=0.02, axis=(1.0, 0.0, 0.0)):
    """Two fingers either side of `centre` along `axis`, as columns of points 4 cm tall."""
    c, a = np.asarray(centre, float), np.asarray(axis, float)
    col = np.array([[0.0, dy, dz] for dy in (-0.009, 0.0, 0.009) for dz in (-0.02, -0.01, 0.0, 0.01, 0.02)])
    return (c - half * a + col, c + half * a + col)


FAR = {"left": finger_pair((0.20, 0.30, 0.30)), "right": finger_pair((0.10, -0.45, 0.30))}


def scene(**kw) -> dict:
    s = dict(dt=DT, mandrel=w.Pose(w.MANDREL_START.pos.copy()), claws=[0.0] * g.N_CLAWS, seat=0.0, stone=w.Pose(w.STONE_START.pos.copy()), pen=w.Pose(w.PEN_START.pos.copy()),
             tweezers=w.Pose(w.TWEEZERS_START.pos.copy()), pusher=w.Pose(w.PUSHER_START.pos.copy()), fingers=dict(FAR), at_rest=True)
    s.update(kw)
    return s


def step(wax, s, n=1) -> dict:
    out = {}
    for _ in range(n):
        out = wax.step(w.Snapshot(dt=s["dt"], mandrel=s["mandrel"], claws=tuple(s["claws"]), seat=s["seat"], stone=s["stone"], pen=s["pen"], tweezers=s["tweezers"],
                                  pusher=s["pusher"], fingers=s["fingers"], at_rest=s["at_rest"]))
    return out


def stone_at(s, up=0.0, tilt_deg=0.0, off=0.0):
    """Put the stone in the head: `up` above the seat line, tilted about x, `off` from the head's axis."""
    a = math.radians(tilt_deg)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    s["stone"] = w.Pose(s["mandrel"].world((off, 0.0, g.Z_S + up)), s["mandrel"].rot @ rot)


def pen_ball_at(s, point_world, held=True):
    """Stand the pen upright with its tip's ball at a point; a hand on its handle unless told otherwise."""
    s["pen"] = w.Pose(np.asarray(point_world, float) - np.array(g.PEN_HOT[0]))
    if held:
        s["fingers"] = dict(FAR, right=finger_pair(s["pen"].world((0.0, 0.0, g.PEN_GRASP_Z)), half=g.PEN_HANDLE_R + 0.0045))


def pen_on_claw(s, i, angle, off=0.0):
    pen_ball_at(s, s["mandrel"].world(task.claw_ball(i, angle) + off * task.claw_out(i, angle)))


def pen_away(s, in_stand=False):
    if in_stand:
        s["pen"] = w.Pose(w.PEN_START.pos.copy())
        s["fingers"] = dict(FAR)
    else:
        pen_ball_at(s, s["mandrel"].world((0.03, -0.03, 0.06)))


def close_claw(wax, s, i, dwell=1.7, push=1.0, to=None, cool=1.6):
    """What the simulator would do under a warm tip: the tip rests on claw i, the wax gives, the claw follows the tip round, the tip leaves."""
    to = g.CLAW_ON_CROWN if to is None else to
    a0 = s["claws"][i]
    pen_on_claw(s, i, a0)
    step(wax, s, round(dwell / DT))
    n = round(push / DT)
    for k in range(1, n + 1):
        s["claws"][i] = a0 + (to - a0) * k / n
        pen_on_claw(s, i, s["claws"][i])
        step(wax, s)
    pen_away(s)
    step(wax, s, round(cool / DT))


def fresh():
    return w.Wax(), scene()


def all_done(wax, s, claws=(0, 1, 2, 3)):
    stone_at(s, 0.0)
    s["seat"] = -g.SEAT_TRAVEL
    step(wax, s, 2)
    for i in claws:
        close_claw(wax, s, i)


# -- the scoring ladder -------------------------------------------------------------------------------------------------------


def test_ladder():
    wax, s = fresh()
    check("start, nothing moved", step(wax, s, 5)["score"], 0.0)
    stone_at(s, g.SEAT_TRAVEL)
    check("stone dropped in the head, not pressed", step(wax, s, 5)["score"], 0.1)
    stone_at(s, 0.0)
    s["seat"] = -g.SEAT_TRAVEL
    check("stone pressed home", step(wax, s, 5)["score"], 0.2)
    for n, want in zip(range(1, 5), (0.35, 0.5, 0.65, 0.8)):
        close_claw(wax, s, (0, 2, 1, 3)[n - 1])
        check(f"pressed, {n} claw(s) closed, pen still in the hand", wax.score, want)
    pen_away(s, in_stand=True)
    check("all four closed, everything put back", step(wax, s, 3)["score"], 1.0)
    check("success() needs the full score", wax.full, True)

    wax, s = fresh()
    all_done(wax, s, claws=(0, 1, 2))
    pen_away(s, in_stand=True)
    check("three claws, everything put back", step(wax, s, 3)["score"], 0.85)

    wax, s = fresh()
    all_done(wax, s, claws=(0, 2))
    pen_away(s, in_stand=True)
    check("two claws, everything put back: the stone is not held, so no credit for putting back", step(wax, s, 3)["score"], 0.5)

    wax, s = fresh()
    stone_at(s, 0.0, tilt_deg=15.0)
    check("stone down but tilted 15 degrees: in the head, not home", step(wax, s, 3)["score"], 0.1)
    stone_at(s, 0.0009)
    check("stone 0.9 mm high (table over 2 mm above the rim): not home", step(wax, s, 3)["score"], 0.1)
    stone_at(s, 0.0004)
    check("stone 0.4 mm high: home", step(wax, s, 3)["score"], 0.2)
    stone_at(s, g.SEAT_TRAVEL, off=0.002)
    check("stone 2 mm off the head's axis: not in the head", step(wax, s, 3)["score"], 0.0)


def test_put_back():
    for label, change, want in (("pen still in the hand", lambda s: pen_away(s), 0.8),
                                ("tweezers left lying on the table", lambda s: s.update(tweezers=w.Pose(np.array([0.30, -0.20, 0.004]), np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))), 0.8),
                                ("pusher 2 cm from its hole", lambda s: s.update(pusher=w.Pose(w.PUSHER_START.pos + np.array([0.02, 0.0, 0.0]))), 0.8),
                                ("mandrel lifted 1 cm off its rests", lambda s: (s.update(mandrel=w.Pose(w.MANDREL_START.pos + np.array([0.0, 0.0, 0.01]))), stone_at(s, 0.0)), 0.8),
                                ("a gripper 3 cm from the mandrel", lambda s: s.update(fingers=dict(FAR, left=finger_pair(s["mandrel"].world((0.0, g.GRASP_Y, 0.03)), half=0.03))), 0.8),
                                ("things still moving", lambda s: s.update(at_rest=False), 0.8)):
        wax, s = fresh()
        all_done(wax, s)
        pen_away(s, in_stand=True)
        change(s)
        check(f"all four closed, but {label}", step(wax, s, 3)["score"], want)
    # the 5 cm is measured between surfaces: a finger's face to the pen's handle
    for gap, want in ((0.052, True), (0.048, False)):
        wax, s = fresh()
        out = g.PEN_HANDLE_R + w.FINGER_HALF + gap
        s["fingers"] = dict(FAR, right=finger_pair(s["pen"].world((-out - 0.02, 0.0, 0.12)), half=0.02))  # on the side away from the tool block
        check(f"a finger {gap * 1000:.0f} mm from the pen's handle: clear of it", step(wax, s)["clear"], want)


# -- the wax rule's red-team cases (prompt 00, 8.2) ---------------------------------------------------------------------------


def pressed():
    wax, s = fresh()
    stone_at(s, 0.0)
    s["seat"] = -g.SEAT_TRAVEL
    step(wax, s, 2)
    return wax, s


def test_red_team():
    # hover without touching
    wax, s = pressed()
    pen_on_claw(s, 0, 0.0, off=0.001)
    step(wax, s, round(10.0 / DT))
    check("hover: the tip 1 mm off a claw for 10 s adds no heat", wax.heat[0], 0.0)
    check("hover: the claw stays cold", wax.soft[0], False)
    check("hover: score", wax.score, 0.2)

    # touch without the dwell: the cold claw springs back
    wax, s = pressed()
    pen_on_claw(s, 0, 0.0)
    step(wax, s, round(1.0 / DT))
    s["claws"][0] = math.radians(12.0)  # pushed, still cold
    pen_on_claw(s, 0, s["claws"][0])
    step(wax, s, 4)
    check("no dwell: the claw is not soft after 1.2 s", wax.soft[0], False)
    check("no dwell: its drive still rests where it was moulded", wax.target[0], 0.0)
    s["claws"][0] = 0.0  # the tip leaves, the claw springs back
    pen_away(s)
    step(wax, s, 20)
    check("no dwell: score", wax.score, 0.2)

    # forced cold: it breaks
    wax, s = pressed()
    s["claws"][1] = math.radians(w.COLD_BREAK_DEG + 3.0)
    step(wax, s)
    check("forced 28 degrees cold: broken", wax.broken[1], True)
    check("a broken claw scores 0", wax.score, 0.0)

    # contact in a single physics substep per control step
    wax, s = pressed()
    for _ in range(round(10.0 / DT)):
        pen_on_claw(s, 2, 0.0)
        s["dt"] = 1.0 / 240.0
        step(wax, s)
        pen_on_claw(s, 2, 0.0, off=0.003)
        s["dt"] = DT - 1.0 / 240.0
        step(wax, s)
    check("one substep of contact per step, for 10 s: never soft", wax.heat[2] < 0.1 and not wax.soft[2], True)

    # the right action at the wrong place
    wax, s = pressed()
    reach = g.RING_OUT_R + g.PEN_TIP_R
    pen_ball_at(s, s["mandrel"].world((reach * math.sin(math.radians(60.0)), 0.0, reach * math.cos(math.radians(60.0)))))
    step(wax, s, round(1.0 / DT))
    check("pen on the shank for 1 s: melt mark", wax.mark, True)
    for i in range(g.N_CLAWS):
        close_claw(wax, s, i)
    pen_away(s, in_stand=True)
    check("melt mark, then everything done: capped", step(wax, s, 3)["score"], w.MARK_CAP)

    wax, s = pressed()
    pen_ball_at(s, s["stone"].world((0.0, 0.0, g.TABLE_OVER_GIRDLE + g.PEN_TIP_R)))
    step(wax, s, round(0.3 / DT))
    check("pen on the stone's table for 0.3 s: no mark yet", wax.mark, False)
    step(wax, s, round(0.4 / DT))
    check("pen on the stone's table for 0.7 s: melt mark", wax.mark, True)

    # the right actions in the wrong order
    wax, s = fresh()
    stone_at(s, g.SEAT_TRAVEL)
    step(wax, s, 2)
    for i in range(g.N_CLAWS):
        close_claw(wax, s, i, to=g.CLAW_CLOSED + math.radians(4.0))
    check("claws closed over an unpressed stone", sum(wax.closed), 4)
    stone_at(s, 0.0)
    s["seat"] = -g.SEAT_TRAVEL
    pen_away(s, in_stand=True)
    check("then pressed: no claw counts", step(wax, s, 5)["score"], 0.2)
    close_claw(wax, s, 0)
    check("one of them warmed and closed again, now over a pressed stone: it counts", wax.score, 0.35)


def test_latches():
    wax, s = pressed()
    pen_on_claw(s, 0, 0.0)
    step(wax, s, round(4.9 / DT))
    check("pen on a claw 4.9 s: not melted", wax.melted[0], False)
    step(wax, s, round(0.2 / DT))
    check("pen on a claw 5.1 s: melted", wax.melted[0], True)
    check("a melted claw scores 0", wax.score, 0.0)
    for i in (1, 2, 3):
        close_claw(wax, s, i)
    pen_away(s, in_stand=True)
    check("melted claw, then the rest done perfectly: still 0", step(wax, s, 3)["score"], 0.0)

    wax, s = fresh()
    s["fingers"] = dict(FAR, right=finger_pair(s["stone"].world((0.0, 0.0, 0.004)), half=0.0085))
    step(wax, s)
    check("fingers pinch the stone", wax.fingered, True)
    all_done(wax, s)
    pen_away(s, in_stand=True)
    check("fingers on the stone, then everything done with the tools: 0", step(wax, s, 3)["score"], 0.0)

    wax, s = pressed()
    s["fingers"] = dict(FAR, right=finger_pair(s["mandrel"].world((0.0, 0.0, g.Z_S + 0.004)), half=0.012))
    step(wax, s)
    check("fingers on the wax claws", wax.fingered, True)

    wax, s = fresh()
    s["fingers"] = dict(FAR, left=finger_pair(s["mandrel"].world((0.0, g.GRASP_Y, 0.0)), half=g.stick_r(g.GRASP_Y) + 0.0045))
    step(wax, s, 5)
    check("a hand holding the mandrel's thick end touches no wax", wax.fingered, False)

    wax, s = pressed()
    s["pen"] = w.Pose(np.array([0.30, -0.20, 0.008]), np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))  # lying on the table
    s["fingers"] = dict(FAR)
    step(wax, s, round(1.9 / DT))
    check("pen lying on the table for 1.9 s", wax.pen_down, False)
    step(wax, s, round(0.2 / DT))
    check("pen lying on the table for 2.1 s: laid down", wax.pen_down, True)
    check("pen laid down scores 0", wax.score, 0.0)

    wax, s = pressed()
    pen_away(s)
    step(wax, s, round(6.0 / DT))
    check("pen carried in a hand for 6 s is not laid down", wax.pen_down, False)

    wax, s = pressed()
    s["stone"] = w.Pose(np.array([0.5, 0.0, -0.05]))
    step(wax, s)
    check("stone off the table: 0", wax.score, 0.0)
    stone_at(s, 0.0)
    check("stone put back afterwards: still 0", step(wax, s, 3)["score"], 0.0)

    # at rest: read from where the parts have been over the last second
    def trail_of(steps, change=None):
        out = []
        for k in range(steps):
            poses = {n: w.Pose(np.array([0.1, 0.0, 0.02])) for n in ("mandrel", "stone", "wax_pen", "tweezers", "pusher")}
            if change:
                change(k, poses)
            out.append(poses)
        return out

    def turned(k, poses, deg):
        a = math.radians(deg)
        poses["stone"] = w.Pose(poses["stone"].pos, np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]))

    n = round(w.REST_S / DT)
    check("nothing has moved for a second: at rest", w.settled(trail_of(n), DT), True)
    check("only 0.9 s seen so far: not yet", w.settled(trail_of(n - 2), DT), False)
    check("the stone trembles 0.05 mm about its place: at rest", w.settled(trail_of(n, lambda k, p: p["stone"].pos.__iadd__([0.00005 * (-1) ** k, 0.0, 0.0])), DT), True)
    check("a tool was 0.5 mm from here half a second ago: not at rest", w.settled(trail_of(n, lambda k, p: p["pusher"].pos.__iadd__([0.0005 if k == n // 2 else 0.0, 0.0, 0.0])), DT), False)
    check("the stone was turned 1 degree half a second ago: not at rest", w.settled(trail_of(n, lambda k, p: turned(k, p, 1.0) if k == n // 2 else None), DT), False)
    check("a move two seconds ago is forgotten", w.settled(trail_of(3 * n, lambda k, p: p["wax_pen"].pos.__iadd__([0.01 if k < n else 0.0, 0.0, 0.0])), DT), True)

    # the mandrel, pressed on with nobody holding it, tips over its V-block until its thin end is on the table
    def tipped_by(deg):
        """The mandrel turned about the V-block, ring end down."""
        a = math.radians(deg)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
        pivot = w.MANDREL_START.pos + np.array([0.0, g.V_EDGE, 0.0])
        return w.Pose(pivot + rot @ np.array([0.0, -g.V_EDGE, 0.0]), rot)

    down = math.degrees(math.asin((g.AXIS_H - g.STICK_R_THIN) / (g.V_EDGE - g.Y_THIN - g.STICK_R_THIN)))  # the tilt at which the thin end lands
    hand = lambda s: finger_pair(s["mandrel"].world((0.0, g.GRASP_Y, 0.0)), half=g.stick_r(g.GRASP_Y) + 0.0045)  # noqa: E731
    wax, s = pressed()
    check("left alone on its rests, the mandrel has not tipped", step(wax, s, 3)["tipped"], False)
    s["mandrel"] = tipped_by(0.5 * down)
    stone_at(s, 0.0)
    check("unheld and tipped half way: not yet", step(wax, s, 3)["tipped"], False)
    s["mandrel"] = tipped_by(down)
    stone_at(s, 0.0)
    check("unheld and tipped until its thin end is on the table: tipped", step(wax, s)["tipped"], True)
    check("a tipped mandrel scores 0", wax.score, 0.0)
    s["mandrel"] = w.Pose(w.MANDREL_START.pos.copy())
    all_done(wax, s)
    pen_away(s, in_stand=True)
    check("tipped, then back on its rests and everything done: still 0", step(wax, s, 3)["score"], 0.0)
    wax, s = pressed()
    s["mandrel"] = tipped_by(down)
    stone_at(s, 0.0)
    s["fingers"] = dict(FAR, left=hand(s))
    check("the same tilt with a hand on the thick end: not tipped (the hand has it)", step(wax, s, 3)["tipped"], False)
    check("that hand is on the mandrel", w.mandrel_held(s["mandrel"], s["fingers"]), True)
    check("a hand on the pen is not a hand on the mandrel", w.mandrel_held(scene()["mandrel"], dict(FAR, right=finger_pair(w.PEN_START.world((0.0, 0.0, g.PEN_GRASP_Z)), half=0.0115))), False)
    at_least("the thin end's drop before it lands, mm", (g.AXIS_H - g.STICK_R_THIN) * 1000, 15.0)
    at_most("the tilt at which it lands, deg", down, 10.0)

    # nothing is cleared because the parts are back where they started; a real reset clears everything
    wax, s = pressed()
    pen_on_claw(s, 0, 0.0)
    step(wax, s, round(5.2 / DT))
    s2 = scene()
    check("melted, then every part back at its start pose: still latched", step(wax, s2, 10)["melted"], True)
    wax.reset()
    check("after a real reset: cleared", step(wax, scene(), 3)["melted"], False)
    check("after a real reset: score", wax.score, 0.0)


def test_seat_and_claw_drives():
    wax, s = fresh()
    stone_at(s, g.SEAT_TRAVEL)
    s["seat"] = -0.5 * w.SEAT_YIELD
    step(wax, s)
    check("seat pressed half its yield: the drive has not moved", wax.seat_target, 0.0)
    s["seat"] = -0.0010
    step(wax, s)
    check("seat pressed 1.0 mm: the drive follows to within the yield", wax.seat_target, -0.0010 + w.SEAT_YIELD, 1e-12)
    s["seat"] = -0.0010 + w.SEAT_YIELD
    step(wax, s, 5)
    check("pusher gone: the seat stays down", wax.seat_target, -0.0010 + w.SEAT_YIELD, 1e-12)
    at_least("a pressed stone springs back by less than the home rule allows, mm to spare", (w.HOME_UP - w.SEAT_YIELD) * 1000, 0.4)
    wax, s = pressed()
    pen_on_claw(s, 3, 0.0)
    step(wax, s, round(1.7 / DT))
    check("a claw under the pen for 1.7 s is soft", wax.soft[3], True)
    s["claws"][3] = math.radians(40.0)
    pen_on_claw(s, 3, s["claws"][3])
    step(wax, s)
    check("a soft claw's drive goes where the claw is pushed", wax.target[3], math.radians(40.0), 1e-12)
    wax, s = pressed()
    close_claw(wax, s, 3)
    check("a closed claw has set where it was left", wax.target[3], g.CLAW_ON_CROWN, 1e-12)
    check("and is no longer soft", wax.soft[3], False)
    at_least("where the demo leaves a claw, past the 1 mm line by, deg", math.degrees(g.CLAW_ON_CROWN - g.CLAW_CLOSED) - task.STOP_SHORT_DEG, 5.0)
    check("a soft claw's drag at 60 degrees a second, as a force on its tip (N)", round(w.SOFT_DAMP * math.radians(60.0) / g.CLAW_H, 3), 0.01)
    check("force on a cold claw's tip that breaks it (N)", round(w.COLD_RATE * math.radians(w.COLD_BREAK_DEG) / g.CLAW_H, 2), 0.87)
    at_least("the demo rests its tip on a claw longer than the wax takes to soften, s to spare", task.WAIT_S - w.SOFT_AFTER_S, 0.2)
    at_least("and takes the claw round before its longest touch ends, s to spare", task.LONGEST_S - task.WAIT_S - (math.degrees(g.CLAW_ON_CROWN) - task.STOP_SHORT_DEG) / task.TURN_RATE, 0.5)
    at_least("the longest touch ends before the wax melts, s to spare", w.MELT_AFTER_S - task.LONGEST_S, 0.75)


# -- the bench's geometry ---------------------------------------------------------------------------------------------------


def stands_out(points: np.ndarray, k: int, n: int = 4000) -> float:
    """A lower bound on how far corner k stands out of the convex hull of the other corners (the best of n directions)."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n, 3))
    u = np.vstack([u / np.linalg.norm(u, axis=1, keepdims=True), points[k] / max(np.linalg.norm(points[k]), 1e-12)])
    others = np.delete(points, k, axis=0)
    return float(np.max(u @ points[k] - np.max(u @ others.T, axis=1)))


def test_geometry():
    check("ring's place on the stick, mm from its thin end", round(g.RING_FROM_THIN * 1000), 142)
    check("stone's mass, g", round(g.STONE_MASS * 1000, 2), 0.63)
    check("claws stand open: mm between opposite tips", round(2 * (g.CLAW_RH + g.CLAW_H * math.sin(math.radians(g.CLAW_LEAN_DEG))) * 1000, 1), 9.2)
    check("a claw is closed 1 mm over the stone's edge at, deg", round(math.degrees(g.CLAW_CLOSED), 1), 47.4)
    at_least("a claw lies on the crown later than the 1 mm line, deg", math.degrees(g.CLAW_ON_CROWN - g.CLAW_CLOSED), 10.0)
    check("overlap as it lies on the crown, mm", round(g.claw_overlap(g.CLAW_ON_CROWN) * 1000, 1), 1.5)
    check("pressed home, table above the rim, mm", round((g.TABLE_OVER_GIRDLE + g.GIRDLE_H / 2) * 1000, 2), 1.44)
    check("before pressing, table above the rim, mm", round((g.TABLE_OVER_GIRDLE + g.GIRDLE_H / 2 + g.SEAT_TRAVEL) * 1000, 2), 2.94)
    # the stone on the bearing has its girdle on the seat line; on the unpressed seat it stands SEAT_TRAVEL higher
    check("bearing: a stone resting on it is home", round((g.BEAR_Z + g.pavilion_drop(g.BEAR_R) - g.Z_S) * 1e6), 0)
    check("seat: a stone resting on it stands proud by the travel", round((g.SEAT_TOP0 + g.pavilion_drop(g.SEAT_R) - g.Z_S) * 1e6), round(g.SEAT_TRAVEL * 1e6))
    at_least("culet clears the shank, mm", (g.Z_S - g.GIRDLE_H / 2 - g.PAV_H - g.RING_OUT_R) * 1000, 0.4)
    # room for the stone and the tweezers between the open claws
    tip_r = g.CLAW_RH + g.CLAW_H * math.sin(math.radians(g.CLAW_LEAN_DEG))
    at_least("stone to an open claw's tip, mm a side", (tip_r - g.STONE_R) * 1000, 0.5)
    centre, rot, half = g.claw_box(0, 0.0)
    corners = [centre + rot @ (np.array([sx, sy, sz]) * half) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    at_least("tweezer tip (along x) to the nearest claw, mm", (min(abs(c[1]) for c in corners) - g.TW_TIP_W / 2) * 1000, 1.0)
    # the pen's ball as it rides a claw round: on the claw, clear of the head and of the stone
    stone = w.Pose(np.array([0.0, 0.0, g.Z_S]))
    worst_cup, worst_stone, worst_on = 1.0, 1.0, 0.0
    for k in range(0, 69):
        a = math.radians(k)
        b = task.claw_ball(0, a)
        worst_on = max(worst_on, abs(g.box_gap(b, *g.claw_box(0, a)) - g.PEN_TIP_R))
        worst_cup = min(worst_cup, g.wax_body_gap(b) - g.PEN_TIP_R)
        worst_stone = min(worst_stone, g.stone_gap(stone.local(b)) - g.PEN_TIP_R)
    check("the ball rides on the claw's face all the way round (mm off)", round(worst_on * 1000, 3), 0.0)
    at_least("the ball's gap to the head along the way, mm (a melt mark is 0.25)", worst_cup * 1000, 1.0)
    at_least("the ball's gap to the stone along the way, mm", worst_stone * 1000, 0.7)
    at_least("the ball sits below the claw's tip by, mm", (g.CLAW_H - task.PUSH_FACE[1]) * 1000, 0.4)
    # the stone's collider: the true stone where it is carried, and corners that hull cooking will keep
    hull = g.stone_hull()
    check("stone collider: 13 corners", len(hull), 13)
    at_least("stone collider: every corner stands out of the rest by, mm", min(stands_out(hull, k) for k in range(len(hull))) * 1000, 0.9)
    flat = hull[1:3].mean(axis=0)  # the middle of the girdle's flat that faces a claw
    check("stone collider: the girdle's flats are at the stone's radius", round(float(np.hypot(flat[0], flat[1])) * 1e6), round(g.STONE_R * 1e6))
    side = (hull[8] + hull[1]) / 2  # the middle of the flat that faces along +x, where the seat, the bearing and the tweezers meet it
    check("stone collider: along x the pavilion is the true cone", round((abs(hull[0][2] - side[2]) / side[0] - g.PAV_SLOPE) * 1e6), 0)
    at_least("the head's rim blocks stand clear of the stone collider's corners, mm", (g.HEAD_WALL_R - float(np.hypot(*hull[1][:2]))) * 1000, 0.1)
    # the stick on its supports, and why it must be held
    at_least("the stick's lowest point above the table at the V, mm", (g.AXIS_H - g.stick_r(g.V_Y)) * 1000, 10.0)
    check("the rest's top is under the stick, within 0.1 mm", abs(g.AXIS_H - g.stick_r(g.REST_Y) - g.REST_SIZE[2]) < 0.0001, True)
    weight = (g.STICK_MASS + g.RING_MASS) * 9.81
    at_least("left alone, the stick lies steady: its weight sits toward the rest from the V's near edge by, mm", (g.STICK_COM_Y - g.V_EDGE) * 1000, 8.0)
    tips_at = weight * (g.STICK_COM_Y - g.V_EDGE) / g.V_EDGE  # N on the ring that lifts the thick end off its rest: the stick turns about the V's edge nearest the ring
    at_least("unheld, the stick tips under the press: the seat's yield over the push that tips it", w.SEAT_HOLD_N / tips_at, 2.0)
    at_least("the stone alone does not tip it: the push that tips it over the stone's weight", tips_at / (g.STONE_MASS * 9.81), 20.0)
    at_least("room between the holding hand's fingers and the rest, mm", (g.GRASP_Y - 0.018 - g.REST_Y - g.REST_SIZE[1] / 2) * 1000, 15.0)
    at_least("room between the ring and the V-block, mm", (g.V_Y - g.V_SIZE[1] / 2 - g.SHANK_W / 2) * 1000, 8.0)
    # tools fit inside the gripper: no more than 56 mm of a tool above the pads' centre (the fingers' cross-bar is 60 mm up)
    for name, top, grasp in (("tweezers", g.TW_LEN, g.TW_GRASP_Z), ("pusher", g.PU_LEN, g.PU_GRASP_Z), ("pen", g.PEN_LEN, g.PEN_GRASP_Z)):
        check(f"{name}: no more than 56 mm above the grasp", top - grasp <= 0.056, True)
    # every tool in its place
    check("tweezers: their tips stand on the trough's floor", abs(g.TW_REST_Z - (g.TOOL_BLOCK[2] - g.TWEEZER_SLOT[2])) < 1e-9, True)
    floor_play = (g.TWEEZER_SLOT[1] - g.TW_TIP_W) / 2
    leg_play = (g.TWEEZER_SLOT[1] + 2 * g.TROUGH_RUN * g.TW_TIP_H - g.TW_LEG_W) / 2  # between a wall and a leg where the leg's full width begins
    at_least("tweezers: tips' play across the trough's floor, mm a side", floor_play * 1000, 0.25)
    at_least("tweezers: the walls' play at the legs, mm a side", leg_play * 1000, 0.25)
    at_most("tweezers: the most they can lean across the trough, deg", math.degrees(math.atan((floor_play + leg_play) / g.TW_TIP_H)), 4.0)
    span = g.TW_GAP + 2 * g.TW_LEG_T
    at_least("tweezers: tips' play along the trough, mm a side", (g.TWEEZER_SLOT[0] - span) / 2 * 1000, 0.6)
    at_most("tweezers: the most they can lean along the trough, deg", math.degrees(math.atan((g.TWEEZER_SLOT[0] - span) / g.TWEEZER_SLOT[2])), w.TOOL_HOME[2] / 2)
    at_least("tweezers: the trough's top takes tips that are off by, mm", (g.TROUGH_TOP - g.TW_TIP_W) / 2 * 1000, 4.0)
    at_least("tweezers: and legs that are off by, mm", (g.TROUGH_TOP - g.TW_LEG_W) / 2 * 1000, 2.5)
    at_most("tweezers: the walls' lean from upright, deg (a tool slides down them)", math.degrees(math.atan(g.TROUGH_RUN)), 15.0)
    at_least("tweezers: the block's floor under the trough, mm", (g.TOOL_BLOCK[2] - g.TWEEZER_SLOT[2]) * 1000, 3.0)
    at_least("pusher: play in its hole, mm a side", (g.PUSHER_HOLE[0] - g.PU_FACE) / 2 * 1000, 0.4)
    at_least("pen: its handle's play in the stand, mm a side", (g.PEN_BORE / 2 - g.PEN_HANDLE_R) * 1000, 0.4)
    at_least("pen: its handle's seat on the ledge, mm a side", (g.PEN_HANDLE_R - g.PEN_TIP_BORE / 2) * 1000, 2.0)
    at_least("pen: its tip hangs clear of the base, mm", (g.PEN_REST_Z - g.PEN_BASE[2]) * 1000, 0.9)
    at_least("a mouth catches a tool that is off by, mm", (g.MOUTH_RISE - 0.001) * 1000, 4.0)


# -- the props' USD -----------------------------------------------------------------------------------------------------------


def prim_paths(text: str) -> set:
    out, stack, pending = set(), [], None
    for mt in re.finditer(r'def\s+\w+\s+"([^"]+)"|[{}]', text):
        tok = mt.group(0)
        if tok == "{":
            stack.append(pending)
            if pending is not None:
                out.add("/" + "/".join(n for n in stack if n))
            pending = None
        elif tok == "}":
            stack.pop()
        else:
            pending = mt.group(1)
    return out


def test_usd():
    rng = np.random.default_rng(1)
    good = True
    for _ in range(200):  # the quaternion mechkit writes for a turned box is the rotation it was given
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        rot = m.quat_matrix((q[1], q[2], q[3], q[0]))
        qw, qx, qy, qz = m._quat_wxyz(rot)
        good = good and bool(np.allclose(m.quat_matrix((qx, qy, qz, qw)), rot, atol=1e-9))
    check("rotation to quaternion and back, 200 rotations", good, True)
    texts = {o.name: o.build() for o in task.OBJECTS}
    for name, text in texts.items():
        check(f"{name}: braces balance", text.count("{") == text.count("}"), True)
        quats = [tuple(float(v) for v in q.split(",")) for q in re.findall(r"quatf xformOp:orient = \(([^)]+)\)", text)]
        check(f"{name}: its {len(quats)} turned boxes carry unit quaternions", all(abs(sum(v * v for v in q) - 1.0) < 1e-6 for q in quats), True)
    paths = prim_paths(texts["mandrel"])
    for i in range(g.N_CLAWS):
        for p in (f"/P_mandrel/claw{i}/visual/cold", f"/P_mandrel/claw{i}/visual/soft", f"/P_mandrel/frame/visual/blob{i}", f"/P_mandrel/claw{i}_joint", f"/P_mandrel/claw{i}/claw{i}_c"):
            check(f"mandrel has {p}", p in paths, True)
    check("mandrel has its seat and the seat's joint", {"/P_mandrel/seat", "/P_mandrel/seat_joint"} <= paths, True)
    joints = re.findall(r"quatf physics:localRot0 = \(([^)]+)\)", texts["mandrel"])
    check("five joints", len(joints), 5)
    for i, q in enumerate(joints[:4]):
        qw, qx, qy, qz = (float(v) for v in q.split(","))
        check(f"claw {i}: joint frame is a unit quaternion", abs(qw * qw + qx * qx + qy * qy + qz * qz - 1.0) < 1e-6, True)
        axis = np.array([qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)])  # where the frame's X points
        check(f"claw {i}: the joint's X is the claw's hinge axis", bool(np.allclose(axis, g.claw_frame(i)[1][:, 1], atol=1e-6)), True)
        block = re.search(rf'def Cube "claw{i}_c".*?xformOp:translate = \(([^)]+)\).*?xformOp:orient = \(([^)]+)\).*?xformOp:scale = \(([^)]+)\)', texts["mandrel"], re.S)
        at, (ow, ox, oy, oz), size = (np.array([float(v) for v in block.group(k).split(",")]) for k in (1, 2, 3))
        centre, rot, half = g.claw_box(i, 0.0)
        box = m.quat_matrix((ox, oy, oz, ow))
        same = np.allclose(at, centre, atol=2e-6) and np.allclose(size, 2 * half, atol=2e-6) and np.allclose(np.abs(box.T @ rot), np.eye(3), atol=1e-5)
        check(f"claw {i}: its collider is the claw the rule measures", bool(same), True)
    check("the mandrel's centre of mass is written", f"physics:centerOfMass = (0.000000, {g.STICK_COM_Y:.6f}, 0.000000)" in texts["mandrel"], True)
    colliders = len(re.findall(r'"PhysicsCollisionAPI"', texts["mandrel"]))
    check("mandrel: 5 rods, 12 shank bars, 12 rim blocks, 4 bearing blocks, 4 claws, 4 seat blocks", colliders, 5 + 12 + 12 + 4 + 4 + 4)
    check("mandrel: its colliders, four soft looks and four puddles are invisible at the start", len(re.findall(r'token visibility = "invisible"', texts["mandrel"])), colliders + 8)
    paths = prim_paths(texts["tweezers"])
    check("tweezers have two legs on joints", {"/P_tweezers/leg_a", "/P_tweezers/leg_b", "/P_tweezers/leg_a_joint", "/P_tweezers/leg_b_joint", "/P_tweezers/leg_a/tip_c"} <= paths, True)
    for name in ("stone", "pusher", "wax_pen"):
        check(f"{name} is one free body", task.OBJECTS[[o.name for o in task.OBJECTS].index(name)].role, "free")
    hulls = {name: text.count('"convexHull"') for name, text in texts.items()}
    check("convex hulls only where a few tenths of a millimetre do not matter", hulls, {"bench": 0, "mandrel": 0, "stone": 1, "tweezers": 0, "pusher": 3, "wax_pen": 1})


# -- the demo's reach -----------------------------------------------------------------------------------------------------------


def test_reach():
    import demokit as d
    from yam_arena import YAM_URDF, analysis, ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, READY
    from yam_arena.urdf import forward_kinematics

    top = station.TABLE_TOP
    ring = np.array([*g.RING_XY, g.AXIS_H])
    head = ring + np.array([0.0, 0.0, g.Z_S])
    mouth = g.TOOL_BLOCK[2] + g.MOUTH_RISE + 0.003
    lane = np.array([0.0, task.LANE, 0.0])
    wide = -FINGER_OPEN
    onto, off = task.FINGER_ROOM
    tw, pu, pen = np.array([*g.TWEEZER_XY, g.TW_REST_Z + g.TW_GRASP_Z]), np.array([*g.PUSHER_XY, g.PU_REST_Z + g.PU_GRASP_Z]), np.array([*g.PEN_STAND_XY, g.PEN_REST_Z + g.PEN_GRASP_Z])
    up = lambda h: np.array([0.0, 0.0, h])  # noqa: E731
    legs = (g.TW_ROOT + (g.TW_GAP - g.TW_ROOT) * (1.0 - g.TW_GRASP_Z / g.TW_HINGE_Z) + 2 * g.TW_LEG_T) / 2
    pen_over = np.array([*g.PEN_STAND_XY, g.PEN_TUBE[2] + g.MOUTH_RISE + 0.003 + g.PEN_GRASP_Z])
    # the working hand's way through the demo: (what, TCP above the table, half the gap between its fingers, the tools standing in their places meanwhile,
    # and which of them it is taking or leaving)
    everything = ("tweezers", "pusher", "pen")
    way = [("out over the bench", np.array(task.OUT_OVER), wide, everything, ""),
           ("beside the tweezers", tw + lane + up(0.03), legs + onto, everything, ""), ("level with the tweezers", tw + lane, legs + onto, everything, "tweezers"),
           ("on the tweezers", tw, legs + onto, everything, "tweezers"), ("tweezers out of their trough", tw + up(0.042), legs, ("pusher", "pen"), ""),
           ("tweezers over the stone", [*g.STONE_BLOCK_XY, g.STONE_BLOCK[2] + 0.014 + g.TW_GRASP_Z], legs, ("pusher", "pen"), ""),
           ("tips on the stone's block", [*g.STONE_BLOCK_XY, g.STONE_BLOCK[2] + g.TW_GRASP_Z], legs, ("pusher", "pen"), ""),
           ("stone up", [*g.STONE_BLOCK_XY, g.STONE_BLOCK[2] + 0.035 + g.TW_GRASP_Z], legs, ("pusher", "pen"), ""),
           ("stone over the ring", head + up(g.SEAT_TRAVEL + 0.02 + g.TW_GRASP_Z), legs, ("pusher", "pen"), ""),
           ("stone between the claws", head + up(g.SEAT_TRAVEL + g.TW_GRASP_Z), legs, ("pusher", "pen"), ""),
           ("tweezers up from the ring", head + up(g.SEAT_TRAVEL + 0.03 + g.TW_GRASP_Z), legs, ("pusher", "pen"), ""),
           ("tweezers in the lane", [g.TWEEZER_XY[0], g.TWEEZER_XY[1] + task.LANE, mouth + g.TW_GRASP_Z], legs, ("pusher", "pen"), ""),
           ("tweezers over their trough", [*g.TWEEZER_XY, mouth + g.TW_GRASP_Z], legs, ("pusher", "pen"), ""), ("tweezers down", tw, legs, ("pusher", "pen"), ""),
           ("away from the tweezers", tw + lane, legs + off, everything, "tweezers"),
           ("beside the pusher", pu + lane + up(0.03), g.PU_KNOB_R + onto, everything, ""), ("level with the pusher", pu + lane, g.PU_KNOB_R + onto, everything, "pusher"),
           ("on the pusher", pu, g.PU_KNOB_R + onto, everything, "pusher"), ("pusher out of its hole", pu + up(0.035), g.PU_KNOB_R, ("tweezers", "pen"), ""),
           ("pusher over the stone", head + up(g.TABLE_OVER_GIRDLE + g.SEAT_TRAVEL + 0.02 + g.PU_GRASP_Z), g.PU_KNOB_R, ("tweezers", "pen"), ""),
           ("pusher on the stone, home", head + up(g.TABLE_OVER_GIRDLE + g.PU_GRASP_Z), g.PU_KNOB_R, ("tweezers", "pen"), ""),
           ("pusher up from the ring", head + up(g.TABLE_OVER_GIRDLE + 0.03 + g.PU_GRASP_Z), g.PU_KNOB_R, ("tweezers", "pen"), ""),
           ("pusher in the lane", [g.PUSHER_XY[0], g.PUSHER_XY[1] + task.LANE, mouth + g.PU_GRASP_Z], g.PU_KNOB_R, ("tweezers", "pen"), ""),
           ("pusher over its hole", [*g.PUSHER_XY, mouth + g.PU_GRASP_Z], g.PU_KNOB_R, ("tweezers", "pen"), ""), ("pusher down", pu, g.PU_KNOB_R, ("tweezers", "pen"), ""),
           ("away from the pusher", pu + lane, g.PU_KNOB_R + off, everything, "pusher"),
           ("beside the pen", pen + lane + up(0.03), g.PEN_HANDLE_R + onto, everything, ""), ("level with the pen", pen + lane, g.PEN_HANDLE_R + onto, everything, "pen"),
           ("on the pen", pen, g.PEN_HANDLE_R + onto, everything, "pen"), ("pen out of its stand", pen + up(0.047), g.PEN_HANDLE_R, ("tweezers", "pusher"), ""),
           ("pen toward the ring", head + [0.0, -0.040, 0.011 - g.PEN_TIP_R + g.PEN_GRASP_Z], g.PEN_HANDLE_R, ("tweezers", "pusher"), "")]
    for i in task.CLAW_ORDER:
        out = task.claw_ball(i, 0.0) + task.APPROACH[0] * g.claw_frame(i)[1][:, 0]
        way += [(f"pen above claw {i}", ring + out + up(task.APPROACH[1] - g.PEN_TIP_R + g.PEN_GRASP_Z), g.PEN_HANDLE_R, ("tweezers", "pusher"), ""),
                (f"pen beside claw {i}", ring + out + up(-g.PEN_TIP_R + g.PEN_GRASP_Z), g.PEN_HANDLE_R, ("tweezers", "pusher"), ""),
                (f"pen on claw {i}, closed, 8 mm of push in the arm", ring + task.claw_ball(i, g.CLAW_ON_CROWN) - 0.008 * task.claw_out(i, g.CLAW_ON_CROWN) + up(-g.PEN_TIP_R + g.PEN_GRASP_Z),
                 g.PEN_HANDLE_R, ("tweezers", "pusher"), ""),
                (f"pen off claw {i}", ring + task.claw_ball(i, g.CLAW_ON_CROWN) + task.APPROACH[0] * g.claw_frame(i)[1][:, 0] + up(task.APPROACH[1] - g.PEN_TIP_R + g.PEN_GRASP_Z), g.PEN_HANDLE_R,
                 ("tweezers", "pusher"), "")]
    way += [("pen in the lane", pen_over + lane, g.PEN_HANDLE_R, ("tweezers", "pusher"), ""), ("pen over its stand", pen_over, g.PEN_HANDLE_R, ("tweezers", "pusher"), ""),
            ("pen down", pen, g.PEN_HANDLE_R, ("tweezers", "pusher"), ""), ("away from the pen", pen + lane, g.PEN_HANDLE_R + off, everything, "pen"),
            ("the right hand away", np.array(task.RIGHT_AWAY), g.PEN_HANDLE_R + off, everything, "")]
    poses = {
        "left": [("over the mandrel's thick end", ring + [0.0, g.GRASP_Y, 0.07]), ("on the mandrel", ring + [0.0, g.GRASP_Y, 0.0]), ("up", ring + [0.0, g.GRASP_Y, 0.08]),
                 ("away", ring + [-0.06, g.GRASP_Y + 0.10, 0.08])],
        "right": [(label, p) for label, p, *_ in way],
    }
    grips = {"left": d.tilted(task.HOLD_TILT_DEG, -90), "right": d.down(90)}
    at_work = {}  # each arm's joints at every key pose
    print(f"    highest working TCP for the right hand: {max(np.asarray(p[1])[2] for p in poses['right']) * 1000:.0f} mm above the table")
    ends = {}
    for side in ("left", "right"):
        q = np.array(READY, float)
        worst_mm, least = 0.0, 180.0
        for label, p in poses[side]:
            target = np.asarray(p, float) + [0.0, 0.0, top]
            q, pe, re_ = ik.solve(target, ik.matrix_to_quat_xyzw(grips[side].rot), q, base_pos=np.asarray(station.BASES[side], float), iters=300)
            margin = float(np.degrees(np.min(np.minimum(q - ik.LOWER, ik.UPPER - q))))
            worst_mm, least = max(worst_mm, pe * 1000), min(least, margin)
            at_work[side, label] = q.copy()
            if pe * 1000 > 0.5 or re_ > 0.01 or margin < 4.0:
                print(f"    {side} '{label}': IK {pe * 1000:.2f} mm, {math.degrees(re_):.2f} deg, joint margin {margin:.1f} deg")
        check(f"{side} arm: IK error under 0.5 mm at every key pose", worst_mm < 0.5, True)
        at_least(f"{side} arm: least joint margin over the key poses, deg", least, 4.0)
        pos, rot = ik.tcp_pose(q, np.asarray(station.BASES[side], float))
        origin = pos - rot @ np.array([0.0, 0.0, 0.134]) - [0.0, 0.0, top]
        ends[side] = tuple(np.array([origin + rot @ np.array([sx * (0.04695 + 0.0045), dy, dz]) for dy in task.FINGER_DY for dz in task.FINGER_DZ]) for sx in (-1.0, 1.0))
    # where the demo leaves its hands: both clear of everything, fingers open
    start = {"pen": w.PEN_START, "tweezers": w.TWEEZERS_START, "pusher": w.PUSHER_START, "mandrel": w.MANDREL_START, "stone": w.Pose(w.MANDREL_START.world((0.0, 0.0, g.Z_S)))}
    gap = w.clearance(np.vstack([p for pair in ends.values() for p in pair]), start)
    at_least("the demo's end poses: clear of everything by, mm", gap * 1000, w.CLEAR * 1000 + 10.0)

    # the robot's own collision hulls, from its description
    def hulls(side, q, finger):
        joints = {**{n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}, **{n: finger for n in FINGER_JOINTS}}
        links = {k: np.r_[np.c_[v[:3, :3], v[:3, 3] + np.asarray(station.BASES[side], float)], [[0.0, 0.0, 0.0, 1.0]]] for k, v in forward_kinematics(YAM_URDF, joints).items()}
        return analysis.world_hulls(links)

    # the hand that holds the mandrel and the hand that works at the ring
    holding = hulls("left", at_work["left", "on the mandrel"], -g.stick_r(g.GRASP_Y))
    near_ring = [k for k in at_work if k[0] == "right" and any(word in k[1] for word in ("claw", "ring", "stone between", "pusher on the stone", "pusher over the stone"))]
    nearest = min((analysis.hull_set_distance(hulls("right", at_work[k], -0.007), holding)[0], k[1]) for k in near_ring)
    print(f"    {len(near_ring)} poses of the working hand at the ring; the arms come nearest at '{nearest[1]}'")
    at_least("the two arms keep clear of each other while one works at the ring, mm (50 or more counts as 50)", min(nearest[0] * 1000, 50.0), 30.0)

    # the working arm and the tools standing in their places, along the whole way: each tool as points up its axis, with its radius
    def column(xy, rest, parts):
        """Balls up a tool's axis: each part is (from, to, radius), heights above its lower end; the top ball lies its radius under the part's top."""
        return [(np.array([[xy[0], xy[1], top + rest + z] for z in np.arange(z0, z1 - r + 1e-9, 0.003)]), r) for z0, z1, r in parts]

    leg = lambda z: g.TW_ROOT / 2 + g.TW_LEG_T + (g.TW_GAP - g.TW_ROOT) / 2 * (1 - z / g.TW_HINGE_Z)  # noqa: E731  (the outside of a leg, from the axis)
    standing = {"pen": column(g.PEN_STAND_XY, g.PEN_REST_Z, [(g.PEN_TIP_LEN, g.PEN_LEN, g.PEN_HANDLE_R)]),
                "tweezers": column(g.TWEEZER_XY, g.TW_REST_Z, [(z, z + 0.015 + leg(z), leg(z)) for z in np.arange(0.03, g.TW_HINGE_Z, 0.015)] + [(g.TW_HINGE_Z, g.TW_LEN, 0.0055)]),
                "pusher": column(g.PUSHER_XY, g.PU_REST_Z, [(0.025, g.PU_SHANK, g.PU_FACE), (g.PU_SHANK, g.PU_LEN, g.PU_KNOB_R)])}
    base = np.asarray(station.BASES["right"], float)
    q = np.array(READY, float)
    prev = ik.tcp_pose(q, base)[0] - [0.0, 0.0, top]
    least_other, least_own = (9.0, ""), (9.0, "")
    for label, p, finger, tools, own in way:
        p = np.asarray(p, float)
        for k in range(1, 9):
            q, _, _ = ik.solve(prev + (p - prev) * k / 8 + [0.0, 0.0, top], ik.matrix_to_quat_xyzw(grips["right"].rot), q, base_pos=base, iters=120)
            arm = hulls("right", q, -finger)
            for name in tools:
                near = min(float(analysis._signed(points, h).min()) - r for points, r in standing[name] for _, h in arm)
                if name == own:
                    least_own = min(least_own, (near, f"{label}: {name}"))
                else:
                    least_other = min(least_other, (near, f"{label}: {name}"))
        prev = p
    print(f"    the working arm comes nearest to a tool it is not taking at '{least_other[1]}', and to one it is taking or leaving at '{least_own[1]}'")
    at_least("the working arm keeps clear of the tools standing in their places, mm", least_other[0] * 1000, 15.0)
    at_least("its open fingers keep clear of the tool they slide onto or away from, mm", least_own[0] * 1000, 4.0)


# -- the task's text against its code -------------------------------------------------------------------------------------------


def test_text():
    path = HERE / "task.json"
    if not path.exists():
        print("    (no task.json yet)")
        return
    t = json.loads(path.read_text())
    check("instruction is the task file's", t["instruction"], task.INSTRUCTION)
    for field, limit in (("title", 120), ("instruction", 1000), ("context", 1000), ("fidelity", 500), ("scores", 300), ("run_limit", 200)):
        check(f"{field} within {limit} characters ({len(t[field])})", len(t[field]) <= limit, True)
    for field in t:
        if isinstance(t[field], str):
            check(f"{field}: no em or en dash", chr(0x2014) in t[field] or chr(0x2013) in t[field], False)
    fid, sc, setup = t["fidelity"], t["scoring"], t["setup"]
    mm = lambda v: f"{v * 1000:g}"  # noqa: E731
    for label, text, phrase in (("soften time", fid, f"{w.SOFT_AFTER_S:g} s"), ("melt time", fid, f"{w.MELT_AFTER_S:g} s"), ("seat force", fid, f"{w.SEAT_HOLD_N:g} N"),
                                ("cold break", fid, f"{w.COLD_BREAK_DEG:g} degrees"), ("mark time", fid, f"{w.MARK_AFTER_S:g} s"), ("tweezer legs", fid, f"{mm(g.TW_LEG_T)} mm"),
                                ("stone weight", sc, f"{w.W_STONE:g} if"), ("home weight", sc, f"{w.W_HOME:g} more"), ("claw weight", sc, f"{w.W_CLAW:g} for each claw"),
                                ("put-back weight", sc, f"{w.W_PUT_BACK:g} more"), ("cap", sc, f"at most {w.MARK_CAP:g}"), ("closed overlap", sc, f"{mm(w.CLOSED_OVERLAP)} mm or more over"),
                                ("home height", sc, f"less than {mm(w.HOME_TABLE_OVER_RIM)} mm above"), ("level", sc, f"within {w.LEVEL_DEG:g} degrees"),
                                ("held", sc, "three or more"), ("clear", sc, f"{w.CLEAR * 100:g} cm or more"), ("off the table", sc, "leaves the table"),
                                ("stick", setup, f"{mm(g.STICK_LEN)} mm"), ("stick mass", setup, f"{mm(g.STICK_MASS)} g"), ("stone size", setup, f"{g.STONE_D * 1000:.1f} mm"),
                                ("stone mass", setup, f"{g.STONE_MASS * 1000:.2f} g"), ("ring place", setup, f"{round(g.RING_FROM_THIN * 1000)} mm from its thin end"),
                                ("claw size", setup, f"{mm(g.CLAW_W)} mm square"), ("claw height", setup, f"{mm(g.CLAW_H)} mm tall"),
                                ("seat travel", setup, f"{mm(g.SEAT_TRAVEL)} mm proud"),
                                ("ring", setup, f"({g.RING_XY[0] * 100:g}, {g.RING_XY[1] * 100:g})"), ("stone block", setup, f"({g.STONE_BLOCK_XY[0] * 100:g}, {g.STONE_BLOCK_XY[1] * 100:g})"),
                                ("tool block", setup, f"({g.TOOL_BLOCK_XY[0] * 100:g}, {g.TOOL_BLOCK_XY[1] * 100:g})"), ("pen stand", setup, f"({g.PEN_STAND_XY[0] * 100:g}, {g.PEN_STAND_XY[1] * 100:g})"),
                                ("V-block place", setup, f"centred {mm(g.V_Y)} mm from the ring"), ("rest place", setup, f"{mm(g.REST_Y)} mm from the ring"),
                                ("V-block size", setup, f"{mm(g.V_SIZE[0])} x {mm(g.V_SIZE[1])} x {mm(g.V_SIZE[2])} mm"), ("tipping rule", sc, "tipped until its thin end touched the table"),
                                ("rest", sc, f"moved {mm(w.REST_SHIFT)} mm or turned {w.REST_TURN_DEG:g} degrees in the last second"),
                                ("tweezers", setup, f"{mm(g.TW_LEN)} mm"), ("pusher", setup, f"{mm(g.PU_LEN)} mm"), ("pen", setup, f"{mm(g.PEN_LEN)} mm"),
                                ("tweezers' place", setup, f"({g.TWEEZER_XY[0] * 100:g}, {g.TWEEZER_XY[1] * 100:g})"), ("pusher's place", setup, f"({g.PUSHER_XY[0] * 100:g}, {g.PUSHER_XY[1] * 100:g})"),
                                ("tool block size", setup, f"{mm(g.TOOL_BLOCK[0])} x {mm(g.TOOL_BLOCK[1])} x {mm(g.TOOL_BLOCK[2])} mm"), ("tweezers' gap", setup, f"tips {mm(g.TW_GAP)} mm apart"),
                                ("trough", setup, f"{mm(g.TWEEZER_SLOT[0])} mm long and {mm(g.TWEEZER_SLOT[2])} mm deep"), ("trough floor", setup, f"floor {mm(g.TWEEZER_SLOT[1])} mm wide"),
                                ("trough top", setup, f"to {g.TROUGH_TOP * 1000:.1f} mm at the block's top"), ("pusher hole", setup, f"{mm(g.PUSHER_HOLE[0])} mm square hole {mm(g.PUSHER_HOLE[2])} mm deep"),
                                ("pen tip", setup, f"tip {mm(2 * g.PEN_TIP_R)} mm across and {mm(g.PEN_TIP_LEN)} mm long"), ("pen bore", setup, f"{mm(g.PEN_BORE)} mm bore"),
                                ("pen stand top", setup, f"top {mm(g.PEN_TUBE[2])} mm up"), ("ring size", setup, f"{2 * g.RING_IN_R * 1000:.1f} mm inside")):
        check(f"text has the code's {label} ({phrase!r})", phrase in text, True)
    steps = int(re.search(r"[\d,]+", t["run_limit"]).group().replace(",", ""))
    check("possible scores listed", "0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.85 or 1.0" in sc, True)
    at_least("run limit, steps", steps, 2400)


TESTS = (test_ladder, test_put_back, test_red_team, test_latches, test_seat_and_claw_drives, test_geometry, test_usd, test_reach, test_text)

if __name__ == "__main__":
    for fn in TESTS:
        print(f"\n== {fn.__name__}")
        fn()
    print(f"\n{len(BAD)} failed" + ("".join(f"\n  BAD {b}" for b in BAD)))
    sys.exit(1 if BAD else 0)
