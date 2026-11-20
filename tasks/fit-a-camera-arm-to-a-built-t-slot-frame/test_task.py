"""Offline tests: the geometry's fits, the thread rule, the score's ladder with its red-team cases, the task file's wiring, and the scripted
demo against the kinematic stand-in in dryrun.py (reach, end states, run length, clearances). No simulator.

    PYTHONPATH=<yam-isaac-lab-arena> python3 test_task.py [--fast]      (--fast leaves out the 1 minute clearance sweep)
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "--quiet" not in sys.argv:
    sys.argv.append("--quiet")

import dryrun  # noqa: E402
import tslot_geometry as g  # noqa: E402
import tslot_rule as r  # noqa: E402

task = dryrun.task
TASK = json.loads((HERE / "task.json").read_text())
LIMITS = {"title": 120, "instruction": 1000, "context": 1000, "fidelity": 500, "scores": 300, "run_limit": 200, "reset_notes": 1500}
RUN_LIMIT = 3600


# -- geometry ---------------------------------------------------------------------------------------------------------------------------------
def test_slot_section():
    boxes = {name: (np.array(c), np.array(s), slope) for name, c, s, slope in g.section_boxes()}
    lips = boxes["lip_p"], boxes["lip_n"]
    assert abs((lips[0][0][0] - lips[0][1][0] / 2) - (lips[1][0][0] + lips[1][1][0] / 2) - g.SLOT_OPEN) < 1e-9  # the opening between the lips
    assert abs((boxes["wall_p"][0][0] - boxes["wall_p"][1][0] / 2) * 2 - g.CAVITY_W) < 1e-9  # the cavity between the upright walls
    assert abs(boxes["core"][0][1] + boxes["core"][1][1] / 2 - (g.BAR - g.SLOT_DEPTH)) < 1e-9  # the floor
    assert abs(g.CAVITY_H - 0.008) < 1e-9 and 0.009 < g.FLOOR_W < 0.011
    for s, name in ((1.0, "slope_p"), (-1.0, "slope_n")):  # each sloping slab's cavity face runs from the upright wall's foot to the floor's edge
        c, size, slope = boxes[name]
        along = np.array([math.cos(math.radians(slope)), math.sin(math.radians(slope))])
        normal = np.array([-along[1], along[0]])
        face = [c + sign * normal * size[1] / 2 for sign in (1.0, -1.0)]
        for p in (np.array([s * g.CAVITY_W / 2, g.BAR - g.LIP - g.WALL_UP]), np.array([s * g.FLOOR_W / 2, g.BAR - g.SLOT_DEPTH])):
            assert min(abs(float((p - f) @ normal)) for f in face) < 1e-9, name
            assert abs(float((p - c) @ along)) <= size[0] / 2
    assert abs(g.cavity_half_width(0.0) - g.CAVITY_W / 2) < 1e-12 and abs(g.cavity_half_width(g.CAVITY_H) - g.FLOOR_W / 2) < 1e-9


def test_frame_layout():
    bars = {name: (axis, centre, span) for name, axis, centre, span in g.frame_bars()}
    assert abs((g.CROSS_X[1] - g.CROSS_X[0] + g.BAR) - 0.36) < 1e-9 and abs((g.FRAME_Y[1] - g.FRAME_Y[0]) - 0.30) < 1e-9
    assert bars["right"][2] == g.SIDE_SPAN and abs(g.SIDE_SPAN[0] - (g.CROSS_X[0] + g.BAR / 2)) < 1e-12  # the side bars' slots end against the cross bars
    assert g.SIDE_SPAN[0] + g.FOOT[0] / 2 < g.LINE_X < g.SIDE_SPAN[1] - g.FOOT[0] / 2
    assert len(g.bar_boxes("x", (0.0, g.MARKED_Y), g.SIDE_SPAN)) == 7 and len(g.corner_boxes()) == 8
    for name, c, size, rot in g.bar_boxes("y", (g.CROSS_X[0], 0.0), g.FRAME_Y) + g.bar_boxes("x", (0.0, g.MARKED_Y), g.SIDE_SPAN):
        if rot is not None:
            assert abs(np.linalg.det(rot) - 1.0) < 1e-9 and np.allclose(rot @ rot.T, np.eye(3)), name


def test_nut_fits_the_slot():
    flange, boss = g.flange_outline(), g.boss_outline()
    assert g.width_across(flange, 0.0) < g.SLOT_OPEN < g.width_across(flange, 90.0)  # lengthwise it passes the opening, crosswise it cannot
    assert g.width_across(g.flange_outline(False), 0.0) < g.SLOT_OPEN  # and so does the real, wider nut
    assert math.hypot(g.NUT_A, g.NUT_B) < g.CAVITY_W  # the flange alone is never stopped by the cavity's walls: the boss does it
    cw, ccw = g.boss_stops()
    assert 88.0 <= cw <= 96.0 and ccw <= 8.0, (cw, ccw)  # a quarter turn clockwise, and next to nothing the other way
    assert g.width_across(boss, 90.0) < g.SLOT_OPEN  # crosswise, the boss fits the opening as the nut is drawn up
    assert all(v > 0.0035 for v in g.wing_reach(90.0)) and all(v < 0 for v in g.wing_reach(0.0))


def test_nut_hangs_where_it_can_swing():
    assert g.NUT_TOP > g.LIP  # foot on the bar: the flange is below the lips
    assert 0.0005 < g.NUT_DROP < g.NUT_E  # and part of the boss is still in the opening, which is what stops the swing at a quarter turn
    assert g.NUT_TOP - g.NUT_E < g.LIP - 0.0005
    assert abs(g.SCREW_END - (g.NUT_TOP + g.NUT_F)) < 1e-12  # the screw's end is flush with the nut's underside
    assert g.SCREW_END < g.SLOT_DEPTH - 0.002  # and clear of the slot's floor
    for top in (g.NUT_DROP, 0.0):  # crosswise, hanging and drawn up, the flange's ends clear the cavity's walls at every depth
        for depth, half in ((0.0, g.NUT_A / 2), (g.NUT_BEVEL_FROM, g.NUT_A / 2), (g.NUT_F, g.NUT_A / 2 - (g.NUT_F - g.NUT_BEVEL_FROM))):
            assert half < g.cavity_half_width(top + depth) - 0.0005, (top, depth)
    assert g.TRAVEL[0] < 0 < g.NUT_DROP < g.TRAVEL[1]


def test_hulls_are_small_enough():
    for x in (-g.SCREW_DX, g.SCREW_DX):
        for mesh in task.nut_meshes(x, collide=True):
            assert len(mesh.v) <= 64 and np.ptp(mesh.v[:, 2]) >= 0.004  # PhysX hulls: 64 points at most, and not a sliver
    assert len(g.blade_points()) == 18


def test_bracket_mass_and_hold():
    mass, com, inertia = g.bracket_mass()
    assert 0.15 < mass < 0.19
    assert abs(com[1] - g.GRIP_Y) < 0.015 and abs(com[0]) < 0.012  # the hand holds it near its centre of mass
    assert np.all(inertia > 0) and inertia[1] < inertia[0]
    assert g.GRIP_Y + 0.034 < g.ARM_END - g.CAM[1]  # the fingers' width stays short of the camera's side plate
    tip = mass * 9.81 * (com[1] - g.FOOT[1] / 2)  # the overhang's moment about the bar's inner edge
    assert tip > 0.1  # it does tip an unfastened foot
    assert r.GRIP_TURNS * g.PITCH * r.K_THREAD * g.FOOT[1] / 2 > 1.3 * tip  # and one nut that grips, by the rule, holds it down


def test_driver_fits_the_screw():
    assert g.HEX_AF < g.SOCKET_AF < g.HEX_AF / math.cos(math.pi / 6)  # the blade goes in, and cannot turn in the socket
    walls = g.socket_walls()
    assert len(walls) == 6
    for (cx, cy), (radial, tangent), turn in walls:
        assert abs(math.hypot(cx, cy) - radial / 2 - g.SOCKET_AF / 2) < 1e-9  # each slab's inner face is one flat of the socket
    assert g.SOCKET_DEPTH >= 0.003 and g.TIP_TO_TCP == g.BLADE + g.GRIP_UP
    assert g.HOLE > g.HEX_AF / math.cos(math.pi / 6) and g.HOLE_DEPTH < g.BLOCK[2]
    assert abs(g.DRIVER_START[2] - (g.BLOCK[2] - g.HOLE_DEPTH)) < 1e-12


# -- the thread rule ------------------------------------------------------------------------------------------------------------------------------
def test_thread_follows_the_nut():
    t = r.Thread()
    assert t.step(0.0, 0.0) == (0.0, 0.0, r.T_RUN)  # carried round with its screw: no turn on it, no travel
    target, lag, friction = t.step(math.pi, 0.0)  # half a turn on the screw, the nut not yet moved
    assert abs(target - 0.0005) < 1e-12 and abs(lag - 0.0005) < 1e-12 and friction > r.T_RUN
    target, lag, friction = t.step(math.pi, 0.0005)  # the slide has followed
    assert abs(lag) < 1e-12 and abs(friction - r.T_RUN) < 1e-12


def test_thread_unwraps_and_clips():
    t = r.Thread()
    angle, travel = 0.0, 0.0
    for _ in range(400):  # eight turns, the reading wrapping at +-pi
        angle += 2 * math.pi / 50
        target, lag, friction = t.step(math.remainder(angle, 2 * math.pi), travel)
        travel = min(target, g.NUT_DROP)  # the lips stop the nut
    assert abs(t.turn - angle) < 1e-9
    assert abs(target - g.TRAVEL[1]) < 1e-12  # the thread runs out where the boss would meet the foot
    assert abs(lag - (g.TRAVEL[1] - g.NUT_DROP)) < 1e-12 and friction <= r.T_MAX
    t = r.Thread()
    for _ in range(400):  # backed off the other way
        angle -= 2 * math.pi / 50
        target, lag, friction = t.step(math.remainder(angle, 2 * math.pi), 0.0)
    assert abs(target - g.TRAVEL[0]) < 1e-12


def wind(turns: float) -> float:
    """The friction of a nut wound `turns` past where the lips stopped it, turned there a little at a time as a screw is."""
    t = r.Thread()
    total, friction = 2 * math.pi * (g.NUT_DROP / g.PITCH + turns), r.T_RUN
    for k in range(1, 201):
        angle = total * k / 200
        friction = t.step(math.remainder(angle, 2 * math.pi), min(g.PITCH * angle / (2 * math.pi), g.NUT_DROP))[2]
    return friction


def test_thread_torque_rises_with_the_clamp():
    assert r.T_RUN < wind(0.25) < wind(0.5) < wind(1.0)
    clamp = r.K_THREAD * r.TIGHT_TURNS * g.PITCH
    assert abs(wind(r.TIGHT_TURNS) - (r.T_RUN + r.NUT_FACTOR * r.THREAD_D * clamp)) < 1e-9
    assert r.T_HEAD + r.T_RUN + r.NUT_FACTOR * r.THREAD_D * clamp < 0.1  # tight, by the demo's measure, takes less torque than a gripper passes through the handle


# -- the score ------------------------------------------------------------------------------------------------------------------------------------
def nut(turn=90.0, top=g.LIP, lag=0.0005, across=0.0, along=g.LINE_X) -> r.Nut:
    return r.Nut(along=along, across=across, top=top, turn=turn, lag=lag)


def obs(nuts=None, x=g.LINE_X, yaw=0.0, finger=0.2, driver=0.2, tilt=0.0) -> r.Obs:
    nuts = nuts or (nut(along=x - g.SCREW_DX), nut(along=x + g.SCREW_DX))
    rot = g.rot_z(yaw) @ dryrun.rot_x(tilt)
    return r.Obs(nuts=tuple(nuts), origin=np.array([x, g.MARKED_Y, g.BAR]), rot=rot, finger_gap=finger, driver_gap=driver)


def settled(o: r.Obs, steps: int = r.HOLD_STEPS + 1) -> float:
    rule = r.Rule()
    return [rule.step(o) for _ in range(steps)][-1]


def test_nut_states():
    assert nut().grips and nut().under_lips and nut().in_slot
    assert not nut(lag=0.0002).grips  # under the lips but not yet a quarter turn past them
    assert not nut(top=g.LIP + g.NUT_DROP, lag=0.0).grips and nut(top=g.LIP + g.NUT_DROP).under_lips  # swung, still hanging 0.7 mm below the lips
    assert nut(turn=60.0, top=g.LIP + g.NUT_E).grips  # stopped short of square and drawn up on its boss: it holds
    assert not nut(turn=0.0, top=g.NUT_TOP).under_lips and nut(turn=0.0, top=g.NUT_TOP).in_slot  # lengthwise in the slot
    assert not nut(turn=0.0, top=g.NUT_E, lag=0.002).grips  # run up lengthwise against the foot: a tight screw on a loose bracket
    assert not nut(turn=90.0, top=g.LIP - 0.001, lag=0.002).grips  # wound, but above where the lips' underside is
    assert not nut(turn=30.0).under_lips and nut(turn=45.0).grips  # at 45 degrees each end is 1.5 mm under its lip; at 30, not under it at all
    assert not nut(across=0.03).in_slot and not nut(top=-0.002).in_slot  # over another part of the table, or lying on top of the bar
    assert not nut(along=g.SIDE_SPAN[1] + 0.01).in_slot
    assert not nut(across=0.0035, turn=90.0).grips  # pushed so far across that one end is barely under its lip


def test_ladder():
    assert r.ladder(obs()) == 1.0
    assert r.ladder(obs(x=g.LINE_X + 0.0019)) == 1.0 and r.ladder(obs(x=g.LINE_X + 0.003)) == 0.75 and r.ladder(obs(x=g.LINE_X - 0.0049)) == 0.75
    assert r.ladder(obs(x=g.LINE_X + 0.006)) == 0.5
    assert r.ladder(obs(yaw=180.0)) == 0.5  # fastened with the arm pointing out of the frame
    one = (nut(), nut(turn=0.0, top=g.NUT_TOP, lag=0.0))
    assert r.ladder(obs(nuts=one)) == 0.5 and r.ladder(obs(nuts=one[::-1])) == 0.5
    assert r.ladder(obs(nuts=(nut(), nut(top=-0.003, across=0.004)))) == 0.5  # one grips, the other never went in
    hanging = (nut(turn=0.0, top=g.NUT_TOP, lag=0.0), nut(turn=0.0, top=g.NUT_TOP, lag=0.0))
    assert r.ladder(obs(nuts=hanging)) == 0.25
    swung = (nut(top=g.LIP + g.NUT_DROP, lag=0.0), nut(top=g.LIP + g.NUT_DROP, lag=0.0))
    assert r.ladder(obs(nuts=swung)) == 0.25  # both swung under the lips, neither drawn up
    assert r.ladder(obs(nuts=(hanging[0], nut(turn=0.0, top=-0.004, lag=0.0)))) == 0.0  # one nut sits on top of the bar
    jammed = (nut(turn=0.0, top=g.NUT_E, lag=0.002), nut(turn=0.0, top=g.NUT_E, lag=0.002))
    assert r.ladder(obs(nuts=jammed)) == 0.25  # both screws tight on nuts that never swung: in the slot, holding nothing
    assert r.ladder(obs(nuts=(nut(across=0.26), nut(across=0.26)))) == 0.0  # on the other side bar


def test_end_conditions():
    assert settled(obs()) == 1.0
    assert settled(obs(finger=0.004)) == 0.0  # a hand still on the arm
    assert settled(obs(finger=float("nan"))) == 0.0  # fingers not known: not clear
    assert settled(obs(driver=0.0)) == 0.0  # the driver left standing in a screw
    assert settled(obs(), steps=r.HOLD_STEPS) == 0.0  # not yet a second
    rule = r.Rule()
    for k in range(60):  # a bracket that is still being moved scores nothing; one second after it stops, it does
        moving = k < 30
        score = rule.step(obs(x=g.LINE_X + (0.0005 * math.sin(k) if moving else 0.0)))
        assert score == (1.0 if k >= 30 + r.HOLD_STEPS else 0.0), k
    rule = r.Rule()
    for k in range(40):  # held on the line and let go: the score comes only with the release
        score = rule.step(obs(finger=0.0 if k < 25 else 0.05))
        assert score == (0.0 if k < 25 else 1.0), k


def test_tipped_bracket():
    o = dryrun.World()
    o.bracket = [np.array([g.LINE_X, g.MARKED_Y, g.BAR]) + dryrun.UP, np.eye(3)]
    o.unheld()  # nobody holds it, no nut grips
    assert o.tipped
    task.states = lambda e, names: {n: o.state(n) for n in names}
    dryrun.m.link_pose = lambda e, part, link: o.link_pose(task.NAMES.index(link[-1]))
    dryrun.m.fingertips = lambda e: o.fingertips()
    seen = task.TRIAL.obs(None)
    assert all(n.in_slot and not n.under_lips for n in seen.nuts) and r.ladder(seen) == 0.25
    assert not r.points_in(obs(tilt=20.0))


# -- the task file's wiring -----------------------------------------------------------------------------------------------------------------------
def test_parts_build():
    names = [o.name for o in task.OBJECTS]
    assert names == ["frame", "block", "bracket", "driver", "supports"]
    for o in task.OBJECTS:
        text = o.build()
        assert text.startswith("#usda 1.0") and "nan" not in text
    b = task.BRACKET.build()
    for link in ("screw_a", "travel_a", "nut_a", "screw_b", "travel_b", "nut_b"):
        assert f'"{link}_joint"' in b
    assert b.count("convexHull") == 4 and "torsionalPatchRadius" in b and "contactOffset = 0.001" in b
    drives = dict(task.BRACKET.drives)
    assert drives["travel_a_joint"][0] == r.K_THREAD and drives["nut_a_joint"][3] == r.T_RUN and drives["screw_a_joint"][3] == r.T_HEAD
    assert "contactOffset = 0.001" in task.FRAME.build()
    assert abs(task.BRACKET.mass or 0.0) == 0.0  # a mechanism carries its mass in its frame


def test_see_nut():
    world = dryrun.World()
    world.bracket = [np.array([g.LINE_X, g.MARKED_Y + 0.0002, g.BAR]) + dryrun.UP, np.eye(3)]
    for k in range(2):
        rot, pos = world.link_pose(k)
        seen = task.see_nut(rot, pos - dryrun.UP, k, 0.0)
        assert abs(seen.top - g.NUT_TOP) < 1e-9 and abs(seen.across - 0.0002) < 1e-9 and seen.turn < 1e-6
        assert abs(seen.along - (g.LINE_X + g.SCREWS[k][0])) < 1e-9
    world.screw[0], world.nut[0], world.travel[0] = -math.radians(120.0), math.radians(30.0), g.NUT_DROP
    rot, pos = world.link_pose(0)
    seen = task.see_nut(rot, pos - dryrun.UP, 0, 0.0004)
    assert abs(seen.turn - 90.0) < 1e-6 and abs(seen.top - g.LIP) < 1e-9 and seen.grips
    assert abs(seen.along - (g.LINE_X + g.SCREWS[0][0])) < 1e-9  # a nut turns about its own screw


def test_gaps():
    origin, rot = np.array([g.LINE_X, g.MARKED_Y, g.BAR]), np.eye(3)
    far = np.array([[0.5, 0.4, 0.3]])
    on_arm = origin + g.GRIP_POINT + [g.ARM_W / 2, 0.0, 0.0]
    driver_away = np.array([0.2, -0.25, 0.005])
    assert task.gaps(origin, rot, far, driver_away, np.eye(3))[0] > 0.1
    assert task.gaps(origin, rot, on_arm[None, :], driver_away, np.eye(3))[0] < 1e-9
    in_screw = origin + g.SCREWS[0] + [0.0, 0.0, g.SOCKET_FLOOR]
    assert task.gaps(origin, rot, far, in_screw, np.eye(3))[1] < 1e-9
    assert task.gaps(origin, rot, far, in_screw + [0.0, 0.0, 0.02], np.eye(3))[1] > r.CLEAR


def test_text():
    for field, limit in LIMITS.items():
        assert len(TASK[field]) <= limit, (field, len(TASK[field]))
    assert TASK["instruction"] == task.INSTRUCTION
    assert TASK["category"] == "Impact on AI timelines"
    blob = json.dumps(TASK, ensure_ascii=False) + (HERE / "README.md").read_text()
    assert "—" not in blob and "–" not in blob
    mass = g.bracket_mass()[0]
    for needle in (f"{g.SLOT_OPEN * 1000:g} mm", f"{g.LIP * 1000:g} mm", f"{g.CAVITY_W * 1000:g} mm", f"{g.SLOT_DEPTH * 1000:g} mm", f"{mass * 1000:.0f} g",
                   f"{g.FOOT[0] * 1000:g} x {g.FOOT[1] * 1000:g} x {g.FOOT[2] * 1000:g} mm", "15.75 x 7.75 x 6.07 mm", f"{g.SUPPORT_H * 1000:g} mm high",
                   f"({g.START_XY[0] * 100:g}, {g.START_XY[1] * 100:g})", f"({g.BLOCK_XY[0] * 100:g}, {g.BLOCK_XY[1] * 100:g})", f"x {g.LINE_X * 100:g}"):
        assert needle in TASK["setup"], needle
    for needle in (f"{r.LINE_FULL * 1000:g} mm", f"{r.LINE_PART * 1000:g} mm", f"{r.UNDER_LIP * 1000:g} mm", "quarter turn", f"{r.CLEAR * 100:g} cm", f"{r.IN_SLOT * 1000:g} mm"):
        assert needle in TASK["scoring"], needle
    for needle in (f"{r.T_RUN * 1000:g} mN m", f"{r.K_THREAD / 1000:g} N/mm", f"{r.T_HEAD * 1000:g} mN m"):
        assert needle in TASK["fidelity"], needle
    assert f"{RUN_LIMIT:,} steps" in TASK["run_limit"]


# -- the scripted demo, against the stand-in --------------------------------------------------------------------------------------------------
def test_demo_full():
    out = dryrun.run("full")
    assert out["success"] and out["score"] == 1.0, out
    assert out["steps"] * 1.5 < RUN_LIMIT, out["steps"]  # room to spare inside the run limit
    assert out["ik_mm"] < 2.0 and out["ik_deg"] < 1.0, out
    assert all(1.2 < t < 2.5 for t in out["turns"]), out["turns"]  # a quarter turn to swing, 0.7 to draw up, half a turn tight
    assert "both nuts pass the slot's opening" in out["notes"] and "the driver stands in its block" in out["notes"]


def test_demo_other_ends():
    for mode, want in (("none", 0.0), ("held", 0.0), ("one_screw", 0.5), ("off_line", 0.75)):
        out = dryrun.run(mode)
        assert out["score"] == want and not out["success"], (mode, out["score"], out["now"])
        assert out["ik_mm"] < 2.0, (mode, out)
    out = dryrun.run("unheld")  # let go before any nut grips: the bracket tips, and the screws then go tight on nothing
    assert out["score"] <= 0.25 and not out["success"], out


def test_demo_clearances():
    if "--fast" in sys.argv:
        print("   (skipped with --fast)")
        return
    dryrun.CLEAR = dryrun.Clearance()
    try:
        out = dryrun.run("full")
    finally:
        things, arms = dryrun.CLEAR.worst()
        dryrun.CLEAR = None
    assert out["success"]
    assert things > 0.004, things  # no arm link comes within 4 mm of anything it is not holding
    assert arms > 0.012, arms  # nor within 12 mm of the other arm


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests)} tests, {failed} failed")
    sys.exit(1 if failed else 0)
