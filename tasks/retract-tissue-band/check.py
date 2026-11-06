"""Offline check of retract_tissue_band.py (no simulator). Run from the repository root with yam_arena importable:

    python tasks/retract-tissue-band/check.py

1. Design: the room the right gripper has at the band's near end and at each fragment once it is uncovered (the
   arm's collision meshes placed by IK), how far the band's near end can be lifted before the bridge stops it, and
   whether the top camera sees the band's slack, the bridge, the fragments and the tray past the arms at their
   ready pose.
2. Scores: hand-built trials (the band, the fragments and the arms step by step, on a stand-in for the episode
   clock) against the score each must get.
3. Demo: plays the right arm's plan through the IK at 20 Hz with the band and each fragment following the gripper
   that holds them, scores it, and prints the smallest distances from the robot to the strips, the bridge, the pad,
   the tray, the band and the fragment it is not taking, the worst IK error and the joint limits.
4. Naive variant: the same with the band slid until it comes out from under the bridge.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState
from yam_arena.urdf import quat_xyzw_to_matrix

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("retract_tissue_band.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ, EVERY = 20, 4
MM = 1e3
SAFE = 0.005  # the room the demo must keep
PARKED = dict.fromkeys(("left", "right"), np.r_[READY, FINGER_OPEN])
BASE = station.BASES["right"]
S, D = T.ORDER  # the fragment uncovered first (40 cm out) and the one uncovered last (33.5 cm out)
SLIDE_S, SLIDE_D = (T.FRAGMENT_SPECS[name][0] for name in (S, D))

# the simulator's episode clock and arm joints, which the task file reads from the embodiment
NOW = {"t": 0.0, "arms": PARKED}
T.clock = lambda: NOW["t"]
T.arm_joints = lambda: NOW["arms"]


# ---- states ----
def box(size, xy, under, yaw=0.0, moving=False) -> ObjectState:
    """A box with its centre over xy and its underside at height `under`, turned yaw about the vertical."""
    q = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * np.array(size) / 2
    corners = corners @ T._rotation(q).T
    pos = np.array([*xy, under - corners[:, 2].min()])
    vel = np.array([0.1 if moving else 0.0, 0.0, 0.0])
    return ObjectState(pos, q, vel, np.zeros(3), pos + corners.min(0), pos + corners.max(0))


def band(slide=0.0, dy=0.0, lift=0.0, yaw=0.0, moving=False) -> ObjectState:
    """The band slid `slide` toward the robot from its start, its underside `lift` above the strips."""
    return box(T.BAND_SIZE, (T.BAND.pos[0] - slide, T.Y0 + dy), T.STRIP_TOP + lift, yaw, moving)


def fragment(name, xy=None, under=T.PAD_TOP, yaw=0.0, moving=False) -> ObjectState:
    return box(T.FRAGMENT_SIZE, T.FRAGMENTS[name].pos[:2] if xy is None else xy, under, yaw, moving)


START = {"band": band(), **{name: fragment(name) for name in T.VALUES}}


def beside(name) -> tuple:
    """A place on the pad beside the channel, level with the fragment's mark."""
    return (T.FRAGMENTS[name].pos[0], T.Y0 - 0.075)


def slid(slide) -> dict:
    return {"band": band(slide)}


def up(name, under=0.05) -> dict:
    """The fragment lifted over where it lay, its underside `under` above the pad."""
    return {name: fragment(name, under=T.PAD_TOP + under)}


def tray(name, under=0.0, moving=False) -> dict:
    """The fragment over its place in the tray, its underside `under` above the tray's floor."""
    return {name: fragment(name, T.TRAY_SPOTS[name], T.TRAY_FLOOR_TOP + under, moving=moving)}


def taken(*names) -> list:
    """Each fragment lifted out and set down in the tray, one after the other."""
    return [move for name in names for move in (up(name), tray(name))]


def gripper_at(xyz, yaw, finger=FINGER_OPEN) -> dict:
    """The arms with the right gripper's TCP at a point, pointing down, and the left arm parked."""
    q, miss, _ = ik.solve(xyz, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), READY, BASE)
    assert miss < 1e-3, f"the right gripper cannot reach {xyz} at yaw {math.degrees(yaw):.0f}"
    return {**PARKED, "right": np.r_[q, finger]}


BAND_Z = T.STRIP_TOP + T.BAND_PADS_ABOVE + PAD_TIP_PAST_TCP  # TCP height at the band grasp
FRAGMENT_Z = T.PAD_TOP + T.PADS_ABOVE_UNDERSIDE + PAD_TIP_PAST_TCP  # TCP height at a fragment grasp
WIDE = -(T.FRAGMENT_SIZE[0] + T.APPROACH) / 2  # finger target coming down on a fragment


# ---- distances from the right arm ----
def _groups() -> dict:
    """The fixed props as (size, state) by group."""
    kind = {
        "pad": "the pad",
        "strip": "the strips",
        "post": "the bridge and its posts",
        "bridge": "the bridge and its posts",
    }
    out = {}
    for b in T.FIXED:
        what = kind.get(b.name.split("_")[0], "the tray")
        out.setdefault(what, []).append((b.size, box(b.size, b.pos[:2], b.pos[2] - b.size[2] / 2)))
    return out


GROUPS = _groups()


def gaps(arms, objects) -> dict:
    """Smallest distance (m) from the right arm (every link, its points within 15 cm of the table) to each group of
    fixed props, to the band and to each fragment."""
    points = T.arm_points(arms["right"], "right")
    points = points[points[:, 2] < T.TOP + 0.15]
    out = {what: min(T.box_gap(points, b, size) for size, b in boxes) for what, boxes in GROUPS.items()}
    out["the band"] = T.box_gap(points, objects["band"], T.BAND_SIZE)
    for name in T.VALUES:
        out[name] = T.box_gap(points, objects[name], T.FRAGMENT_SIZE)
    return out


def shown(g, *keys) -> str:
    return ", ".join(f"{k} {g[k] * MM:.1f}" for k in keys)


# ---- trials ----
def step(objects, arms=PARKED, hook=T.score):
    NOW["arms"] = arms
    NOW["t"] += 1 / HZ
    return hook(dict(objects))


def play(moves, restart=True) -> float:
    """The score at the end of a trial that starts from the layout and applies the moves one step at a time. A
    move changes some objects' states; its "arms" entry places the arms for that step only."""
    if restart:
        NOW["t"] = 0.0
    world, out = dict(START), step(START)
    for move in moves:
        move = dict(move)
        arms = move.pop("arms", PARKED)
        world.update(move)
        out = step(world, arms)
    play.world = world
    return out


def check_design():
    props = ("the strips", "the bridge and its posts", "the pad", "the tray")
    # R1: the band is pinched across its width at its near end with the fingers fully open, clear of everything
    g = gaps(gripper_at((T.GRIP_X, T.Y0, BAND_Z), T.BAND_YAW), START)
    closing = min(
        gaps(gripper_at((T.GRIP_X, T.Y0, BAND_Z), T.BAND_YAW, f), START)["the strips"]
        for f in np.linspace(FINGER_OPEN, -T.BAND_SIZE[1] / 2, 8)
    )
    print(
        f"  band grasp, fingers open (mm): {shown(g, *props, 'the band')}; to the strips while closing {closing * MM:.1f}"
    )
    assert all(g[k] >= SAFE for k in (*props, "the band")) and closing >= SAFE, "the band grasp has no room"
    # R2: each fragment, with the band slid just enough, is pinched along the channel clear of the strips and the
    # band; 1 cm less slide and the band is in the fingers' way
    for name, (slide, _) in T.FRAGMENT_SPECS.items():
        x, y, _ = T.FRAGMENTS[name].pos
        arms = gripper_at((x, y, FRAGMENT_Z), T.FRAGMENT_YAW, WIDE)
        g = gaps(arms, {**START, **slid(slide)})
        short = gaps(arms, {**START, **slid(slide - 0.01)})["the band"]
        other = D if name == S else S
        print(
            f"  {name} grasp (band slid {slide * 100:.1f} cm): {shown(g, *props, 'the band', other)}; band slid 1 cm less {short * MM:.1f}"
        )
        assert g["the pad"] >= T.PADS_ABOVE_UNDERSIDE - 1e-4
        assert all(g[k] >= SAFE for k in ("the strips", "the bridge and its posts", "the tray", "the band", other)), (
            f"{name}: no room"
        )
        assert short < SAFE, f"{name} could be taken with 1 cm less slide"
    # R3: lifting the band's near end turns it about its far end on the strips until its top meets the bridge
    for slide in (0.0, SLIDE_S, SLIDE_D):
        lift = T.BRIDGE_CLEAR * T.BAND_SIZE[0] / (T.SLACK - slide)
        print(
            f"  band slid {slide * 100:4.1f} cm: its near end rises at most {lift * MM:.1f} mm before the bridge stops it"
        )
    assert T.BRIDGE_CLEAR * T.BAND_SIZE[0] / T.SLACK < SAFE, "the band can be lifted off the strips at the start"


def check_view():
    """The top camera (320 x 176) sees the band's slack, the bridge, the fragments once uncovered and the tray,
    past both arms at their ready pose."""
    cam = station.top_camera()
    lens = cam.lens.resized(320, 176)
    rot, pos = quat_xyzw_to_matrix(np.array(cam.rot_xyzw)), np.array(cam.pos)

    def project(points):
        pc = (np.asarray(points) - pos) @ rot  # camera frame: looks along -z, y up
        depth = -pc[:, 2]
        return np.c_[lens.cx + lens.fx * pc[:, 0] / depth, lens.cy - lens.fy * pc[:, 1] / depth, depth]

    nearest = {}
    for side in ("left", "right"):
        for u, v, d in project(T.arm_points(np.r_[READY, FINGER_OPEN], side)):
            key = (round(u), round(v))
            nearest[key] = min(nearest.get(key, np.inf), d)
    points = {
        "band far end": (T.BAND_X[1], T.Y0, T.BAND_TOP),
        "band near end": (T.BAND_X[0], T.Y0, T.BAND_TOP),
        "bridge": (T.BRIDGE_CENTRE[0], T.Y0, T.BAND_TOP + T.BRIDGE_CLEAR + T.BRIDGE_BAR),
        **{name: (*T.FRAGMENTS[name].pos[:2], T.PAD_TOP + T.FRAGMENT_SIZE[2]) for name in T.VALUES},
        **{
            f"tray corner {i}": (
                T.TRAY_CENTRE[0] + sx * T.TRAY_INSIDE[0] / 2,
                T.TRAY_CENTRE[1] + sy * T.TRAY_INSIDE[1] / 2,
                T.TOP,
            )
            for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        },
    }
    for label, point in points.items():
        u, v, d = project([point])[0]
        inside = 0 <= u < 320 and 0 <= v < 176
        behind = any(nearest.get((round(u) + du, round(v) + dv), np.inf) < d for du in (-1, 0, 1) for dv in (-1, 0, 1))
        print(
            f"  {label:14s} at pixel ({u:5.1f}, {v:5.1f}){'' if inside else ', OUT OF FRAME'}{', BEHIND AN ARM' if behind else ''}"
        )
        assert inside and not behind, label
    a, b = project([(0.30, T.Y0, T.TOP), (0.31, T.Y0 + 0.01, T.TOP)])
    print(f"  1 cm along x is {abs(b[1] - a[1]):.1f} px, 1 cm along y is {abs(b[0] - a[0]):.1f} px, near the bridge")


def check_scores():
    holding_band = gripper_at((T.GRIP_X, T.Y0, BAND_Z), T.BAND_YAW, -(T.BAND_SIZE[1] / 2 - T.GRIP_IN))
    holding_d = gripper_at(
        (*T.TRAY_SPOTS[D], T.TRAY_FLOOR_TOP + T.PADS_ABOVE_UNDERSIDE + PAD_TIP_PAST_TCP), T.TRAY_YAW, -0.007
    )
    over_tray = gripper_at((*T.TRAY_SPOTS[S], T.TRAY_FLOOR_TOP + 0.044 + PAD_TIP_PAST_TCP), T.TRAY_YAW, -0.007)
    back = slid(0.0)
    done = [slid(SLIDE_D), *taken(S, D), back]
    cases = {
        "start, nothing moved": ([], 0.0),
        "band slid 13.5 cm, both fragments in the tray, band laid back": (done, 1.0),
        "same, fragments in the other order": ([slid(SLIDE_D), *taken(D, S), back], 1.0),
        "same, band laid back 1.5 cm past its mark": ([slid(SLIDE_D), *taken(S, D), slid(-0.015)], 1.0),
        "band slid 8 cm, far fragment in the tray, band laid back": ([slid(0.08), *taken(S), back], 0.4),
        "band slid 8 cm, far fragment in the tray, band left where it is": ([slid(0.08), *taken(S)], 0.2),
        "band slid 13.5 cm, both in the tray, band left where it is": ([slid(SLIDE_D), *taken(S, D)], 0.5),
        "band slid 16 cm (1 cm of slack left), both in the tray, band laid back": (
            [slid(0.16), *taken(S, D), back],
            0.5,
        ),
        "band slid 17.5 cm (out from under the bridge), both in the tray, band laid back": (
            [slid(0.175), *taken(S, D), back],
            0.0,
        ),
        "band slid out, pushed back under the bridge to its mark, then everything done": (
            [slid(0.175), back, *done],
            0.0,
        ),
        "band slid to 1 cm of slack, back to 3.5 cm, then everything done": ([slid(0.16), *done], 0.5),
        "band pushed 10 cm away from the robot, out the far side of the bridge, then everything done": (
            [slid(-0.10), *done],
            0.0,
        ),
        "both in the tray, band back on its mark but lifted 2 cm": ([*done, {"band": band(lift=0.02)}], 0.5),
        "both in the tray, band back but turned 15 degrees": ([*done, {"band": band(yaw=math.radians(15))}], 0.5),
        "both in the tray, band back but 2 cm off the channel's axis": ([*done, {"band": band(dy=0.02)}], 0.5),
        "both in the tray, band's near end 3 cm short of its mark": ([*done, slid(0.03)], 0.5),
        "both in the tray, band's near end 2.5 cm past its mark": ([*done, slid(-0.025)], 0.5),
        "both in the tray, band back, the gripper still on it": ([*done, {**back, "arms": holding_band}], 0.5),
        "both in the tray, band back but still moving": ([*done, {"band": band(moving=True)}], 0.5),
        "near fragment in the tray, far one held 4 cm over the tray at the end": (
            [slid(SLIDE_D), *taken(D), back, up(S), {**tray(S, under=0.04), "arms": over_tray}],
            0.6,
        ),
        "far fragment in the tray, near one in the tray with the gripper still on it": (
            [slid(SLIDE_D), *taken(S), back, up(D), {**tray(D), "arms": holding_d}],
            0.4,
        ),
        "far fragment in the tray, then taken out and left on the pad beside the channel": (
            [slid(SLIDE_D), *taken(S, D), up(S), {S: fragment(S, beside(S))}, back],
            0.8,
        ),
        "far fragment slid 4 cm along the channel, then lifted into the tray": (
            [slid(SLIDE_D), {S: fragment(S, (T.FRAGMENT_X[S] - 0.04, T.Y0))}, *taken(S, D), back],
            0.6,
        ),
        "far fragment lifted and dropped back in the channel": (
            [slid(SLIDE_D), up(S), {S: fragment(S)}, *taken(D), back],
            0.6,
        ),
        "far fragment lifted and dropped on the table past the pad": (
            [slid(SLIDE_D), up(S), {S: fragment(S, (0.56, T.Y0), T.TOP)}, *taken(D), back],
            0.8,
        ),
        "far fragment lifted and set on a strip's top": (
            [slid(SLIDE_D), up(S), {S: fragment(S, (T.FRAGMENT_X[S], T.Y0 - 0.0375), T.STRIP_TOP)}, *taken(D), back],
            0.6,
        ),
        "far fragment lifted and set on the band": (
            [slid(SLIDE_D), up(S), {S: fragment(S, (0.20, T.Y0), T.BAND_TOP)}, *taken(D), back],
            0.6,
        ),
        "near fragment in the tray on top of the far one": (
            [
                slid(SLIDE_D),
                *taken(S),
                up(D),
                {D: fragment(D, T.TRAY_SPOTS[S], T.TRAY_FLOOR_TOP + T.FRAGMENT_SIZE[2])},
                back,
            ],
            0.7,
        ),
        "far fragment in the tray, near one in the tray but still moving": (
            [slid(SLIDE_D), *taken(S), up(D), tray(D, moving=True), back],
            0.4,
        ),
        "far fragment dropped off the table": (
            [slid(SLIDE_D), up(S), {S: fragment(S, (0.40, -0.80), 0.0)}, *taken(D), back],
            0.6,
        ),
        "far fragment swept out of the channel over a strip, never lifted": (
            [slid(SLIDE_D), {S: fragment(S, beside(S))}, *taken(D), back],
            0.6,
        ),
        "band slid 8 cm, both in the tray (near one still covered in reality), band laid back": (
            [slid(0.08), *taken(S, D), back],
            1.0,
        ),
    }
    for name, (moves, want) in cases.items():
        got = play(moves)
        print(f"  {got:.4f}  {name}")
        assert abs(got - want) < 1e-9, f"{name}: score {got}, want {want}"

    # success is the 1.0 state only, and holds while it lasts
    play(done)
    assert all(step(play.world, hook=T.success) for _ in range(HZ))
    assert not step({**play.world, **tray(D, moving=True)}, hook=T.success)
    assert not step(play.world, holding_band, hook=T.success)
    assert not step(play.world, holding_d, hook=T.success)
    play([slid(0.16), *taken(S, D), back])
    assert not step(play.world, hook=T.success)  # 1 cm of slack at some time: 0.5, and the episode runs on
    play([slid(SLIDE_D), *taken(S, D)])
    assert not step(play.world, hook=T.success)  # the band is not laid back
    play([slid(SLIDE_D), *taken(S), back])
    assert not step(play.world, hook=T.success)

    # two episodes back to back: the memory clears when the clock goes back, and only then
    assert play([slid(0.175), *done]) == 0.0
    assert play(done, restart=False) == 0.0  # same episode, the band back under the bridge first
    assert play(done) == 1.0  # new episode
    # every sum of full, half or no shares, halved up to twice, is a multiple of 0.025
    values = np.array(list(T.VALUES.values()))
    for shares in np.ndindex(3, 3):
        for factor in (1.0, 0.5, 0.25):
            steps = values @ (np.array(shares) / 2) * factor / 0.025
            assert abs(steps - round(steps)) < 1e-9


def _grip_yaw(rot) -> float:
    turn = rot @ grip_rotation(0.0).T
    return math.atan2(turn[1, 0], turn[0, 0])


def _rest_height(xy) -> float:
    """Height of what a fragment let go over xy comes to rest on (the band aside)."""
    if T._inside(xy, T.TRAY_CENTRE, T.TRAY_INSIDE):
        return T.TRAY_FLOOR_TOP
    if T._inside(xy, T.PAD_CENTRE, T.PAD_SIZE[:2]):
        return T.STRIP_TOP if T.CHANNEL / 2 <= abs(xy[1] - T.Y0) <= T.STRIP_OUTER else T.PAD_TOP
    return T.TOP


def run_demo(variant) -> dict:
    plan = ArmPlan(T.demo(variant)["right"], BASE)
    sizes = {"band": T.BAND_SIZE, **dict.fromkeys(T.ORDER, T.FRAGMENT_SIZE)}
    width = {"band": T.BAND_SIZE[1], **dict.fromkeys(T.ORDER, T.FRAGMENT_SIZE[0])}  # across the closing fingers
    pose = {name: (np.array(spec.pos), 0.0) for name, spec in (("band", T.BAND), *T.FRAGMENTS.items())}
    least = {}
    lo, hi = np.full(6, np.inf), np.full(6, -np.inf)
    held, grasp, target, held_s, slack = None, None, "band", 0, np.inf
    NOW["t"] = 0.0

    def note(what, value, t, label):
        if value < least.get(what, (np.inf,))[0]:
            least[what] = (value, t, label)

    for n, t in enumerate(np.arange(1 / HZ, plan.duration + 3.0, 1 / HZ)):
        action, label = plan.action7(t)
        arms = {**PARKED, "right": np.r_[action[:6], action[6] * FINGER_OPEN]}
        lo, hi = np.minimum(lo, action[:6]), np.maximum(hi, action[:6])
        tcp, rot, finger, _ = plan.target(t)
        yaw = _grip_yaw(rot)
        if label != "done":
            key = label.split(":")[0]
            target = "band" if key == "band" else next(name for name in T.ORDER if name.split("_")[1][0] == key)
        pads_on = abs(finger) <= width[held or target] / 2 + 0.001
        at_it = T._inside(tcp[:2], pose[target][0][:2], np.array(sizes[target][:2]) + 0.02)
        at_it &= tcp[2] < pose[target][0][2] + sizes[target][2] / 2 + 0.02
        if held is None and pads_on and at_it:  # the pads close on it
            held, grasp = target, (pose[target][0] - tcp, pose[target][1], yaw)
        if held is not None and pads_on:  # it follows the gripper
            offset, yaw0, grip0 = grasp
            c, s = math.cos(yaw - grip0), math.sin(yaw - grip0)
            pose[held] = (tcp + np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ offset, yaw0 + yaw - grip0)
        elif held is not None:  # let go: it comes to rest on what is under it
            xyz, turned = pose[held]
            rest = T.STRIP_TOP if held == "band" else _rest_height(xyz[:2])
            pose[held] = (np.array([*xyz[:2], rest + sizes[held][2] / 2]), turned)
            held = None
        objects = {}
        for name, (xyz, turned) in pose.items():
            objects[name] = box(sizes[name], xyz[:2], xyz[2] - sizes[name][2] / 2, turned, moving=name == held)
        final = step(objects, arms)
        held_s = held_s + 1 if T.success(objects) else 0
        slack = min(slack, T.TRACKER.slack)
        if n % EVERY == 0:
            g = gaps(arms, objects)
            for what in GROUPS:
                note(f"robot to {what}", g[what], t, label)
            if target != "band":
                note("robot to the band while taking a fragment", g["the band"], t, label)
            for name in T.ORDER:
                if name != target:
                    note("robot to the fragment it is not taking", g[name], t, label)
    margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
    return {
        "plan": plan,
        "final": final,
        "held_s": held_s / HZ,
        "least": least,
        "margin": margin,
        "slack": slack,
        "band": pose["band"][0],
    }


def check_demo():
    run = run_demo("scripted")
    plan, least, tracker = run["plan"], run["least"], T.TRACKER
    print(
        f"  ideal run: score {run['final']:.2f}, success held {run['held_s']:.1f} s, out {tracker.out}, warned {tracker.warned}"
    )
    print(
        f"  least slack {run['slack'] * 100:.1f} cm; band's near end back at x {(run['band'][0] - T.BAND_SIZE[0] / 2) * 100:.1f} cm"
    )
    for what, (value, t, label) in least.items():
        print(f"  {value * MM:6.1f} mm  {what}, at {t:.1f} s ({label})")
    print(f"  worst IK {plan.worst_ik_mm:.2f} mm / {plan.worst_ik_deg:.3f} deg over {plan.duration:.1f} s")
    print(f"  smallest joint-limit margin per joint (deg): {np.round(np.degrees(run['margin']), 1)}")
    assert run["final"] == 1.0 and run["held_s"] >= T.SUCCESS_HOLD_S and not tracker.out and not tracker.warned
    assert run["slack"] > T.WARN + 0.01
    assert least["robot to the pad"][0] >= T.PADS_ABOVE_UNDERSIDE - 1e-4  # the fingertips, at a fragment grasp
    for what in ("the strips", "the bridge and its posts", "the tray"):
        assert least[f"robot to {what}"][0] >= SAFE, what
    assert least["robot to the band while taking a fragment"][0] >= SAFE
    assert least["robot to the fragment it is not taking"][0] >= SAFE
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1 and (run["margin"] > 0).all()
    assert plan.duration + 3.0 < 2400 / HZ


def check_naive():
    run = run_demo("naive")
    print(
        f"  band slid {T.NAIVE_SLIDE * 100:.0f} cm: out {T.TRACKER.out}, least slack {run['slack'] * 100:.1f} cm, score {run['final']:.2f}"
    )
    assert T.TRACKER.out and run["final"] == 0.0


if __name__ == "__main__":
    print("design:")
    check_design()
    print("view:")
    check_view()
    print("scores:")
    check_scores()
    print("demo:")
    check_demo()
    print("naive variant:")
    check_naive()
    print("ok")
