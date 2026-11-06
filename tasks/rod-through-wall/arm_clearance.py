"""Offline check of the room around the arms in the demo of rod_through_wall.py (no simulator). Run from the
repository root with yam_arena importable:

    python tasks/rod-through-wall/arm_clearance.py

It plays both arms' plans through the IK and places the URDF's collision meshes at every fourth control step. It
prints the smallest distance from each arm to the wall, to the stand and to its own base, and between the two
wrists. The wall height (16 cm) and the hand-off distance (8.5 cm) come from this check: in a straight-down grip
the wrist link reaches 9.8 cm ahead of the TCP, from 14 cm above it.
"""

import importlib.util
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
from yam_arena.demo import ArmPlan
from yam_arena.urdf import forward_kinematics, rpy_matrix

spec = importlib.util.spec_from_file_location("task", Path(__file__).with_name("rod_through_wall.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

HZ, EVERY = 20, 4
MIN_CLEARANCE = 0.02
WRIST = ("link4", "link5", "gripper", "tip_left", "tip_right")


def _vertices(stl: Path) -> np.ndarray:
    data = stl.read_bytes()  # binary STL: 80-byte header, count, then 50 bytes per triangle
    count = struct.unpack("<I", data[80:84])[0]
    record = np.dtype([("normal", "<3f4"), ("corners", "<9f4"), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=record, count=count, offset=84)["corners"].reshape(-1, 3).astype(float)


def _link_points() -> dict:
    """Each link's collision-mesh vertices in the link's frame."""
    out = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        clouds = []
        for collision in link.findall("collision"):
            mesh, origin = collision.find("geometry/mesh"), collision.find("origin")
            if mesh is None:
                continue
            xyz, rpy = (np.array(origin.get(k).split(), dtype=float) for k in ("xyz", "rpy"))
            clouds.append(_vertices(Path(YAM_URDF).parent / mesh.get("filename")) @ rpy_matrix(rpy).T + xyz)
        out[link.get("name")] = np.unique(np.round(np.vstack(clouds), 4), axis=0)
    return out


LINKS = _link_points()


def arm_points(action7, base) -> dict:
    q = dict(zip(ARM_JOINTS, action7[:6], strict=True))
    q["joint7"] = q["joint8"] = action7[6] * FINGER_OPEN
    poses = forward_kinematics(YAM_URDF, q)
    return {name: p @ poses[name][:3, :3].T + poses[name][:3, 3] + np.asarray(base) for name, p in LINKS.items()}


def to_boxes(points, low, high) -> float:
    """Smallest distance from the points to a set of boxes given by their corners."""
    outside = np.maximum(np.maximum(low[:, None] - points, points - high[:, None]), 0.0)
    return float(np.linalg.norm(outside, axis=2).min())


def between(a, b) -> float:
    return min(float(np.sqrt(((a[i : i + 400, None] - b[None]) ** 2).sum(-1)).min()) for i in range(0, len(a), 400))


def main():
    plans = {arm: ArmPlan(waypoints, station.BASES[arm]) for arm, waypoints in T.demo().items()}
    stand_low = np.array([[*(np.array(T.STAND_UV) - np.array(T.STAND_SIZE[:2]) / 2), 0.0]])
    stand_high = np.array([[*(np.array(T.STAND_UV) + np.array(T.STAND_SIZE[:2]) / 2), T.STAND_SIZE[2]]])
    worst = {}

    def note(what, value, t, label):
        if value < worst.get(what, (np.inf,))[0]:
            worst[what] = (value, t, label)

    end = max(plan.duration for plan in plans.values())
    for step, t in enumerate(np.arange(0.0, end + 1 / HZ, 1 / HZ)):
        actions = {arm: plan.action7(t) for arm, plan in plans.items()}
        if step % EVERY:
            continue
        clouds = {arm: arm_points(a, station.BASES[arm]) for arm, (a, _) in actions.items()}
        for arm, links in clouds.items():
            label = actions[arm][1]
            moving = np.vstack([p for name, p in links.items() if name != "base"])
            in_wall_frame = np.c_[T.to_wall(moving[:, :2]), moving[:, 2] - T.TOP]
            hand = np.vstack([links[name] for name in WRIST[2:]])
            note(f"{arm} arm to the wall", to_boxes(in_wall_frame, T._LOW, T._HIGH), t, label)
            note(f"{arm} arm to the stand", to_boxes(in_wall_frame, stand_low, stand_high), t, label)
            base_low, base_high = links["base"].min(0)[None], links["base"].max(0)[None]
            note(f"{arm} gripper to its base", to_boxes(hand, base_low, base_high), t, label)
        wrists = {arm: np.vstack([links[name] for name in WRIST]) for arm, links in clouds.items()}
        if np.linalg.norm(wrists["left"].mean(0) - wrists["right"].mean(0)) < 0.35:
            labels = " / ".join(label for _, label in actions.values())
            note("left wrist to right wrist", between(wrists["left"], wrists["right"]), t, labels)
    for what, (value, t, label) in worst.items():
        print(f"  {value * 1000:6.1f} mm  {what}, at {t:.1f} s ({label})")
        assert value > MIN_CLEARANCE, f"{what}: {value * 1000:.1f} mm"
    print("ok")


if __name__ == "__main__":
    main()
