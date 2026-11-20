"""demokit: closed-loop scripted demos for a yam_arena task file, on the harness's own embodiment and arm plans.

A task file defines ``demo_script(ctx)``, a generator: each ``yield`` is one control step, and each arm primitive is a generator too. A script reads where things are now,
so grasps follow the object and nothing is tuned by hand for droop. It reads only what a policy could observe: object poses, visible joints, its own arm's joints. Never a scorer's state.

    def demo_script(ctx):
        l, r = ctx.left, ctx.right
        yield from ctx.together(l.move((0.3, 0.2, TOP + 0.15), grip=down(-90)), r.move((0.2, -0.2, TOP + 0.2), grip=level(60)))
        yield from r.pick("can", 0.06, width=0.052)
        if ctx.mode == "drop":   # a cheat mode: TASK_DEMO=drop
            yield from r.open()
            return
        yield from r.place((0.3, -0.1, TOP), 0.06)

Run it with the harness's Python, from the task's folder:

    python demokit.py task.py OUT --no-record --mode full,none,drop    debug: stores no frames, renders 1 step in 20 at 320 x 176, and runs every mode in one simulator session
    python demokit.py task.py OUT --no-record --stills 40              debug, with a picture from each camera every 40 steps
    python demokit.py task.py OUT                                      final: the harness's own render at 640 x 360. Writes OUT/full/demo.mp4 and setup.png. Have the GPU to yourself
    python demokit.py task.py OUT --fast --mode full,none              scorer and motion debugging: no cameras, physics on the CPU, about 6 times faster. Check the result once on the GPU
    python demokit.py reach 0.30 -0.10 0.06                            which grips each arm can hold at x, y, height above the table. No simulator

Each run writes OUT/<mode>/result.json, with the harness's task_score, and prints a line with its steps per second and slowest 30 steps.
The final mode renders as the harness does and resizes. The debug mode changes only the render size, not the physics: take the score you report from a final run.

Times are in seconds, lengths in metres, grip angles in degrees. Positions are world TCP positions; `TABLE_TOP` is the table's height.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yam_arena.demo as harness_demo
from yam_arena import ik, station
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, GRIPPER, READY
from yam_arena.demo import ArmPlan, Waypoint
from yam_arena.urdf import axis_angle_matrix

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
TABLE_TOP = station.TABLE_TOP
SPEED = 0.25  # how fast a move goes when not given a time: m/s
TURN_SPEED = 45.0  # and degrees/s: the wrist is soft, and lags a faster turn by enough to dip the fingers
SERVO_STEP = 0.005  # the most a servo shifts its command in one step
IK_WARN = (5.0, 3.0)  # mm and degrees: a primitive worse than this says so, since the IK otherwise tilts the grip without a word
BLANK_STD = 12.0  # a frame's brightness spread: about 60 for a real view, about 5 for an overloaded GPU's flat grey


# -- grips ---------------------------------------------------------------------------------


class Grip(float):
    """A waypoint's yaw that carries a whole TCP rotation, which `ArmPlan` passes to `grip_rotation` without looking."""

    rot: np.ndarray = np.eye(3)


def _as_grip(rot: np.ndarray) -> Grip:
    g = Grip(0.0)
    g.rot = rot
    return g


def _install_grips() -> None:
    if getattr(harness_demo.grip_rotation, "_takes_grips", False):
        return
    plain = harness_demo.grip_rotation

    def grip_rotation(yaw=0.0):
        rot = getattr(yaw, "rot", None)
        return plain(yaw) if rot is None else rot

    grip_rotation._takes_grips = True
    grip_rotation._plain = plain
    harness_demo.grip_rotation = grip_rotation


_install_grips()
_plain_grip = harness_demo.grip_rotation._plain


def down(yaw_deg: float = 0.0) -> Grip:
    """Straight down, turned about the vertical. Yaw 0 closes the fingers along y. Fingers along x: -90 on the left arm, +90 on the right."""
    return _as_grip(_plain_grip(math.radians(yaw_deg)))


def tilted(theta_deg: float, phi_deg: float, flip: bool = False) -> Grip:
    """Tilted `theta_deg` from straight down towards the direction `phi_deg` from +x, the fingers closing across that direction. `flip` turns the gripper over about its own axis."""
    phi = math.radians(phi_deg)
    r = axis_angle_matrix(np.array([math.sin(phi), -math.cos(phi), 0.0]), math.radians(theta_deg)) @ _plain_grip(phi)
    return _as_grip(r @ axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.pi) if flip else r)


def level(phi_deg: float, flip: bool = False) -> Grip:
    """Fingers level, pointing along `phi_deg` from +x and closing across it, with the palm beside the object and its top open."""
    return tilted(90.0, phi_deg, flip)


# -- one arm -------------------------------------------------------------------------------


class Arm:
    """One arm. Every method but `tcp`, `at`, `approach_axis` is a generator with one yield per control step. A move starts from where the last command left the arm."""

    def __init__(self, ctx: "Ctx", side: str):
        self.ctx, self.side = ctx, side
        self.base = np.asarray(station.BASES[side], dtype=float)
        self.q = np.array(READY, dtype=float)
        self.finger = FINGER_OPEN
        self.cmd = np.r_[self.q, 1.0]
        self.grip = down()
        self.trim = np.zeros(3)  # what `servo` has found the arm needs on top of its goal: droop under a load
        self.worst = (0.0, 0.0)

    def tcp(self) -> np.ndarray:
        """Where the TCP is now, from the observed joints."""
        return ik.tcp_pose(self.ctx.observed(self.side)[:6], self.base)[0]

    def at(self) -> np.ndarray:
        """The position last commanded, without the trim: what a new move holds."""
        return ik.tcp_pose(self.q, self.base)[0] - self.trim

    def approach_axis(self, grip: Grip | None = None) -> np.ndarray:
        """The world direction in which the fingers point."""
        return (self.grip if grip is None else grip).rot[:, 2]

    def _note(self, mm: float, deg: float, label: str) -> None:
        self.worst = (max(self.worst[0], mm), max(self.worst[1], deg))
        if mm > IK_WARN[0] or deg > IK_WARN[1]:
            print(f"[demokit] {self.side} '{label}': IK missed by {mm:.1f} mm, {deg:.1f} deg: out of reach or over a joint limit", flush=True)

    def _solve(self, pos, rot, finger: float | None = None) -> tuple[float, float]:
        q, pe, re_ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        f = self.finger if finger is None else finger
        self.cmd = np.r_[q, float(np.clip(f / FINGER_OPEN, 0.0, 1.0))]
        return pe * 1000, float(np.degrees(re_))

    def move(self, pos, grip: Grip | None = None, finger: float | None = None, t: float | None = None, label: str = "move"):
        """Go to `pos`, on the harness's smooth path. With no `t`, at `SPEED` and `TURN_SPEED`. `grip` and `finger`, if given, change on the way."""
        grip = self.grip if grip is None else grip
        f = self.finger if finger is None else finger
        goal = np.asarray(pos, dtype=float) + self.trim
        if t is None:
            start, rot = ik.tcp_pose(self.q, self.base)
            turn = math.degrees(np.linalg.norm(ik._rot_error(grip.rot, rot)))
            t = max(0.4, float(np.linalg.norm(goal - start)) / SPEED, turn / TURN_SPEED)
        plan = ArmPlan([Waypoint(t, tuple(float(v) for v in goal), f, label, grip)], self.base, q_start=self.q, finger_start=self.finger)
        for k in range(1, max(1, round(t * CONTROL_HZ)) + 1):
            self.cmd = plan.action7(k * DT)[0]
            self.q = plan.q.copy()
            yield
        self.finger, self.grip = f, grip
        self._note(plan.worst_ik_mm, plan.worst_ik_deg, label)

    def servo(self, pos, tol: float = 0.003, timeout: float = 2.0, gain: float = 0.2, read=None, label: str = "servo"):
        """Shift the command until the TCP, seen from the observed joints, is within `tol` of `pos`: closes out droop and lag. Later moves keep the shift until `open`. Returns whether it got there.
        `read`, a function, gives another point to bring there, such as the end of a held part read from its pose. `pos` may be a function of the seconds since the start, for a goal that moves, such as a straight line into a slot: the servo then runs until the goal stops moving and the point is there."""
        c = ik.tcp_pose(self.q, self.base)[0]
        where = self.tcp if read is None else read
        good, worst = 0, (0.0, 0.0)
        err = np.zeros(3)
        goal, last = np.zeros(3), None
        for k in range(max(1, round(timeout * CONTROL_HZ))):
            goal = np.asarray(pos(k * DT) if callable(pos) else pos, dtype=float)
            still = last is not None and np.linalg.norm(goal - last) < 1e-6
            last = goal
            err = goal - where()
            good = good + 1 if np.linalg.norm(err) < tol and (still or not callable(pos)) else 0
            if good >= 3:
                break
            c = c + np.clip(gain * err, -SERVO_STEP, SERVO_STEP)
            mm, deg = self._solve(c, self.grip.rot)
            worst = (max(worst[0], mm), max(worst[1], deg))
            yield
        if read is None:
            self.trim = c - goal
        self._note(*worst, label)
        if good < 3:
            print(f"[demokit] {self.side} '{label}': servo stopped {np.linalg.norm(err) * 1000:.1f} mm short", flush=True)
        return good >= 3

    def follow(self, target, t: float = 2.0, grip: Grip | None = None, speed: float = 0.15, until=None, label: str = "follow"):
        """Track a point that moves: `target()` gives it, read every step. Ends after `t`, or once `until()` is true."""
        rot = (self.grip if grip is None else grip).rot
        c = ik.tcp_pose(self.q, self.base)[0]
        worst = (0.0, 0.0)
        for _ in range(max(1, round(t * CONTROL_HZ))):
            d = np.asarray(target(), dtype=float) + self.trim - c
            n = float(np.linalg.norm(d))
            c = c + d * min(1.0, speed * DT / max(n, 1e-9))
            mm, deg = self._solve(c, rot)
            worst = (max(worst[0], mm), max(worst[1], deg))
            yield
            if until is not None and until():
                break
        if grip is not None:
            self.grip = grip
        self._note(*worst, label)

    def arc(self, center, axis, deg: float, t: float | None = None, label: str = "arc"):
        """Swing the TCP and the grip together about a line through `center` along `axis`, by the right-hand rule: a door's edge, a lever's end. Any angle."""
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        center = np.asarray(center, dtype=float)
        p0, r0 = ik.tcp_pose(self.q, self.base)
        t = max(0.4, abs(deg) / TURN_SPEED) if t is None else t
        n = max(1, round(t * CONTROL_HZ))
        worst = (0.0, 0.0)
        rot = r0
        for k in range(1, n + 1):
            r = axis_angle_matrix(axis, math.radians(deg) * ik.smoothstep(k / n))
            rot = r @ r0
            mm, deg_err = self._solve(center + r @ (p0 - center), rot)
            worst = (max(worst[0], mm), max(worst[1], deg_err))
            yield
        self.grip = _as_grip(rot)
        self._note(*worst, label)

    def turn(self, deg: float, axis=(0.0, 0.0, 1.0), t: float | None = None, label: str = "turn"):
        """Turn the grip about `axis` through the TCP, holding the TCP still: a knob, a valve. A knob on +z follows `deg` with the sign of its joint."""
        yield from self.arc(ik.tcp_pose(self.q, self.base)[0], axis, deg, t, label)

    def open(self, t: float = 0.5, label: str = "open"):
        """Open fully, and forget the trim: a load let go takes its droop with it."""
        yield from self.move(self.at(), finger=FINGER_OPEN, t=t, label=label)
        self.trim = np.zeros(3)

    def close(self, width: float | None = None, force: float = 25.0, t: float = 0.5, label: str = "close"):
        """Close on something `width` across, with about `force` newtons a finger. With no width, fully."""
        f = FINGER_CLOSED if width is None else float(np.clip(-(width / 2 - force / GRIPPER["stiffness"]), FINGER_OPEN, FINGER_CLOSED))
        yield from self.move(self.at(), finger=f, t=t, label=label)

    def wait(self, t: float):
        for _ in range(max(1, round(t * CONTROL_HZ))):
            yield

    def pick(self, what, height: float, width: float | None = None, grip: Grip | None = None, above: float = 0.10, back: float = 0.0, lift: float = 0.08, force: float = 25.0, t: float | None = None, label: str = "pick"):
        """Come over a thing, by `above` over it and `back` behind it along the grip's axis, go down to `height` above its origin, close, and lift. `what` is a part's name, or a function giving a point. It is read again before each stage, so that the grip goes where the thing is. A level grip can come down round an upright thing with `back` 0. `t` is the time for the approach."""
        where = (lambda: self.ctx.pos(what)) if isinstance(what, str) else what
        grip = self.grip if grip is None else grip
        up = np.array([0.0, 0.0, height])
        off = np.array([0.0, 0.0, above]) - back * self.approach_axis(grip)
        yield from self.move(where() + up + off, grip=grip, finger=FINGER_OPEN, t=t, label=f"{label}: approach")
        yield from self.servo(where() + up + off, timeout=1.0, label=f"{label}: approach")
        yield from self.move(where() + up, label=f"{label}: in")
        yield from self.servo(where() + up, timeout=1.0, label=f"{label}: in")
        yield from self.close(width, force, label=f"{label}: close")
        yield from self.move(self.at() + np.array([0.0, 0.0, lift]), label=f"{label}: lift")

    def place(self, point, height: float, above: float = 0.05, back: float = 0.09, rise: float = 0.0, label: str = "place"):
        """Carry to over `point`, lower until the origin of what is held is on it, with the TCP `height` above, let go, and back away along the grip's axis. `rise` first lifts the open fingers: use it when they stand below the top of what they released."""
        goal = np.asarray(point, dtype=float) + np.array([0.0, 0.0, height])
        over = goal + np.array([0.0, 0.0, above])
        yield from self.move(over, label=f"{label}: over")
        yield from self.servo(over, timeout=1.0, label=f"{label}: over")
        yield from self.move(goal, t=0.8, label=f"{label}: down")
        yield from self.servo(goal, timeout=1.0, label=f"{label}: down")
        yield from self.open(label=f"{label}: open")
        if rise:
            yield from self.move(goal + np.array([0.0, 0.0, rise]), t=0.6, label=f"{label}: rise")
        yield from self.move(self.at() - back * self.approach_axis(), label=f"{label}: back")


# -- a demo's context ------------------------------------------------------------------------


class Ctx:
    """What a demo script gets: both arms, the live scene as a policy could see it, and `mode`, the cheat mode from the environment variable TASK_DEMO."""

    def __init__(self, emb, obs, mode: str):
        self.emb, self.obs, self.mode = emb, obs, mode
        self.left, self.right = Arm(self, "left"), Arm(self, "right")
        self.t = 0.0

    def arm(self, side: str) -> Arm:
        return self.left if side == "left" else self.right

    def observed(self, side: str) -> np.ndarray:
        """This arm's seven observed values: six joints and the gripper, 0 closed to 1 open."""
        from inspect_robots_yam.packing import STATE_KEY, split

        return split(np.asarray(self.obs.state[STATE_KEY], dtype=float))[0 if side == "left" else 1]

    def states(self) -> dict:
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def state(self, name: str):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, (name,))[name]

    def pos(self, name: str) -> np.ndarray:
        """A part's origin now, in the world. For a plain Box or Cylinder, its bounding box's centre."""
        return np.asarray(self.state(name).pos, dtype=float)

    def joint(self, part: str, link: str) -> float:
        return self.state(part).joint(link)

    def deg(self, part: str, link: str) -> float:
        return self.state(part).deg(link)

    def see(self, points) -> np.ndarray:
        """What the top camera shows now at these world points: RGB 0..255, one row a point. The render may be up to `--render-every` steps old, so hold still that long first."""
        from yam_arena.urdf import quat_xyzw_to_matrix

        if _BLIND:
            raise RuntimeError("ctx.see has nothing to read in a run with no cameras: run this script without --fast or --no-cameras")
        cam = self.emb.rig.top_cam
        image = self.obs.images["top_cam"]
        h, w = image.shape[:2]
        lens = cam.lens.resized(w, h)
        p = (np.atleast_2d(points) - np.asarray(cam.pos)) @ quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw))  # OpenGL: -z forward, +y up
        u = np.clip(np.round(lens.cx + lens.fx * p[:, 0] / -p[:, 2]).astype(int), 0, w - 1)
        v = np.clip(np.round(lens.cy - lens.fy * p[:, 1] / -p[:, 2]).astype(int), 0, h - 1)
        return np.asarray(image)[v, u].astype(float)

    def together(self, *generators):
        """Run primitives in lockstep, one step each per control step, until all have ended."""
        done = object()
        live = [iter(g) for g in generators]
        while live:
            live = [g for g in live if next(g, done) is not done]
            if live:
                yield

    def until(self, condition, timeout: float = 10.0, label: str = "until"):
        """Hold both arms until `condition()` is true. Returns whether it came true."""
        for _ in range(max(1, round(timeout * CONTROL_HZ))):
            if condition():
                return True
            yield
        print(f"[demokit] '{label}': not true after {timeout} s", flush=True)
        return False

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


# -- video -----------------------------------------------------------------------------------


def _ffmpeg() -> str:
    found = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class Recorder:
    """Streams each camera to its own file as frames arrive, and joins them at the end: the top camera large, the wrist cameras beside it, without text."""

    CAMS = ("top_cam", "left_cam", "right_cam")

    def __init__(self, out: Path, size: tuple[int, int]):
        self.out, self.size = out, size
        self.pipes = {c: subprocess.Popen([_ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(CONTROL_HZ), "-i", "-",
                                           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out / f"{c}.mp4")], stdin=subprocess.PIPE) for c in self.CAMS}
        self.frames, self.blank, self.setup = 0, 0, False

    def add(self, images: dict) -> None:
        self.frames += 1
        if max(images[c].std() for c in self.CAMS) < BLANK_STD:
            self.blank += 1
        elif not self.setup:
            _save_png(images["top_cam"], self.out / "setup.png")
            self.setup = True
        for c in self.CAMS:
            self.pipes[c].stdin.write(images[c].tobytes())

    def finish(self) -> Path:
        for p in self.pipes.values():
            p.stdin.close()
            p.wait()
        path = self.out / "demo.mp4"
        graph = "[0:v]scale=854:480:flags=lanczos[t];[1:v]scale=426:240:flags=lanczos[l];[2:v]scale=426:240:flags=lanczos[r];[l][r]vstack[s];[t][s]hstack[v]"
        subprocess.run([_ffmpeg(), "-loglevel", "error", "-y", *sum((["-i", str(self.out / f"{c}.mp4")] for c in self.CAMS), []), "-filter_complex", graph, "-map", "[v]",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
        return path


def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


# -- the runner ------------------------------------------------------------------------------

_BLIND = False


def _render_at(width: int, height: int) -> None:
    """Have the cameras really render at this size. The harness renders 640 x 360 whatever `cam_width` says, and resizes, so that a small `cam_width` alone saves nothing."""
    if "yam_arena.embodiment" in sys.modules:
        raise RuntimeError("the cameras are already built")
    import yam_arena.constants as constants

    constants.CAM_RENDER_WIDTH, constants.CAM_RENDER_HEIGHT = width, height


def _no_cameras() -> None:
    """Start the simulator with no camera sensors and no renderer. Observations then carry black images, so a script that reads the cameras cannot run this way. Nothing on disk changes."""
    global _BLIND
    _BLIND = True
    from isaaclab.app import AppLauncher
    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    start_app = AppLauncher.__init__

    def app_without_cameras(self, *a, **k):
        k["enable_cameras"] = False
        start_app(self, *a, **k)
        import yam_arena.embodiment as embodiment  # needs the app

        build = embodiment.BimanualYamEmbodiment.__init__

        def without_cameras(body, *a2, **k2):
            k2["enable_cameras"] = False
            build(body, *a2, **k2)

        embodiment.BimanualYamEmbodiment.__init__ = without_cameras

    AppLauncher.__init__ = app_without_cameras

    def blind_observation(self):
        from inspect_robots.types import Observation
        from inspect_robots_yam.config import DEFAULT_CAMERAS
        from inspect_robots_yam.packing import STATE_KEY, pack

        images = {name: np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8) for name in DEFAULT_CAMERAS}
        per_arm = []
        for name in ("left_robot", "right_robot"):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            grip = float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0))
            per_arm.append(np.append(q[arm_ids].cpu().numpy().astype(np.float64), grip))
        return Observation(images=images, state={STATE_KEY: pack(*per_arm)}, instruction=self._instruction, image_times=dict.fromkeys(images, self._t), state_time=self._t)

    YamArenaEmbodiment._observation = blind_observation


def _render_every(emb, n: int) -> None:
    """Render only one control step in `n`. Physics runs the same; images between renders are stale. For a debug run, where the script does not read images."""
    emb._base.cfg.sim.render_interval = emb._base.cfg.decimation * n


def _put_back(emb) -> None:
    """The harness's reset puts back the arms only: free objects and mechanisms stay as the last trial left them. So that several modes can share a session, put them back too."""
    base = emb._base
    for obj in base.scene.rigid_objects.values():
        pose = obj.data.default_root_pose.torch.clone()
        pose[:, :3] += base.scene.env_origins
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=obj.data.default_root_vel.torch.clone() * 0.0)
    for name, art in base.scene.articulations.items():
        if name.endswith("_robot"):
            continue
        art.write_joint_position_to_sim_index(position=art.data.default_joint_pos.torch.clone())
        art.write_joint_velocity_to_sim_index(velocity=art.data.default_joint_vel.torch.clone() * 0.0)


def _step_limit(task: Path) -> int:
    path = task.parent / "task.json"
    if path.exists():
        found = re.search(r"[\d,]+", str(json.loads(path.read_text()).get("run_limit", "")))
        if found:
            return int(found.group().replace(",", ""))
    return 2400


def play(emb, module, mode: str, out: Path, size: tuple[int, int], max_steps: int, hold_s: float, record: bool, stills: int, seed: int | None, render_every: int = 1) -> dict:
    """One trial of `demo_script`, in the session that `emb` holds."""
    from inspect_robots import Scene

    os.environ["TASK_DEMO"] = mode
    out.mkdir(parents=True, exist_ok=True)
    if emb._env is not None:
        _put_back(emb)
    obs = emb.reset(Scene(id=f"demo-{mode}", instruction=getattr(module, "INSTRUCTION", "demo")), seed=seed)
    if render_every > 1:
        _render_every(emb, render_every)
    ctx = Ctx(emb, obs, mode)
    script = module.demo_script(ctx)
    rec = Recorder(out, size) if record else None
    if rec:
        rec.add(obs.images)
    info: dict = {}
    steps, ended, success, reason = 0, None, False, None
    times: list[float] = []
    script_s, sim_s, worst_30 = 0.0, 0.0, 0.0
    start = last = time.perf_counter()
    while steps < max_steps:
        t0 = time.perf_counter()
        if ended is None:
            try:
                next(script)
            except StopIteration:
                ended = steps
                print(f"[demokit] {mode}: the script ended at step {steps}, {steps * DT:.1f} s", flush=True)
        t1 = time.perf_counter()
        result = emb.step(ctx.action())
        t2 = time.perf_counter()
        steps += 1
        ctx.t += DT
        ctx.obs = result.observation
        info = result.info
        script_s += t1 - t0
        sim_s += t2 - t1
        times.append(t2)
        if len(times) > 30:
            span = t2 - times.pop(0)
            if steps > 90:  # after the renderer's first frames
                worst_30 = max(worst_30, span)
        if rec:
            rec.add(result.observation.images)
        if stills and steps % stills == 0:
            for c, image in result.observation.images.items():
                _save_png(image, out / f"{c}_{steps:05d}.png")
        if steps % 100 == 0:
            now = time.perf_counter()
            print(f"[demokit] {mode}: step {steps}, {100 / (now - last):.1f} steps/s, score {info.get('score')}", flush=True)
            last = now
        if result.terminated:
            success, reason = result.termination_reason == "success", result.termination_reason
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    wall = time.perf_counter() - start
    video = str(rec.finish()) if rec else None
    res = {"mode": mode, "task_score": 1.0 if success else float(info.get("score") or 0.0), "success": success, "termination": reason, "steps": steps, "seconds": round(steps * DT, 2), "script_ended_at_step": ended,
           "size": list(size), "steps_per_s": round(steps / wall, 2), "script_s": round(script_s, 2), "sim_s": round(sim_s, 2), "wall_s": round(wall, 2), "slowest_30_steps_s": round(worst_30, 2),
           "blank_frames": rec.blank if rec else None, "video": video,
           "ik_worst": {a.side: [round(a.worst[0], 1), round(a.worst[1], 1)] for a in (ctx.left, ctx.right)},
           "final_info": {k: v for k, v in info.items() if isinstance(v, (int, float, bool))}}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[demokit] RESULT {mode}: task_score {res['task_score']}, success {success}, steps {steps} ({res['seconds']} s), {res['steps_per_s']} steps/s, script {res['script_s']} s, sim {res['sim_s']} s, "
          f"slowest 30 steps {res['slowest_30_steps_s']} s, IK worst {res['ik_worst']}" + (f", {rec.blank} blank frames" if rec else ""), flush=True)
    return res


def run(task: str, out: str, modes: list[str], *, cameras: bool = True, record: bool = True, size: tuple[int, int] = (640, 360), render: tuple[int, int] | None = None, max_steps: int | None = None, hold_s: float = 2.0,
        stills: int = 0, seed: int | None = None, jitter: float = 0.0, device: str = "cuda:0", render_every: int = 1) -> list[dict]:
    begin = time.perf_counter()
    path = Path(task).resolve()
    sys.path.insert(0, str(path.parent))
    if render:
        _render_at(*render)
    if not cameras:
        _no_cameras()
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    os.environ["TASK_DEMO"] = modes[0]
    emb = yam_arena_embodiment(task=str(path), object_jitter=jitter, control_hz=CONTROL_HZ, cam_width=size[0], cam_height=size[1], device=device)
    module = emb.task.module
    if not hasattr(module, "demo_script"):
        raise SystemExit(f"{path.name} has no demo_script(ctx)")
    limit = max_steps or _step_limit(path)
    results = []
    for k, mode in enumerate(modes):
        results.append(play(emb, module, mode, Path(out) / mode, size, limit, hold_s, record, stills, seed, render_every))
        if k == 0:
            print(f"[demokit] up to the end of the first mode: {time.perf_counter() - begin:.0f} s", flush=True)
    return results


def reach(x: float, y: float, height: float) -> None:
    """Which grips each arm can hold here. IK from the ready pose, within 2 mm and 2 degrees. No collisions."""
    top = TABLE_TOP
    phis = list(range(-90, 181, 30))

    def holds(side: str, grip: Grip) -> bool:
        plan = ArmPlan([Waypoint(2.0, (x, y, top + height), FINGER_OPEN, "go", grip)], station.BASES[side])
        for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            plan.worst_ik_mm, plan.worst_ik_deg = 0.0, 0.0  # what counts is the error at the end
            plan.action7(t)
        return plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0

    print(f"TCP at x {x}, y {y}, {height} above the table. Angles that solve:")
    for side in ("left", "right"):
        print(f"  {side}: down, yaw " + " ".join(str(a) for a in range(-120, 121, 30) if holds(side, down(a))))
        for theta in (45, 60, 90):
            print(f"  {side}: tilt {theta}, phi " + " ".join(str(p) for p in phis if holds(side, tilted(theta, p))) + "  flipped " + " ".join(str(p) for p in phis if holds(side, tilted(theta, p, True))))


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "reach":
        ap = argparse.ArgumentParser(prog="demokit.py reach")
        ap.add_argument("x", type=float)
        ap.add_argument("y", type=float)
        ap.add_argument("height", type=float)
        a = ap.parse_args(sys.argv[2:])
        reach(a.x, a.y, a.height)
        return
    ap = argparse.ArgumentParser(description="Run a task file's demo_script. See this file's docstring.")
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--mode", default=os.environ.get("TASK_DEMO", "full"), help="a cheat mode, or several with commas, which share one simulator session")
    ap.add_argument("--no-record", action="store_true", help="debug: no video, and the cameras really render at --size, one step in 20")
    ap.add_argument("--size", type=_size, help="WIDTHxHEIGHT of the observations: default 640x360, and 320x176 with --no-record")
    ap.add_argument("--render", type=_size, help="WIDTHxHEIGHT the cameras really render at, if not what the mode above says")
    ap.add_argument("--render-every", type=int, help="render one control step in N. Default 1; with --no-record, 20, or --stills if given")
    ap.add_argument("--stills", type=int, default=0, help="write a picture from each camera every N steps")
    ap.add_argument("--max-steps", type=int, help="default: the run limit in task.json, else 2400")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold after the script ends")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--jitter", type=float, default=0.0, help="metres free objects shift at a seeded reset: closed-loop grasps should cope")
    ap.add_argument("--device", default="cuda:0", help="where physics runs: cuda:0 or cpu")
    ap.add_argument("--no-cameras", action="store_true", help="no camera sensors and no renderer, with --no-record: observations are black")
    ap.add_argument("--fast", action="store_true", help="--no-cameras and --device cpu, for debugging motion and scorers. Final demos and model runs stay on the GPU, with cameras")
    a = ap.parse_args()
    if a.fast:
        a.no_cameras, a.device = True, "cpu"
    if a.no_cameras and a.stills:
        ap.error("--stills needs cameras")
    if a.no_cameras:
        a.no_record, a.render, a.render_every = True, None, 1
    size = a.size or ((320, 176) if a.no_record else (640, 360))
    render = a.render or (size if a.no_record and not a.no_cameras else None)
    every = a.render_every or ((a.stills or 20) if a.no_record else 1)
    results = run(a.task, a.out, a.mode.split(","), cameras=not a.no_cameras, record=not a.no_record, size=size, render=render, max_steps=a.max_steps, hold_s=a.hold, stills=a.stills, seed=a.seed, jitter=a.jitter, device=a.device, render_every=every)
    if any(r["blank_frames"] for r in results):
        print("BLANK FRAMES: the cameras saw nothing on some steps, as happens when runs overlap on a GPU. Run it again with nothing else on the GPU.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
