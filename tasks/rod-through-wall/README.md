# Pass a rod through the tightest window you can

A wall with three square windows (8, 3.5 and 5 cm) stands between the arms. The robot must pass a 26 cm rod
through a window to a stand on the other side, without touching the wall or dropping the rod. Neither gripper
fits through a window, so the left arm feeds the rod and the right arm takes it. The smallest window scores 1.0
and leaves 7.5 mm of clearance on every side of the rod. The largest scores 0.4 and leaves 3 cm. Doing nothing
scores 0.

Files:

- `rod_through_wall.py`: the scene, the scoring hooks and a scripted demo (small window, score 1.0).
- `task.json`: the task's details.
- `check.py`: offline check of the scoring and of the demo's IK.
- `arm_clearance.py`: offline check of the room between the arms, the wall and the stand in the demo.
- `trace.py`: the same task with the rod's pose and clearance printed every 10 steps, for tuning the demo.

Environment: Isaac Lab Arena 0.3.1, Isaac Sim 6.1, inspect-robots 0.60.0, yam-isaac-lab-arena at 0059d02. The
scene uses boxes only, no library models.

## Offline checks (no simulator)

```bash
python tasks/rod-through-wall/check.py
python tasks/rod-through-wall/arm_clearance.py
```

`check.py` scores hand-built paths of the rod: each window to the stand, a rod set down behind the wall, a rod
held half way through, a pass 1 mm from a pillar, a drop, a pass over the top and one round the end. It then
plays both arms' plans through the IK at 20 Hz with the rod following the gripper that holds it: the run must
score 1.0, and the IK error and the joint limits must be within bounds. `arm_clearance.py` places the arms'
collision meshes along the demo and prints the smallest distances (about 1.5 minutes).

## Run a trial

Run from the repository root. The task needs `-E object_jitter=0`: the scoring starts over whenever the rod lies
at its start pose, which is true after every reset of a fixed layout.

```bash
TASK=tasks/rod-through-wall/rod_through_wall.py
INSTR="Pass the rod through the wall to the blue stand on the other side, using the smallest window you can, without touching the wall or dropping the rod."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dream/rod-through-wall

# Scripted demo (small window, about 27 s)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/demo/rod-through-wall
```

The run prints `task_score`. Render the camera videos with `inspect-robots video <log>.json`. Add
`-E headless=false` to watch a run in the Isaac Sim window.

- **Runtime per trial:** up to 1,200 control steps (1 minute at 20 Hz). The demo plan lasts 26.5 s and ends
  when it succeeds.
- **Seeds:** none. The layout is fixed (`object_jitter=0`), and the scripted demo is deterministic.

## Scoring

The score is read from the rod's pose after every control step. Positions are taken in the wall's frame: along
the wall, out of its near face, and height above the table.

- **Touch (score 0):** at any step, a point of the rod's surface is within 2 mm of the wall. The rod's surface
  is checked every 2.5 mm.
- **Drop (score 0):** at any step, the rod's centre moves down faster than 0.5 m/s (a free fall of about
  1.3 cm). This also applies behind the wall and onto the stand.
- **Not through (score 0):** at the end the rod's centre is not behind the wall's mid-plane, or its last pass to
  the far side was not through a window. A pass is the step at which the rod's centre crosses the mid-plane. It
  is through a window when the part of the rod inside the wall's thickness lies within that window. Over the top
  and round an end are not.
- **Otherwise:** the value of the window of the last pass (large 0.4, medium 0.7, small 1.0) if the rod rests on
  the stand, and a quarter of it if the rod is anywhere else. On the stand means: at rest, centre over the
  stand's top face, lowest point within 5 mm of it.

`task_score` gives 1.0 when `success()` holds for 3 s (`SUCCESS_HOLD_S`): the rod on the stand after a pass
through the small window, with no touch and no drop. The 3 s outlast the release and the retreat, so the rod must
stay put after the gripper leaves. Otherwise `task_score` is `score()` on the last step.

The hooks get no reset signal, so the record of touches, drops and passes starts over whenever the rod lies at
its start pose (centre within 1 cm, direction within 5 degrees).

## Score a real trial

1. Tape aluminium foil on the window edges and on the rod, and wire them as a buzz-wire circuit that logs any
   contact. Chalk on the window edges, checked on the rod after the trial, also works.
2. Score 0 if the circuit closed at any time, or if the rod left the grippers and fell more than 1.3 cm.
3. Score 0 if the mark at the rod's middle has not passed the wall at the end, or if the rod last passed the
   wall over its top or round an end.
4. Read the window of the last pass from the video: large 0.4, medium 0.7, small 1.0.
5. Give that value if the rod lies at rest on the blue stand, out of the gripper, with its middle mark over the
   block (top photo). Give a quarter of it otherwise.
