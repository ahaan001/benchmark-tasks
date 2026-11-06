# Pass a rod through the tightest window you can

A wall with three square windows (8, 3.5 and 5 cm) stands between the arms. The robot must pass a 26 cm rod
through a window to a stand on the other side, without touching the wall or dropping the rod. Neither gripper
fits through a window, so one arm feeds the rod and the other takes it. The smallest window scores 1.0 and leaves
7.5 mm of clearance on every side of the rod. The largest scores 0.4 and leaves 3 cm. Doing nothing scores 0.

## Files

- `rod_through_wall.py`: the scene, the scoring hooks and a scripted demo (small window, score 1.0).
- `task.json`: the task's details.
- `check.py`: offline check of the scoring and of the demo's IK.
- `arm_clearance.py`: offline check of the room between the arms, the wall and the stand in the demo.
- `trace.py`: the same task with the rod's pose and clearances printed every 5 steps, for tuning the demo.

Environment: Isaac Lab Arena 0.3.1, Isaac Sim 6.1, inspect-robots 0.60.0, yam-isaac-lab-arena at 0059d02. The
scene uses boxes only, no library models and no third-party assets.

## Run a trial

Run from the repository root.

```bash
TASK=tasks/rod-through-wall/rod_through_wall.py
INSTR="Pass the rod through the wall to the blue stand on the other side, using the smallest window you can, without touching the wall or dropping the rod."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dream/rod-through-wall

# Scripted demo (small window)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/demo/rod-through-wall
```

The run prints `task_score`. Render the camera videos with `inspect-robots video <log>.json`. Add
`-E headless=false` to watch a run in the Isaac Sim window.

Offline checks, without the simulator:

```bash
python tasks/rod-through-wall/check.py
python tasks/rod-through-wall/arm_clearance.py
```

`check.py` scores 20 hand-built trials and then plays both arms' plans through the IK at 20 Hz with the rod
following the gripper that holds it. `arm_clearance.py` places the arms' collision meshes along the demo and
prints the smallest distances (about 1 minute).

## Scoring

The score is read after every control step from the rod's pose and the arms' measured joints. Positions are
taken in the wall's frame: along the wall, out of its near face, and height above the table.

Score 0, for the rest of the trial, once any of these happens:

- **Touch by the rod:** a point of the rod's surface is within 2 mm of the wall. The surface is checked every
  2.5 mm.
- **Touch by the robot:** a point of either arm (links, gripper, fingers) is within 5 mm of the wall. The arms
  are placed from their measured joints and the URDF's collision meshes, checked every 4 mm.
- **Drop:** the rod's centre moves down faster than 0.5 m/s (a free fall of about 1.3 cm), also behind the wall
  and onto the stand.

Score 0 at the end if:

- the rod's centre is not behind the wall's mid-plane, or
- the rod's last pass to the far side was not through a window. A pass is the step at which the rod's centre
  crosses the mid-plane. It is through a window when the part of the rod inside the wall's thickness lies within
  that window. Over the top and round an end are not.

Otherwise the window of the last pass sets the value (large 0.4, medium 0.7, small 1.0):

- **the value** if the rod is at rest on the stand (centre over the stand's top face, lowest point within 5 mm
  of it) and every part of both grippers is at least 3 cm from it,
- **a quarter of the value** if the rod is anywhere else behind the wall: still held, on the table, or still in
  the window.

`success()` is the 1.0 state held for 1 s (`SUCCESS_HOLD_S`), which ends the episode. Otherwise `task_score` is
`score()` on the last step. Possible scores: 0, 0.1, 0.175, 0.25, 0.4, 0.7, 1.0.

## Design notes

- **The start pose favours no window.** The rod points at the 2.5 cm strip of wall between the largest and the
  smallest window. A straight push hits the wall and scores 0, so the policy has to choose a window and line the
  rod up with it.
- **Reward and risk rise together.** Clearance on each side of the rod is 3 cm (large), 1.5 cm (medium) and
  7.5 mm (small). The stand is in line with the small window, 16 cm behind the wall.
- **The hand-off is forced.** The open fingers are 14 cm across, so no gripper passes a window, and the wall
  keeps each arm on its own side.
- **A touch or a drop cannot be undone.** Both are kept until the episode's next reset, which the scorer reads
  from the embodiment's episode clock going back to 0. Putting the rod back at its start does not clear them
  (`check.py` tests this).
- **The robot counts.** "Without touching the wall" covers the arms and grippers as well as the rod.
- **The route counts.** Only the rod's last pass to the far side is scored, so a rod carried round the wall
  scores 0, and a rod pulled back out of one window and passed through another scores the second window.
- **Placing includes letting go.** A rod held on the stand, or held anywhere behind the wall, earns a quarter of
  the window's value.
- **Wall height and hand-off distance.** In a straight-down grip the wrist link reaches 9.8 cm ahead of the TCP
  from 14 cm above it. The wall is 16 cm high and both grippers stop 8.5 cm from it, so the wrists pass above
  the wall and clear of each other (`arm_clearance.py`).
- **The demo is open loop** from the known start pose. Its constants for the wrist's twist under the grip and
  the rod's sag were measured once in the simulator; it reads nothing from the simulator while it runs.

## Runtime and seeds

- **Run limit:** 1,200 control steps (1 minute at 20 Hz).
- **Demo:** the plan lasts 27.5 s. It ends in success after 515 steps (25.8 s simulated), about 85 s of wall
  time on one L40S including start-up.
- **Seeds:** none. The layout is fixed (`object_jitter=0`), and the scripted demo is deterministic.

## Score a real trial

1. Tape aluminium foil on the panel and on the rod, and wire them as a buzz-wire circuit that logs any contact.
   Chalk on the window edges, checked on the rod after the trial, also works. Film the trial.
2. Score 0 if the circuit closed at any time, if the video shows any part of the robot touching the wall, or if
   the rod left the grippers and fell more than 1.3 cm.
3. Score 0 if the mark at the rod's middle has not passed the wall at the end, or if the rod last passed the
   wall over its top or round an end.
4. Read the window of the last pass from the video: large 0.4, medium 0.7, small 1.0.
5. Give that value if the rod lies at rest on the blue stand with its middle mark over the block and both
   grippers at least 3 cm away. Give a quarter of it otherwise.
