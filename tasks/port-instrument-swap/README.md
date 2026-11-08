# Swap the instrument in the port without disturbing the port

A draped silicone abdomen has two ports. One holds a fixed scope. In the other, a 2 cm hole between two loose red
flanges, stands a grey instrument: a 1.2 cm rod. The robot is asked to take it out, lay it in the steel tray, and
put the blue instrument from the stand down the same hole until it stops, without disturbing the port. The hole
has 4 mm of room each side of a rod, so each stroke must run straight up or down its axis: a rod 2 to 4 mm off the
axis touches a flange, further off pushes it. The hole takes one rod at a time, so the old instrument must be fully
out before the new one can seat. A touch of a flange halves the score, a third touch or a pushed flange scores 0,
and doing nothing scores 0.

## Files

- `port_instrument_swap.py`: the scene, the scoring hooks and a scripted demo (the full exchange, score 1.0).
- `task.json`: the task's details.
- `check.py`: offline check of the layout, the camera view, the scoring and the demo.
- `trace.py`: the same task with the demo's tracking error, the rods' positions, the distances to the flanges
  and the tracker's state printed every 5 steps, for tuning the demo.

Environment: Isaac Lab Arena 0.3.1, Isaac Sim 6.1, inspect-robots 0.60.0, yam-isaac-lab-arena at 0059d02. The
scene uses boxes only, no library models and no third-party assets.

## Run a trial

Run from the repository root.

```bash
TASK=tasks/port-instrument-swap/port_instrument_swap.py
INSTR="Take the grey instrument out of the port between the two red flanges and lay it in the steel tray, then put the blue instrument from the stand down the same port until it stops, without disturbing the port."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dream/port-instrument-swap

# Scripted demo (the full exchange)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/demo/port-instrument-swap
```

The run prints `task_score`. Render the camera videos with `inspect-robots video <log>.json`. Add
`-E headless=false` to watch a run in the Isaac Sim window.

Offline check, without the simulator (about 20 s):

```bash
python tasks/port-instrument-swap/check.py
```

It measures a rod's room to the flanges on and off the hole's axis, checks the gripper's room at the grasp and
what the top camera sees, scores 26 hand-built trials, plays the demo through the IK at 20 Hz with the held rod
following the gripper, and plays the angled variant of the demo (strokes 1 cm and 3 mm off the axis) to count its
touches.

## Scoring

The score is read after every control step from the poses of the flanges and the rods and from the arms'
measured joints.

Score 0, for the rest of the trial, once any of these happens:

- **A flange pushed:** its centre more than 1 cm from its mark, or tilted more than 15 degrees, or down.
- **Third touch.** A touch starts when any point of either rod's surface (checked every 2.5 mm) or of the robot
  (both arms, placed from their measured joints and the URDF's collision meshes, checked every 4 mm) is within
  2 mm of a flange, or when a flange moves (off its mark by 3 mm or 3 degrees, or not at rest), and ends once
  both flanges are at rest and everything is 3.5 mm from them again. A flange left standing 5 mm off its mark
  counts once.

Otherwise the score is the sum of two parts, halved once for each touch:

- **0.3** if the grey rod is at rest inside the tray's walls, any way up, with every part of both grippers 2 cm
  or more from it. 0 anywhere else, or still held.
- **0.7** if the blue rod stands in the port: centre over the hole, base within 5 mm of the hole's floor, tilted
  less than 5 degrees, at rest, let go. **0.35** if it is in the port (centre over the hole, base below the skin)
  but short of the floor, leaning or moving, and let go. 0 anywhere else, or still held.

`success()` is the 1.0 state: the grey rod at rest in the tray, the blue rod seated, both let go, both flanges
within 3 mm and 5 degrees of their marks and at rest, no touch and no 0 condition. Held for 1 s
(`SUCCESS_HOLD_S`) it ends the episode. Otherwise `task_score` is `score()` on the last step. Possible scores: 0,
0.075, 0.0875, 0.15, 0.1625, 0.175, 0.25, 0.3, 0.325, 0.35, 0.5, 0.65, 0.7, 1.

## Design notes

- **Two tight vertical strokes through loose walls.** The hole leaves 4 mm each side of a rod. A rod drawn out or
  lowered 2 to 4 mm off the axis is within 2 mm of a flange (a touch); 4 mm or more off, it hits one. Any lateral
  error during either stroke is recorded, not only the end state.
- **The order is enforced by the hole, not by a rule.** Two rods do not fit a 2 cm hole, so the blue rod cannot
  seat until the grey one is out; a rod set on top of the seated grey rod is not in the port.
- **The natural shortcuts meet the hazard.** Drawing the grey rod out at a slight angle, or bringing the blue rod
  down a few millimetres off the axis, touches a flange: the angled variant of the demo (`check.py`) records
  touches on both strokes. Carrying a rod low across the port hits the flanges, which stand 2 cm above the skin.
- **Nothing holds the port.** A real cannula is held by the abdominal wall and a trocar seal; here two loose
  15 g flanges stand for it, so any load on the port shows as a flange moving.
- **A touch halves, it does not zero.** Brushing a flange is a lesser harm than tearing the port site, so the
  scores spread out. A flange moved 1 cm or tilted 15 degrees scores 0.
- **Nothing can be undone.** Touches and a pushed flange are kept until the episode's next reset, which the
  scorer reads from the embodiment's episode clock going back to 0. Standing a flange back on its mark clears
  nothing (`check.py` tests this, and two episodes back to back).
- **Release is part of placing.** A rod still in the pads earns nothing, read from the grippers' collision
  meshes 2 cm from the rod.
- **The demo is open loop** from the known start layout and reads nothing from the simulator while it runs.
  Grip yaw 0 throughout (the fingers close along y); each stroke through the hole takes 3 s in two legs so the arm
  settles before the pads pass the flanges.

## Runtime and seeds

- **Run limit:** 1,200 control steps (1 minute at 20 Hz).
- **Demo:** the plan lasts 37.5 s. In Isaac Sim it succeeds at step SUCCESS_STEP (SUCCESS_S s), about WALL_S s
  of wall time on one L40S including the simulator's start-up, and is deterministic.
- **Seeds:** none. The layout is fixed (`object_jitter=0`), and the scripted demo is deterministic.

## Score a real trial

A real trial needs a ruler and a camera. Tape the flanges' outlines on the skin before the trial. Watch the
video for any contact of the robot or an instrument with a flange, and for a flange moving on its outline; a
flange more than 1 cm off its outline or tilted more than 15 degrees at any moment scores 0. At the end, measure
the blue rod's protrusion above the skin (6 cm when seated; more than 6.5 cm is short of the floor), its tilt
with a square, whether the grey rod lies inside the tray, and whether both grippers are clear of both rods.
