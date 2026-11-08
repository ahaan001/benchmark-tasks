# Mix a cloudy insulin vial by rolling it ten full turns without lifting it off the mat

A cloudy insulin vial stands in a taped square on a medication mat, with an identical clear vial 10 cm to its
right. The robot is asked to mix it by rolling it ten full turns on the mat without lifting or tipping it, and to
leave the clear vial alone. The vial stands up, so rolling is turning it about its own axis in place; the wrist
turns about 210 degrees at a time, so the robot must pinch, turn, let go and pinch again many times. Each full
turn earns 0.1; a lifted, tipped or displaced vial, or a disturbed clear vial, scores 0; doing nothing scores 0.

## Files

- `insulin_vial_roll.py`: the scene, the scoring hooks and a scripted demo (19 pinch-and-turn cycles, score 1.0).
- `task.json`: the task's details.
- `check.py`: offline check of the layout, the wrist's turning range, the scoring and the demo.
- `trace.py`: the same task with the demo's tracking error, the vial's turn, lift, tilt and offset, the clear
  vial's shift and the tracker's state printed every 5 steps, for tuning the demo.

Environment: Isaac Lab Arena 0.3.1, Isaac Sim 6.1, inspect-robots 0.60.0, yam-isaac-lab-arena at 0059d02. The
scene uses boxes and cylinders only, no library models and no third-party assets.

## Run a trial

Run from the repository root.

```bash
TASK=tasks/insulin-vial-roll/insulin_vial_roll.py
INSTR="Mix the cloudy insulin vial by rolling it ten full turns on the mat. Do not lift or tip it, and leave the clear vial where it is."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dream/insulin-vial-roll

# Scripted demo (ten turns)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/demo/insulin-vial-roll
```

The run prints `task_score`. Render the camera videos with `inspect-robots video <log>.json`. Add
`-E headless=false` to watch a run in the Isaac Sim window.

Offline check, without the simulator (about a minute):

```bash
python tasks/insulin-vial-roll/check.py
```

It measures how far the grip can turn at the grasp pose before a wrist joint reaches its limit, the room between
the robot's meshes and the clear vial, the mat, the needle box, the meter and the sharps container at every grip
angle the demo uses, scores 24 hand-built trials, plays the demo through the IK at 20 Hz with the vial following
the grip while it is pinched, and plays a variant that lifts the vial during each twist.

## Scoring

The score is read after every control step from the two vials' poses and the arms' measured joints.

Score 0, for the rest of the trial, once any of these happens:

- **Cloudy vial lifted:** its lowest point more than 2 cm above the mat.
- **Cloudy vial tipped:** its axis more than 15 degrees from upright.
- **Cloudy vial out of the square:** its centre more than 3 cm from the square's centre along x or y.
- **Clear vial disturbed:** its centre more than 1 cm from where it stood, or turned more than 20 degrees.

Otherwise the cloudy vial's rotation about its own axis is added up over the trial, in both directions, from
its yaw at each control step (the change between steps is always far below a half turn), but only while its
lowest point is within 5 mm of the mat. Each completed full turn is worth 0.1, up to 1.0 at ten. At the end the
vial must be let go (every point of both grippers' collision meshes, placed from the measured joints, at least
2 cm from it) and standing inside the square, or the score is 0.

`success()` is the 1.0 state: ten or more full turns, the vial let go, upright, inside the square and at rest,
with no rule broken. Held for 1 s (`SUCCESS_HOLD_S`) it ends the episode. Otherwise `task_score` is `score()` on
the last step. Possible scores: 0, 0.1, ..., 1.0.

## Design notes

- **The object never travels.** The skill is a cycle (pinch, turn to the wrist's limit, let go, turn back,
  pinch again) repeated until the count is reached. Offline, with a 3 degree margin on every joint, the grip at
  the grasp pose turns from -90 to +140 degrees; the demo uses -80 to +130, 210 degrees per pinch, 19 pinches
  for 3,990 degrees (ten turns plus 10 percent for slip between the pads and the vial).
- **Nothing holds the vial.** A valve, a cap or a screw keeps itself in place while it is turned; a free vial
  stays upright and on its spot only through the grip. Lifting it to turn it freely, or pinching it off centre so
  it tips, is what the two latched rules catch. That is the "do not shake" rule in measurable form.
- **Partial credit is the count.** Each full turn earns 0.1, so the task reports how far a policy got rather
  than pass or fail. Rotation while the vial is lifted 5 mm to 2 cm earns nothing but breaks no rule; above 2 cm
  the trial is over.
- **Both directions count.** The label asks for mixing, not a direction, and a policy that turns the vial back
  and forth mixes it as well as one that always turns the same way.
- **The clear vial is a perception test.** It is identical but for its contents and stands 10 cm away, within a
  careless sweep of the open fingers (the demo keeps 47 mm from it). Brushing it without moving it 1 cm or
  turning it 20 degrees has no effect, by design.
- **Nothing can be undone.** Broken rules are kept until the episode's next reset, which the scorer reads from
  the embodiment's episode clock going back to 0. Putting a vial back where it started clears nothing
  (`check.py` tests this, and two episodes back to back).
- **Letting go is part of finishing.** A vial still within 2 cm of a gripper at the end scores 0, however many
  turns it made.
- **The other items are dressing.** The pen-needle box, the meter and the sharps container make the scene read
  as a home injection station; knocking them about is not scored.
- **The demo is open loop** from the known start layout and reads nothing from the simulator while it runs.
  Each turn is split in two waypoints, because the plan interpolates rotations the short way round. The lift
  variant of the demo (`demo("lift")`, offline only) raises the vial 3 cm during each twist and scores 0 with
  the lift rule latched at the first twist.

## Runtime and seeds

- **Run limit:** 2,400 control steps (2 minutes at 20 Hz).
- **Demo:** the plan lasts 89.5 s (1,790 steps; about 4.5 s per pinch-and-turn cycle). In Isaac Sim it succeeds
  at step 1,782 (89.1 s), is deterministic, and took about 10 minutes of wall time on one L40S that two other
  simulations were sharing (the first twist turned the vial 201 of the commanded 210 degrees; drift under 2 mm).
- **Seeds:** none. The layout is fixed (`object_jitter=0`), and the scripted demo is deterministic.

## Score a real trial

A real trial needs a ruler and the trial's video.

1. Stand the vials on their marks with the cloudy vial's line facing away from the arms. Film the trial with a
   camera that sees the mat (the rig's top camera does). Take a top photo at the end.
2. Score 0 if the video shows the cloudy vial's base more than 2 cm off the mat at any time, the vial leaning
   more than 15 degrees, or its centre outside the taped square; or the clear vial moved more than 1 cm or
   turned more than 20 degrees.
3. Count the cloudy vial's full turns from the line on its label, in either direction, ignoring any turning
   done while the vial was off the mat by more than 5 mm.
4. At the end, score 0 if any part of a gripper is within 2 cm of the vial or the vial's centre is outside the
   square. Otherwise the score is 0.1 per full turn, at most 1.0.

Counting turns by eye from a line on the label is less exact than the simulator's pose reading. That is accepted
in exchange for a setup anyone can build from a mat, tape, two vials and a few household items.
