# Lay a sterile dressing over the wound, handling it only by its edges

A white sterile dressing, 7.5 x 7.5 cm and 4 mm thick, lies on the opened pack's 5 x 5 cm tray so that its edges
overhang. A red 4 x 3 cm wound lies on a skin pad 20 cm away. The robot is asked to lay the dressing flat over the
wound so it is completely covered, touching it only by its edges. The dressing's central 5 x 5 cm, top and
underside, is the key part: any part of the robot within 2 mm of it halves the score, a second touch scores 0.
The dressing must end flat on the skin, let go and at rest; sliding it more than 2 cm across the skin halves the
score; doing nothing scores 0.

## Files

- `dressing_by_edges.py`: the scene, the scoring hooks and a scripted demo (edge grasp, flat placement, score 1.0).
- `task.json`: the task's details.
- `check.py`: offline check of the layout, the camera view, the scoring and the demo.
- `trace.py`: the same task with the demo's tracking error, the dressing's pose, the tracker's state and the score
  printed every 5 steps, for tuning the demo.

Environment: Isaac Lab Arena 0.3.1, Isaac Sim 6.1, inspect-robots 0.60.0, yam-isaac-lab-arena at 0059d02. The
scene uses boxes and cylinders only, no library models and no third-party assets.

## Run a trial

Run from the repository root.

```bash
TASK=tasks/dressing-by-edges/dressing_by_edges.py
INSTR="Take the white dressing from its opened pack and lay it flat over the red wound so the wound is completely covered, touching the dressing only by its edges, then let go."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dream/dressing-by-edges

# Scripted demo (edge grasp, flat placement over the wound)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/demo/dressing-by-edges
```

The run prints `task_score`. Render the camera videos with `inspect-robots video <log>.json`. Add
`-E headless=false` to watch a run in the Isaac Sim window.

Offline check, without the simulator (a few minutes):

```bash
python tasks/dressing-by-edges/check.py
```

It measures the room the gripper has at the dressing on the pack and over the wound and its distance to the key
part while grasping, carrying and releasing, which of the demo's sites each arm reaches, and what the top camera
sees past the arms; scores 24 hand-built trials; plays the demo through the IK at 20 Hz with the dressing following
the gripper that holds it; and plays the nudge variant of the demo, which sets the dressing down 3 cm off and
pushes it into place.

## Scoring

The score is read after every control step from the dressing's pose and from the arms' measured joints.

Halved, for the rest of the trial, each time the robot touches the key part: any point of the robot's collision
meshes (the URDF's, placed from the measured joints, sampled every 2 mm) within 2 mm of the dressing's central
5 x 5 cm, top or underside. A touch ends once every point is 5 mm away again; a second touch scores 0 for the rest
of the trial.

Otherwise, from the final state:

- **0** unless the dressing lies flat on the skin pad: its centre over the pad, its underside within 3 mm of the
  pad's top (it may rest on the 2 mm wound insert), tilted less than 5 degrees, let go (every part of both
  grippers 2 cm or more from it) and at rest.
- **1.0** if all four corners of the wound are under the dressing's footprint, **0.5** if the wound's centre is
  under it but a corner is not, **0** if the wound's centre is uncovered.
- **Halved** if, while lying on the skin, the dressing was slid more than 2 cm from where it came to rest.
  Lifting it and setting it down again is not a slide.

`success()` is the 1.0 state: wound fully covered, dressing flat, let go and at rest, with no key-part touch and
no slide. Held for 1 s (`SUCCESS_HOLD_S`) it ends the episode. Otherwise `task_score` is `score()` on the last
step. Possible scores: 0, 0.125, 0.25, 0.5 and 1.0.

## Design notes

- **The dressing cannot be picked from a flat surface.** It is 4 mm thick and 7.5 cm wide, so a parallel gripper
  can only hold it by its two edge faces, and only where they overhang the pack tray: the open fingers (9.4 cm
  between the pads) come down with 9.9 mm of room on each side, 22 mm from the tray, and close on the edge faces
  with the pad tips 1 mm below the dressing's underside (`check.py`).
- **The key part turns carry-and-place into a contact constraint.** The fingers must stay out of the central
  5 x 5 cm while grasping (1 cm away when closed on the edges), carrying and releasing (2.2 cm away with the
  fingers open), and a finger pad that rides onto the dressing's top face in a crooked grasp is a touch.
- **The natural correction costs half.** A dressing set down off the wound is most easily fixed by pushing it
  into place. The nudge variant of the demo does exactly that (3 cm off, then pushed) and scores 0.5.
- **Placement is scored by coverage.** The wound is 4 x 3 cm under a 7.5 cm dressing, so a centred placement has
  1.75 cm of margin along x and 2.25 cm along y; 2 cm off along x uncovers a corner (0.5), 5 cm off uncovers the
  centre (0).
- **Release is part of placing.** The dressing earns nothing while any part of a gripper is within 2 cm of it;
  the fingers opened around it are 9.9 mm away, so the gripper has to rise.
- **Nothing can be undone.** Touches and the slide are kept until the episode's next reset, which the scorer reads
  from the embodiment's episode clock going back to 0 (`check.py` plays two episodes back to back).
- **The scene is a dressing trolley.** Drape, skin pad with a wound insert, opened pack, tape roll, saline bottle
  and glove box. The tape, saline and gloves are not scored; the tape roll is loose and may be knocked over.
- **One arm.** The right arm reaches the pack, the wound and the carry path; the left arm reaches neither site
  (`check.py` solves the IK for both arms). The glove box stands where the top camera sees it past the right arm
  at its ready pose.
- **The demo is open loop** from the known start layout and reads nothing from the simulator while it runs: it
  pinches the dressing's edges beside the tray, lifts it to 10 cm, carries it flat over the wound, lowers it to
  2.5 mm above the skin (0.5 mm above the wound insert), opens, rises and moves clear.

## Runtime and seeds

- **Run limit:** 1,200 control steps (1 minute at 20 Hz).
- **Demo:** the plan lasts 14.0 s. In Isaac Sim it succeeds at step 259 (13.0 s), about 100 s of wall time on one
  L40S including the simulator's start-up, and is deterministic (open loop from a fixed layout).
- **Dream-1:** ran the full 1,200 steps in about 4 minutes of wall time.
- **Seeds:** none. The layout is fixed (`object_jitter=0`), and the scripted demo is deterministic.

## Score a real trial

A real trial needs a ruler, the video and a top photo.

1. Mark the dressing's place on the pack tray and the tape roll's place. Film the trial from above (the rig's top
   camera sees the pack, the dressing and the wound). Take a top photo at the end.
2. Watch the video for the robot touching the dressing anywhere but within 1.25 cm of its edges, on either face.
   Halve the score for the first touch; score 0 for a second.
3. Score 0 if the dressing does not lie flat on the skin pad at the end, is tilted 5 degrees or more, is still
   within 2 cm of a gripper, or is still moving.
4. From the top photo: 1.0 if no part of the wound shows, 0.5 if part of it shows but its centre is covered, 0 if
   the centre shows.
5. Halve the score if the video shows the dressing slid more than 2 cm across the skin after it came to rest.

Judging a 2 mm touch from a video is less exact than the simulator's reading of the collision meshes. That is
accepted in exchange for a setup anyone can build from a drape, a silicone pad, a dressing pack and a dressing.
