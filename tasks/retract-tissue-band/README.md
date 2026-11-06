# Retract a tissue band to reach the fragments under it without pulling it off

A pink tissue band lies over a wound channel and runs under a yellow ligament bridge, with 15 cm of band showing
beyond the bridge. Two metal fragments lie under that far part of the band. The robot is asked to put them in a
steel tray and lay the band back. The fragments can only be reached by sliding the band back toward the robot:
the first appears after 7 cm, the second after 13.5 cm, and after 17 cm the band comes out from under the bridge,
which scores 0. The second fragment is worth more (0.6 against 0.4). Coming within 1.5 cm of the limit halves the
score, so does leaving the band retracted at the end, and doing nothing scores 0.

## Files

- `retract_tissue_band.py`: the scene, the scoring hooks and a scripted demo (both fragments, band laid back,
  score 1.0).
- `task.json`: the task's details.
- `check.py`: offline check of the layout, the camera view, the scoring and the demo.
- `trace.py`: the same task with the demo's tracking error, the band's ends and slack, the tracker's state and each
  fragment's share printed every 5 steps, for tuning the demo.
- `zero_action.py`: the same task with an empty plan, which records a zero-action video to stand in for the
  Dream-1 run if the Dream API refuses the instruction and sends no actions.

Environment: Isaac Lab Arena 0.3.1, Isaac Sim 6.1, inspect-robots 0.60.0, yam-isaac-lab-arena at 0059d02. The
scene uses boxes only, no library models and no third-party assets.

## Run a trial

Run from the repository root.

```bash
TASK=tasks/retract-tissue-band/retract_tissue_band.py
INSTR="Put the two metal fragments that lie under the pink tissue band into the steel tray, then lay the band back where it was."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dream/retract-tissue-band

# Scripted demo (both fragments, band laid back)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$TASK -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/demo/retract-tissue-band
```

The run prints `task_score`. Render the camera videos with `inspect-robots video <log>.json`. Add
`-E headless=false` to watch a run in the Isaac Sim window. A zero-action video is the scripted command with
`-E task=tasks/retract-tissue-band/zero_action.py -P hold_s=120`.

Offline check, without the simulator (a few minutes):

```bash
python tasks/retract-tissue-band/check.py
```

It measures the room the gripper has at the band's near end and at each uncovered fragment, how far the band can be
lifted before the bridge stops it, and what the top camera sees past the arms; scores 32 hand-built trials; plays
the demo through the IK at 20 Hz with the band and each fragment following the gripper that holds them; and plays
the naive variant of the demo, which slides the band 18 cm.

## Scoring

The score is read after every control step from the poses of the band and the fragments and from the arms'
measured joints.

Score 0, for the rest of the trial, once the band comes out from under the bridge: the centre of its far end face
passes the bridge's near face (a slide of 17 cm or more toward the robot), or the centre of its near end face passes
the far face (pushed away). Pushing the band back under the bridge changes nothing.

Halved, for the rest of the trial, once the band's far end comes within 1.5 cm of the bridge's near face without
coming out (a slide of 15.5 cm or more).

Otherwise each fragment earns a share of its value (0.4 for the one 40 cm out, uncovered first; 0.6 for the one
33.5 cm out, uncovered last):

- **nothing** unless it was lifted off the pad at some time (underside 1.5 cm or more above it) without first
  being slid more than 3 cm from where it stood, is let go (every part of both grippers 2 cm or more from it; the
  grippers are placed from their measured joints and the URDF's collision meshes, checked every 2 mm) and is at
  rest,
- **its full value** if it lies on the tray's floor: centre inside the walls, underside within 5 mm of the floor,
- **half its value** anywhere else with its centre outside the strips' outer faces and not off the table: on the
  pad beside the channel, on the table, or in the tray but resting on the other fragment,
- **nothing** in the channel, on a strip, on the band, or off the table.

The sum is halved if at the end the band is not laid back: lying on both strips (underside within 5 mm of their
tops), its near end within 2 cm of its mark, its centre within 1 cm of the channel's axis, turned less than 10
degrees from it, let go and at rest.

`success()` is the 1.0 state: both fragments on the tray's floor and the band laid back, all let go and at rest,
with the band never out nor within 1.5 cm of coming out. Held for 1 s (`SUCCESS_HOLD_S`) it ends the episode.
Otherwise `task_score` is `score()` on the last step. Possible scores: multiples of 0.025 from 0 to 1.

## Design notes

- **Reward and risk rise together.** The first fragment costs 7 cm of the band's 17 cm of slack; the second costs
  13.5 cm, 2 cm before the halving and 3.5 cm before the band comes out. The 4 cm from the band's far end to a
  fragment's centre is what the fingers need: 1 cm less and they come down 0.3 mm from the band (`check.py`).
- **The easy moves meet the hazard.** Lifting the band, the usual way to get a cover out of the way, is stopped by
  the bridge after 4.4 mm at the start; pulling it until it moves freely takes it out from under the bridge. The
  naive variant of the demo slides the band 18 cm and scores 0.
- **The numbers make the trade.** Stopping after the first fragment and laying the band back earns 0.4. The second
  adds 0.6, but overshooting its 13.5 cm by 2 cm halves everything to 0.5, and by 3.5 cm scores 0. Leaving the
  band retracted at the end halves the score, because a field is closed when the case is over.
- **Only the slack is visible.** The fragments are hidden until the band uncovers them. The band left beyond the
  bridge is in the top camera throughout, 1.6 pixels per cm at 320 x 176, so the policy can see how much it has
  spent.
- **Nothing can be undone.** Coming out and the 1.5 cm warning are kept until the episode's next reset, which the
  scorer reads from the embodiment's episode clock going back to 0. Sliding the band back under the bridge or
  putting the fragments back clears nothing (`check.py` tests this, and two episodes back to back).
- **Placing includes letting go.** A fragment within 2 cm of a gripper, or still moving, earns nothing; the band
  counts as laid back only once every part of both grippers is 2 cm from it.
- **Taking out means lifting.** A fragment swept along or out of the channel without being lifted earns nothing,
  even if it is picked up later.
- **No order is scored.** Either fragment may come first.
- **The grasp has to fit the channel.** With the fingers closing across the 6 cm channel they pass 2.4 mm from a
  strip at its top; closing along the channel they pass 13.4 mm from the strips and 9.1 mm from the retracted band.
  The band is pinched across its 8 cm width with the fingers fully open (9.3 cm between the pads), 6 mm above the
  strips, 7.4 mm from the band on each side on the way down.
- **The demo is open loop** from the known start layout and reads nothing from the simulator while it runs. It
  slides the band 13.5 cm, takes the fragments with the fingers closing along the channel, turns the grip a quarter
  turn on the way to the tray because the wrist cannot hold that grip over the tray, sets the fragments 6 cm apart on
  the tray's floor, and slides the band back to its mark.

## Runtime and seeds

- **Run limit:** 2,400 control steps (2 minutes at 20 Hz).
- **Demo:** the plan lasts 50.8 s. TO FILL after the simulator run: steps to success and wall time on one L40S.
- **Seeds:** none. The layout is fixed (`object_jitter=0`), and the scripted demo is deterministic.

## Score a real trial

A real trial needs no instruments. It is scored by eye and with a ruler.

1. Mark the band's near end on the strips and each fragment's place on the pad. Film the trial with a camera that
   sees the bridge and the band beyond it (the rig's top camera does). Take a top photo at the end.
2. Score 0 if the video shows the band coming out from under the bridge at any time: its far end passing the
   bridge's near face, or its near end passing the far face.
3. Halve the score if the band's far end was at any time within 1.5 cm of the bridge's near face.
4. For each fragment, give nothing if the video shows it was slid more than 3 cm before it was lifted, or never
   lifted 1.5 cm, or if a gripper is within 2 cm of it at the end. Otherwise give its value (0.4 for the one that
   stood 40 cm out, 0.6 for the one 33.5 cm out) if it lies on the tray's floor, half its value anywhere else
   outside the strips' outer faces, nothing in the channel, on a strip, on the band or off the table.
5. Add the values. Halve the sum if the band does not lie flat on both strips with its near end within 2 cm of its
   mark, centred within 1 cm, turned less than 10 degrees, with both grippers 2 cm or more away.

Judging the band's position from a video is less exact than the simulator's reading of its pose. That is accepted
in exchange for a setup anyone can build from a silicone pad, four foam strips and bars, two wooden posts, two
metal cubes and a tray.
