# Embed a thin skin strip in paraffin: press it flat with forceps while the other hand slides the mould onto the cold spot

Paraffin embedding is the hand step between tissue processing and sectioning in histology labs. At an embedding station the technician lays each
processed piece cut face down in a base mould of molten paraffin, presses it flat with warm forceps, moves the mould onto the cold spot and keeps
pressing until the bottom layer of wax has set; the cassette and a top-up of wax follow. Thin specimens such as skin curl, so they are held "against the
bottom of a metal paraffin mold with the proper orientation until the paraffin has cooled" (US patent 8,329,120). A strip that sets curled is sectioned
at an angle: its sections miss part of the cut face, or the block is re-melted and embedded again.

The difficulty is a press that must last through a move:

- **One hand presses, the other moves.** The forceps' tips must stay on an 8 x 10 x 2 mm strip while the other hand slides the mould 10 cm across onto
  the cold block, so the tips follow a moving target at a millimetre scale.
- **Until it sets.** On the cold block the bottom sets round whatever shape the strip has: lifted early, the strip curls and sets curled.
- **Placed first.** The strip was dropped in near one end; it must be slid to the middle under the tips before the move.
- **Hands out of the wax.** The moving hand holds the mould by its outer walls only; a gripper in the wax ends the trial at 0.

The robot needs both arms. The scoring asks for the result only: any division of the work is allowed (one arm pushing the mould with the
forceps' tips still on the strip would count, if it kept the press). The strip starts cut face down; the robot's part is to centre it, keep it flat and
hold it through the move and the set, not to choose the cut face.

## Files

- `task.py`: the scene (station, mould, skin strip, forceps), the per-step trial and the scripted demo.
- `embedkit.py`: the declared rules (curling, setting) and the scoring ladder, pure Python; `write_pose` holds a curled or set strip.
- `mechkit.py`, `demokit.py`: from the accepted task `heat-shrink-lug`, unchanged. `mechkit` writes the props as USD text and hooks `yam_arena.scene`
  and `yam_arena.inspect_embodiment` as they load (nothing on disk changes); `demokit` runs closed-loop scripted demos.
- `test_embed_tissue_flat_in_paraffin.py`: offline checks with no simulator (below).

## Requirements

`yam-isaac-lab-arena` with Isaac Lab-Arena 0.3.1 (`source <ARENA_DIR>/yam.env`). The scene uses no library assets and no third-party meshes.

## Run a trial

From this folder:

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --policy dream --scorer task_score --max-steps 3600 --store-frames --log-dir logs \
    --instruction "$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')"
```

The scene is the same for every seed. Run at 20 Hz: the rules advance 0.05 s a control step.

## The demo

```bash
python demokit.py task.py OUT                                     # the full-score run, with video
python demokit.py task.py OUT_checks --fast --mode none,still,lift,nohold,dip   # the failure checks
PYTHONPATH=<yam-isaac-lab-arena> python3 test_embed_tissue_flat_in_paraffin.py   # offline: no simulator
```

The demo is a scripted reference, not a policy. It reads the poses of the mould, the strip and the forceps, all in view of the cameras, and never
the rules' state.

1. The left hand takes the forceps from the well and tilts them 30 degrees, tips towards the mould, so that the hand stays off to the mould's left side.
2. The tips come down onto the strip, 2 mm into its top (closed loop on where the tips are), and slide it to the middle of the mould, 2 mm at a time.
3. The right hand pinches the mould by its long walls at its near end and slides it 10 cm across onto the cold block; the left hand follows the
   mould's pose every step, keeping the press.
4. The right hand lets go and backs away; the left keeps pressing for 9.5 s, then lifts the forceps straight up and stands them back in the well.

Failure checks (`--mode`):

| mode | what the script does | score |
|---|---|---|
| `lift` | presses and centres the strip, then lifts the forceps for the move and does not press again | 0.0 (curled on the cold block) |
| `nohold` | as the demo, but lifts the forceps as soon as the mould arrives | 0.0 (curled before the set) |
| `still` | presses the strip but never moves the mould | 0.0 (never sets) |
| `dip` | lowers the right fingers into the wax | 0.0 |
| `none` | does nothing | 0.0 |

## Scoring

Measured once the block has cooled and been knocked out of the mould: a photo of the block face against a millimetre scale, and the trial video.

- **0 for the trial** if at any time a gripper or its sleeve touched the wax in the mould, the mould tipped more than 10 degrees or was lifted more than 1 cm, or the
  strip left the mould (latched until the next reset).
- **0 unless** the mould stood with its whole floor on the cold spot for 8 s with the strip in it, and the strip set flat: in the face photo its black inked outline is
  sharp all round, with no stretch longer than 1 mm greyed under wax (a curled end lies under the face and shows grey).
- **If flat:** 0.5; plus 0.25 if its centre is within 3 mm of the block face's centre (where its diagonals cross); plus 0.25 for the forceps back in their well and no gripper
  within 3 cm of the mould.
- **Full score (1.0):** all of these, still for 1 s; the trial then ends. Doing nothing scores 0. Possible scores: 0, 0.5, 0.75, 1.0.

In the simulator: "pressed" is the forceps' tip point on the strip's top face, inside its outline, over its middle third or curling end, from 1 mm
pushed in to 0.3 mm above (in contact); a curled strip sets curled as soon as any part of the mould's floor is over the cold block; "flat" is the inked
face's four corners within 1 mm of each other in height and its lowest corner within 1 mm of the floor when the bottom sets; "a gripper touched the
wax" is any of the 12 fingertip points per hand (`mechkit.fingertips`) inside the mould below its rim.

## Design notes

- **The curling rule.** Molten paraffin is not simulated and the strip is rigid, so a declared rule stands in for the curl: a strip the tips have not
  pressed for 0.3 s is held with one end 2 mm up, tilted 15 degrees (its inked face then spans 2.6 mm of depth). Pressed again while the wax is molten, it
  lies flat; moved while curled, it is left 4 mm behind the mould's motion once; curled while the mould stands on the cold block, the bottom sets round
  the curl and pressing no longer helps. The rule never moves the strip for the robot: a pressed strip lies on the floor under the tips by physics.
- **The set.** After 8 s on the cold block the strip is fixed in the mould where it lies (its pose rewritten each control step relative to the
  mould's, the technique of the accepted tasks `bayonet-mate` and `casement-stay-last-hole`).
- **Why the forceps tilt.** Each gripper is about 7 x 10 cm across its body; with the forceps upright the two hands would meet above the mould. Tilted
  30 degrees, the pressing hand sits 5 cm to the side, as a technician's hand does.
- **Weights.** The strip weighs 2 g in the simulator (a real one about 0.2 g) so that the contact solver handles it; the forceps 25 g, the mould with
  its wax 22 g.

## Real-trial kit

| Item | Source | Price |
|---|---|---|
| Embedding station with warm plate, cold spot and forceps well, or: a lab hot plate under a 6 mm aluminium plate at 60 C, an aluminium block on a fresh cold pack (about -5 C) beside it, flush, with a 1 cm PTFE strip between, and a 12 mm tube for the forceps | histology suppliers; any lab hot plate with a 60 C setting | not checked |
| Stainless steel base mould 24 x 37 x 9 mm | histology suppliers (a standard size) | not checked |
| Histology paraffin, 56 to 58 C melting point, poured 6 mm deep from a 60 C wax bath | histology suppliers | not checked |
| Embedding forceps, 150 mm, serrated tips | histology suppliers | not checked |
| Skin strips 8 x 10 x 2 mm, cut from one processed piece of pig skin, cut face (the face to be sectioned) inked with tissue-marking dye; dropped in cut face down near the far end, which must stand 2 to 4 mm off the floor (read against the 9 mm wall) or the strip is retired | prepared once by any histology lab, one batch per test series | reused: recovered by re-melting the block at 60 C |
| Heat-resistant silicone sleeves for the gripper fingertips | any | not checked |

A thermometer on the cold block reads -5 to 0 C at the start of each trial. Safety: the wax is at 60 C and the warm plate below 65 C. The grippers touch only the forceps' grip and the mould's outer walls, sleeved; wax that
reaches a sleeve sets in seconds and is peeled off.

## Runtime and results

- Run limit 3,600 control steps (180 s at 20 Hz). One fixed layout; `object_jitter=0`; no randomness.
- Offline (kinematic stand-in in the test file, arms reaching their commands exactly): the full demo reaches success at step 758 (38 s); the failure
  checks score as in the table above.
- Simulator results: to be filled in from the Brev runs.

## Credits

`mechkit.py` and `demokit.py` are from the accepted task `heat-shrink-lug`, unchanged. The pose write in `embedkit.write_pose` follows the accepted tasks
`bayonet-mate` and `casement-stay-last-hole`. All shapes are made in code.

## Sources

- US patent 8,329,120, "Microtome sectionable biopsy support for orienting tissue samples": thin samples held against the mould's bottom with forceps
  until the paraffin has cooled; skin biopsies curl.
- ARUP Laboratories, histology terms: "Embedding and orientation".
- NSCC Histotechnology (open textbook): "Embedding tissues in paraffin wax" and "Embedding orientation".
