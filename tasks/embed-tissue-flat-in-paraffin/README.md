# Embed a thin skin strip in paraffin: press it flat with forceps while the other hand slides the mould onto the cold spot

Paraffin embedding is the hand step between tissue processing and sectioning in histology labs. At an embedding station the technician lays each
processed piece cut face down in a base mould of molten paraffin, presses it flat with warm forceps, moves the mould onto the cold spot and keeps
pressing until the bottom layer of wax has set; the cassette and a top-up of wax follow. Thin specimens such as skin curl, so they are held "against the
bottom of a metal paraffin mold with the proper orientation until the paraffin has cooled" (US patent 8,329,120). A strip that sets curled is sectioned
at an angle: its sections miss part of the cut face, or the block is re-melted and embedded again.

The difficulty is a press that must last through a move:

- **One hand presses, the other moves.** The forceps' tips must stay on a 10 x 8 x 2 mm strip while the other hand slides the mould 10 cm across onto
  the cold block, so the tips follow a moving target at a millimetre scale.
- **Until it sets.** On the cold block the bottom sets round whatever shape the strip has: let go early, the strip bows up again and sets bowed.
- **Placed first.** The strip was dropped in near one end; it must be slid to the middle under the tips before the move.
- **Hands out of the wax.** The moving hand holds the mould by its outer walls only; a gripper in the wax ends the trial at 0.

The robot needs both arms. The scoring asks for the result only: any division of the work is allowed (one arm pushing the mould with the
forceps' tips still on the strip would count, if it kept the press). The strip starts cut face down; the robot's part is to centre it, keep it flat and
hold it through the move and the set, not to choose the cut face.

## Files

- `task.py`: the scene (station, mould, the skin strip as two halves on a sprung hinge, bent-tip forceps), the per-step trial and the scripted demo
  (with its press controller).
- `embedkit.py`: how the strip lies (read from its pose and its hinge), the declared setting rules and the scoring ladder, pure Python; `write_pose`
  holds a set strip in its block.
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
python demokit.py task.py OUT_checks --fast --mode none,hover,still,lift,late,nohold,dip   # the failure checks
PYTHONPATH=<yam-isaac-lab-arena> python3 test_embed_tissue_flat_in_paraffin.py   # offline: no simulator
```

The demo is a scripted reference, not a policy. It reads the poses of the mould, the strip and the forceps, all in view of the cameras, and never
the rules' state or the strip's hinge.

1. The left hand takes the bent-tip forceps from the well and tilts them 50 degrees, so that the bent tips stand upright and the hand stays 6 cm off
   to the mould's left side.
2. The right hand pinches the mould by its long walls at its near end and holds it still.
3. The tips come to 3 mm above the bowed strip's crown (closed loop on where the tips are) and go straight down at 4 mm/s to 3 mm below where the
   flat strip's top lies: they meet the crown, push it down the 2 mm to the floor against the spring, and the arm's own stiffness then presses. From
   then on a press controller runs every control step: if the tips drift more than 2.5 mm along the strip from its fold line, or to within 1 mm of
   its side edges, it shifts the hand level towards the crown; its height creeps down (at most 4 mm below the first press); while centring it moves
   the hand by a quarter of the strip's offset from the mould's middle, so the pressed strip goes with the tips.
4. The right hand slides the mould 10 cm across onto the cold block while the controller carries the left hand with the mould and keeps the
   strip in the mould's middle; it goes on centring for the first 2 s on the cold block, then both hands hold still for 7.5 s.
5. The right hand lets go and backs away; the left lifts the forceps straight up, brings their tips over the well (closed loop) and stands them back.

Failure checks (`--mode`):

| mode | what the script does | score |
|---|---|---|
| `hover` | everything the demo does, but the tips stop above the strip's crown and never touch it (in the check run they came within 0.1 mm) | 0.0 (the strip stays bowed and sets bowed) |
| `lift` | presses and centres the strip, then lifts the forceps for the move and does not press again | 0.0 (bowed on the cold block) |
| `late` | as `lift`, then presses the crown again for 9.5 s once the mould is on the cold block | 0.0 (the bow had set; the press no longer flattens it) |
| `nohold` | as the demo, but lifts the forceps as soon as the mould arrives | 0.0 (bowed before the set) |
| `still` | presses the strip flat but never moves the mould | 0.0 (never sets) |
| `dip` | lowers the right fingers into the wax | 0.0 |
| `none` | does nothing | 0.0 |

## Scoring

Measured once the block has cooled and been knocked out of the mould: a photo of the block face against a millimetre scale, and the trial video.

- **0 for the trial** if at any time a gripper or its sleeve touched the wax in the mould, the mould tipped more than 10 degrees or was lifted more than 1 cm, or the
  strip left the mould (latched until the next reset).
- **0 unless** the mould stood with its whole floor on the cold spot for 8 s with the strip in it, and the strip set flat: in the face photo its black inked outline is
  sharp all round, with no stretch longer than 1 mm greyed under wax (a part that set off the floor lies under the face and shows grey).
- **If flat:** 0.5; plus 0.25 if its centre is within 3 mm of the block face's centre (where its diagonals cross); plus 0.25 for the forceps back in their well and no gripper
  within 3 cm of the mould.
- **Full score (1.0):** all of these, still for 1 s; the trial then ends. Doing nothing scores 0. Possible scores: 0, 0.5, 0.75, 1.0.

A reference for "greyed", made once per batch: embed one spare strip with a 1 mm pin under one end. Its block is photographed in the same
frame and light as each trial's block; a stretch of outline as grey as the reference's raised end, or greyer, counts as greyed.

In the simulator: how the strip lies is read from its pose and its hinge angle, never from where the forceps are. "Flat" is the eight corners of
the inked face (four a half) within 1 mm of each other in height and the lowest within 1 mm of the floor at the moment the bottom sets; a bowed
strip sets bowed once it has stood not flat for 0.3 s while any part of the mould's floor is over the cold block; "a gripper touched the wax" is
any of the 12 fingertip points per hand (`mechkit.fingertips`) inside the mould below its rim (stricter than the wax itself by the 2 mm of
freeboard above it); "the forceps back in their well" is their closed tips inside the well's 20 mm tube, below its top.

## Design notes

- **The bow is physics.** Molten paraffin is not simulated, but the strip's bow is: the strip is two rigid 8 x 5 x 2 mm halves on a hinge along
  the middle of its top face, with a spring (6.0e-4 N m/rad) that rests with each half tilted 30 degrees, both ends down. Left alone it stands on
  its two ends with the middle of its inked face 1.9 mm off the floor. It goes down only as far as a contact pushes its crown: 0.16 N brings
  the inked face within the 1 mm that counts as flat (the crown 1.3 mm down), 0.25 N (25 gram-force) holds it fully down on the floor, and the
  hinge's damping (1.8e-4 N m s/rad) adds about 0.15 N while the crown is pushed down in half a second. The hinge stops at flat. The real strips
  are gauged to the same stiffness: 10 to 50 g on a pocket scale to hold one flat (below).
  No rule about the strip looks at where the forceps are (only the clean-up check does). In the check runs (CPU physics): with the tips hovering over the strip, 0.1 mm from its crown at the
  nearest, the hinge moved at most 1.4 of its 60 degrees and the strip set bowed; in the demo the hinge went through its 60 degrees as the tips
  came down on the crown (about 1 s) and the strip lay flat from then until the set; with the forceps lifted, the strip was no longer flat 0.2
  to 0.3 s later; pressed again for 9.5 s after its bow had set, its hinge did not move.
- **The set.** Two declared rules stand in for the wax. A strip that stands not flat for 0.3 s while any part of the mould's floor is over the
  cold block sets bowed: its hinge is locked where it is, and pushing no longer flattens it. The 0.3 s is a chosen value, stricter than a real
  cold spot, where the bottom film takes longer to gel: a run that passes here kept the press on through the arrival, which also passes a real
  trial. After 8 s on the cold block the strip is fixed in the
  mould as it lies (hinge locked, its pose rewritten each control step relative to the mould's, the technique of the accepted tasks `bayonet-mate`
  and `casement-stay-last-hole`).
- **Why a bow, pressed on its crown.** One closed pair of tips pushes at one point. A strip bowed up in its middle and pushed on its crown is
  bent flat between three points (the tips and its two ends on the floor); a strip curled up at one end and pushed there only rocks on the floor.
  The real strips are therefore prepared bowed (below), and the simulated strip has the same shape.
- **Why bent-tip forceps.** Each gripper is about 7 x 10 cm across its body; with straight forceps held upright the two hands meet above the mould,
  and straight forceps held tilted press at a slant and push the strip sideways. Embedding forceps with bent tips, held with the body tilted 50
  degrees, press straight down while the hand sits 6 cm to the side, as a technician's hand does.
- **The strip's friction.** In molten wax the strip sits in a film that holds it to the floor; the simulator gives it high friction (1.5) so that
  the tips' press holds it in place and it moves only when the tips carry it.
- **Weights.** The strip weighs 2 g in the simulator (a real one about 0.2 g) and each half's rotational inertia is about ten times its own, so
  that the contact solver handles a part this small; the forceps weigh 25 g, the mould with its wax 22 g.

## The skin strip: source, preparation and checks

The strip is real skin, prepared the way a histology lab prepares any skin slice, with one added step that gives every strip the same bow.
What is ordinary practice is cited; the bow's size and its spring-back are not in any published protocol, so every strip is gauged before every
trial instead (step 8).

1. **Source.** Raw, uncured pork back skin with its fat (rind on), from a butcher or an abattoir: the usual source of pig skin for skin research
   and suture training (Geisenberger et al. 2022 used butcher skin from back and belly; dermis about 1.9 mm thick on the back). One piece of
   5 x 5 cm gives a batch. It is kept at 0 to 4 C and sliced and put in fixative within 24 hours.
2. **Slices.** With a fresh scalpel or microtome blade, cut slices at right angles to the skin surface, 2.5 mm thick, each 11 mm long along the
   surface and 9 mm deep (epidermis, dermis and the fat under it). This is the bread-loaf slice of skin grossing (2 to 3 mm slices, embedded on
   the cut face). Cut 12.
3. **Ink.** Blot one cut face dry, paint it with black tissue marking dye, and let it dry 3 minutes (the suppliers' instructions: the dye bonds
   in 1 to 3 minutes on blotted fresh or fixed tissue and survives processing).
4. **Bow.** Lay each slice in a tissue cassette inked face down across a 3 mm stainless steel rod (a dowel pin 25 mm long), the rod under the
   slice's middle and across its 9 mm width; put a foam biopsy pad on top and close the lid, so that the pad holds both ends down over the rod.
   Fixed tissue keeps the shape it is fixed in, which is why labs normally flatten skin between pads; here the same effect sets the bow. The rod
   stays in from the first minute of fixation to the end of processing.
5. **Fix.** 10% neutral buffered formalin, at least 20 times the tissue's volume, 24 to 48 hours at room temperature.
6. **Process.** A routine overnight paraffin schedule on a tissue processor: 70%, 80%, 95% and three changes of 100% ethanol, three changes of
   xylene or a substitute, three changes of paraffin (melting point 56 to 58 C) at 60 C, 30 to 60 minutes a step. Formalin and xylene are
   handled by the lab under its fume hood; many histology core laboratories list paraffin processing without embedding as a per-cassette service.
   The strips come back in their cassettes, wax-infiltrated and not embedded. From here on they are inert and are handled with warm forceps only.
7. **Size.** Tissue shrinks in processing. After processing, with the strip solid at room temperature, keep those that measure 10 x 8 x 2 mm
   within 0.5 mm on each side (callipers).
8. **Gauge, before every trial.** Lay the strip inked face down in the mould's molten wax (60 C) for one minute. It is used only if (a) both
   ends touch the floor, a 1.5 mm pin passes under its middle and a 2.5 mm pin does not; (b) with the mould on a pocket scale (0.1 g, on a silicone
   pad, tared, for the few seconds of the reading), holding the strip flat on its crown with the forceps reads 10 to 50 g; (c) let go, its middle is back above 1 mm (the 1 mm pin passes)
   within 2 seconds; (d) it is not cracked. A strip that fails is retired and the next one of the batch is gauged.
9. **Between trials and sessions.** After a trial the block is re-melted in the 60 C wax bath (re-embedding a block is routine practice) and the
   strip is lifted out with warm forceps; with the batch in rotation the next trial starts at once with another gauged strip. Between sessions
   the strips are kept solid in paraffin at room temperature; they are held in molten wax only while trials run, since long hot holds make
   processed tissue brittle.
10. **The rig.** Only fixed, wax-filled strips come near the robot: they are inert and dry to the touch when cold. The grippers never touch the
    strip or the wax.

## Real-trial kit

| Item | Source | Price |
|---|---|---|
| Embedding station with warm plate, cold spot and forceps well, or: a lab hot plate under a 6 mm aluminium plate (16 x 16 cm) at 60 C; beside it an aluminium block 12 x 7 x 2 cm on a fresh cold pack (about -5 C), shimmed until a straightedge laid across shows no step; a 1 cm PTFE strip between them; and a 20 mm square steel tube, 45 mm tall, stood on the plate with high-temperature silicone, for the forceps | histology suppliers; any lab hot plate with a 60 C setting | not checked |
| Stainless steel base mould 24 x 37 x 9 mm | histology suppliers (a standard size) | not checked |
| Histology paraffin, 56 to 58 C melting point, poured 6 mm deep from a 60 C wax bath | histology suppliers | not checked |
| Embedding forceps, 150 mm, tips bent about 50 degrees, serrated | histology suppliers | not checked |
| Skin strips, 10 x 8 x 2 mm after processing, cut face inked black, bowed 1.5 to 2.5 mm: a batch of 12 prepared as in the section above | pork back skin from a butcher; black tissue marking dye, tissue cassettes, foam biopsy pads and paraffin processing from a histology lab or its suppliers; a 3 mm stainless dowel pin per cassette | reused: recovered by re-melting the block at 60 C |
| Pin gauges 1.0, 1.5 and 2.5 mm, callipers, a pocket scale reading 0.1 g, a shallow metal tray under the station | any tool supplier | not checked |
| Heat-resistant silicone sleeves for the gripper fingertips | any | not checked |

A thermometer on the cold block reads -5 to 0 C at the start of each trial. Safety: the wax is at 60 C and the warm plate below 65 C; the station
stands in a shallow metal tray, so the 5 ml of wax in a knocked mould sets on the tray and not on the table or the arms. The grippers touch only the forceps' grip and the mould's outer walls, sleeved; wax that
reaches a sleeve sets in seconds and is peeled off.

## Runtime and results

- Run limit 3,600 control steps (180 s at 20 Hz). One fixed layout; `object_jitter=0`; no randomness.
- Scripted demo, GPU physics with cameras (one L40S): 1.0, success at step 867 (43.4 s of simulated time), about 180 s of wall time.
- Failure checks, CPU physics (`--fast`): full 1.0 (success at step 872); hover, lift, late, nohold, still, dip and none 0.0.
- Dream-1: 0.0; no refusal; it ran all 3,600 steps (about 19 minutes of wall time) without taking up the forceps; the mould stayed on the warm plate.
- Offline (kinematic stand-in in the test file, arms reaching their commands exactly): the same modes score the same; the full demo succeeds at
  step 835.

## Credits

`mechkit.py` and `demokit.py` are from the accepted task `heat-shrink-lug` (MIT licence), unchanged. The pose write in `embedkit.write_pose` follows the accepted tasks
`bayonet-mate` and `casement-stay-last-hole`. All shapes are made in code.

## Sources

- US patent 8,329,120, "Microtome sectionable biopsy support for orienting tissue samples": thin samples held against the mould's bottom with forceps
  until the paraffin has cooled; formalin makes biopsy samples curl.
- ARUP Laboratories, histology terms: "Embedding and orientation".
- NSCC Histotechnology (open textbook): "Embedding tissues in paraffin wax", "Embedding orientation", "Grossing tissues" (slices of 3 to 4 mm at
  most) and "Operating the tissue processor" (an overnight schedule).
- National Cancer Institute, protocol 10334, appendix F, "Cutting and embedding skin": cuts at right angles to the surface at 2 to 3 mm, embedded
  cut surface down; curled skin is uncurled with forceps and straightened before the paraffin hardens.
- Leica Biosystems, "101 steps to better histology": slice thickness, fixative at 20 to 1, re-embedding.
- Davidson Marking System and StatLab tissue marking dye, instructions for use: blot, apply, dry 1 to 3 minutes; the dye withstands processing.
- Geisenberger et al., International Journal of Legal Medicine, 2022 (PMC9576652): fresh pig skin from a butcher, back and belly; dermis thickness.
- StainsFile, "Paraffin wax" and manual processing schedules: melting point 56 to 58 C; tissue can stand hours in molten wax, and long holds make it brittle.
