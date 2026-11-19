# Wax-weld three blade patterns into a casting cluster with a hot knife, holding each square and still until the wax sets

An investment-casting wax room. Before a ceramic shell can be built, every injected wax pattern is welded by hand onto a wax tree (a cluster). Puffer and
Phipps (ACEEE 2003) describe the step: "a manual process using a hot knife blade passed between the two wax surfaces is used to attach the wax patterns to
the runners". They call it the most labour-intensive step of the process and say nearly all of it is done by hand. Aero-engine blade clusters carry 6 to 30 patterns
(Prasad, InTech 2012). A wax-welding patent (US 11,084,113) puts the tool's tip in the weld plane and then withdraws it so the melted wax flows. Patent
US 6,889,743 lists what goes wrong: pattern positions vary from one tree to the next, and patterns break off at the weld.

The scene: a sprue standing on its pour cup on an assembly stand already carries three blade patterns, one below the three empty marks and two on the next
face level with the lower two. Three loose blade patterns lie in a foam tray, and an electric wax knife lies in its rest.

The difficulty is in holding and timing, not in finding:

- **Two hands, one joint.** One hand holds the pattern by its platform with its gate end a few millimetres off the mark. The other slides the hot blade
  into that gap and lays it against the sprue, and the first hand brings the gate onto the blade. Heat flows only while the blade touches both faces at once.
- **In order.** The blade comes out, the gate goes onto the sprue at once and square, and the pattern is held still while the wax sets. If it is let go
  early it falls; if it is moved, the joint cracks and must be heated again. If the gate is pressed on first, the blade cannot get between the faces.
- **Not too long.** Held in the joint over 8 s, the gate end melts back.
- **A crowded cluster.** Each mark sits between patterns already welded. The knife may touch nothing but the gate end and the sprue, and nothing may knock
  a welded pattern. The geometry forces the order: a pattern welded higher up first blocks the holding hand's way to a lower mark.

## Files

| File | What it is |
|---|---|
| `task.py` | The scene, the scorer's hooks, `score()`, `success()` and `demo_script()` |
| `cluster.py` | Every size and pose, and the boxes that the weld rule and the colliders share (NumPy only) |
| `weld_rule.py` | The declared weld rule, the latches and the score (NumPy only) |
| `mechkit.py`, `demokit.py` | The props kit and the closed-loop demo kit from the accepted task `heat-shrink-lug`; additions listed at the top of `mechkit.py` |
| `test_weld_blade_waxes.py` | Offline tests: the rule on hand-built traces with the red-team cases, latches, geometry, reach and joint margins of the demo, gripper clearances, and the text against the code |
| `make_assets.py`, `assets/` | Printable pattern master, mould box and knife rest (STL, mm), with `assets/LICENSE.md` |
| `task.json` | The submission's fields |

## Requirements

- `yam-isaac-lab-arena` with Isaac Lab-Arena 0.3.1 (`source <ARENA_DIR>/yam.env`).
- Importing `mechkit` hooks `yam_arena.scene` and `yam_arena.inspect_embodiment` without modifying them.

## Run a trial

From this folder:

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --policy dream -P rows_per_request=20 --scorer task_score --max-steps 4800 \
    --instruction "$(python3 -c 'import json; print(json.load(open("task.json"))["instruction"])')"
```

The scene is the same for every seed.

## The demo

```bash
python demokit.py task.py OUT                                          # the full-score run, with video and setup.png
python demokit.py task.py OUT_checks --mode none,hover,early,order --fast   # failure checks (one: the first pattern only, for debugging)
PYTHONPATH=<yam-isaac-lab-arena> python3 test_weld_blade_waxes.py      # offline tests, no simulator
```

The demo is a scripted reference, not a policy. It reads the patterns' and the knife's poses from the simulator; all are in view. It never reads the
scorer. The left hand takes the knife by its handle and keeps it level, blade toward +x. For each mark, lowest first:

1. The right hand takes a loose pattern by the two ends of its platform and brings its gate end, square, 4 mm off the mark.
2. The left hand slides the blade into that gap from -x and lays it on the sprue's face. The right hand brings the gate onto the blade, and both hold for 3 s.
3. The blade slides out along the gap. The right hand presses the gate onto the sprue and holds still for 11 s, then lets go and backs away.

Last, the knife goes back in its rest and both hands move clear. A full run takes about 2,800 control steps (140 s); the run limit is 4,800.

Two details keep the demo honest on the real arm: every push is capped at 3 mm past where it needed to go (the arm leans, it does not wind up), and the
holding hand corrects its own slow sag while the wax sets. `WELD_DEBUG=1` prints where the blade and the gate are against the sprue at each stage.

## Scoring

Read at the end; the trial video is used for the at-any-time rules.

- **0 if at any time** a pattern already on the cluster is knocked 2 mm or 2 degrees out of place, a loose pattern's 30 mm platform is squeezed 1 mm or more
  narrower between a gripper's fingers, or the hot blade touches a gripper's finger. These are latched until the episode resets.
- **0.25 per loose pattern** that at the end hangs from the sprue by its own weld, with no finger within 1 cm of it, its gate centred within 3 mm of an
  empty mark (one pattern per mark) and square to the sprue's face within 3 degrees. **Minus 0.10** if its gate melted back.
- **Minus 0.10** for each pattern, loose or already welded, whose airfoil, platform or root the blade touched.
- Only once a pattern scores: **0.15** for the knife lying back in its rest (within 2 cm, its axis within 10 degrees; a roll about its round handle does not count), and **0.10** for every finger point 3 cm or more
  from the knife, every pattern and the sprue, held 1 s.
- `success()` is the full 1.0, held 1 s. Doing nothing scores 0.

**On a real trial:** a steel rule along the sprue for the mark; an engineer's square against the sprue's face for squareness; the gate's length by rule
for melt-back (42 mm instead of 45); photos for melt marks; video for knocks and for letting go.

## The weld rule (declared; `weld_rule.py`)

The simulator cannot melt wax, so melting and setting are bookkeeping on real contact and real stillness:

1. **Heat.** A joint gains heat only while points on the blade's faces are within 0.5 mm of the gate's end face and of the sprue at the same time, the
   blade covers the middle of the gate's face, and the gate is at a mark (centred within 3 mm, within 4 mm of the face). Each control step adds 0.05 s.
2. **Close.** Within 4 s of the blade leaving, the gate's end face (three of five points) must be on the sprue at the mark it was heated at, with 2 s of heat or more. Otherwise the
   joint is cold and must be heated again.
3. **Set.** In the first 2 s after the faces meet the wax is still liquid: the pattern may be set square there, and the set starts again. It must then stay
   within 1 mm and 2 degrees for 10 s. Leaving that band cracks the joint (cold again). Letting go needs
   no rule: each loose pattern hangs on six free joints (three slides, three hinges) from a hidden anchor, so it falls when released.
4. **Lock.** After the set, the six joints get a spring at their present positions (3,000 N/m and 20 N m/rad, with effort limits of 1.5 N and 0.1 N m):
   the pattern is welded. The three patterns already on the cluster are built the same way and start set, so a knock past those limits moves them.

Nothing in the rule moves, holds or releases a part. The times are estimates for blue injection wax at a 220 C knife (the knife's published range is
66 to 500 C). No published hold or heating time was found; calibrate them on a real trial.

Red-team cases, all in the offline tests, all scoring as intended:

- a blade hovering in the gap;
- a blade on the sprue only;
- a blade over the gate's edge only;
- heating 2.5 cm below a mark, then sliding onto it;
- heating at one mark and closing at another;
- one step of contact;
- the gate pressed on first;
- letting go at 4 s;
- moving 3 mm at 9 s;
- 1.5 s and 9 s of heat;
- closing 5 s after the blade left;
- a knocked neighbour put back;
- a crushing grip;
- the blade on a finger;
- a pattern still held at the end;
- two patterns on one mark;
- a weld 4 degrees out of square;
- two episodes back to back.

## Design notes

- **Why the platform.** The gripper's fingers widen to 68 mm 5 cm above their tips. A grip on the gate or root would reach into the sprue once the gate is on
  it, or land on the gate. A grip across the platform's two ends keeps the fingers and the gripper body 18 mm or more from the sprue, the knife and the
  cluster, and the knife hand's 26 mm or more (offline test).
- **Why the knife lies level.** A blade brought down from above puts the two gripper bodies one over the other. Slid in level from the side, as "passed
  between the two wax surfaces", the knife hand stays 4 cm or more from the holding hand.
- **Colliders** are boxes, including the airfoil's five twisted sections, so that millimetre faces are exact.
- **Episode memory** clears only on a real reset (`mechkit.on_reset`).
- **Release and rest** are read from finger points on both arms' links, not from a midpoint.

## The real-trial kit

| Item | Source | Price |
|---|---|---|
| Electric wax welder, 66 to 500 C, set to 220 C, with a flat blade tip (or a spatula tip ground flat to 0.8 mm) | Romanoff Vari-Temp 110 V (romanoff.com); alternative Renfert Waxlectric II | $283.50; $448.29 |
| Blue injection wax, 1 lb | Ferris blue injection wax (pmcsupplies.com) | $14.95 |
| Pattern master and mould box | print `assets/blade_pattern_master.stl` and `assets/mould_box.stl` | under $5 |
| RTV silicone for the mould, about 500 g | any two-part mould-making silicone | about $30 |
| Sprue and pour cup | the same wax, poured in a printed box | in the wax |
| Assembly stand | steel plate 120 x 120 x 8 mm with a 10 mm pin | under $20 |
| Knife rest and guard | print `assets/knife_rest.stl` (PETG or better) | under $5 |
| Drip tray; Kapton tape over the finger pads | aluminium sheet; hardware store | under $15 |
| Steel rule, engineer's square, camera | any | under $30 |

Each trial uses three patterns (about 51 g of wax, under $1.50), poured in batches ahead of time. The knife's cord runs off the table to its
controller; the simulation has no cord. A printed locating template checks the three patterns already on the cluster. The three new welds are cut off with the warm knife and the sprue's
face is scraped flat (about 2 minutes); the cluster with its three welded patterns is reused. The knife is hot (220 C). It lies in a guarded rest, the pads
are taped, and the operator resets with heat gloves after the knife has cooled in its rest.

## Runtime and seeds

`object_jitter=0`; the demo is deterministic. Measured on an L40S:

| Run | Steps | Simulated | Wall time | Score |
|---|---|---|---|---|
| Scripted demo, cameras at 640 x 360 | 2,794 | 140 s | 9.8 min | 1.0 |
| Scripted demo, `--fast` (CPU, no cameras) | 2,796 | 140 s | 2 min | 1.0 |
| Dream-1, 320 x 176 | 4,800 (limit) | 240 s | 27 min | 0.0 |

Failure checks (`--fast`): none 0.0; hover 0.0 (each joint goes cold); order 0.0; early 0.5 (the pattern let go at 4 s falls).

## Credits

`mechkit.py` and `demokit.py` come from the accepted task `heat-shrink-lug`, including `lock`, which the weld uses. The six-joint articulation follows its
`mechanism` and `Link(parent=...)`. The release check from robot link positions follows the accepted `lock-before-loading` task.
