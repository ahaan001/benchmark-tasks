# Bolt a camera arm to a built T-slot frame with two drop-in T-nuts

A frame of 40 mm T-slot profile lies fastened to the table: a closed rectangle, so that the slots of its side bars are shut at both ends.
A printed camera bracket waits beside it: a foot with two M6 cap screws, a drop-in T-nut hanging lengthwise on each, an arm, and a
RealSense D405 on a side plate at the arm's end (the low, worm's-eye camera of a work cell). A T-handle hex driver stands in a block.

The robot lifts the bracket by its arm, lowers the foot onto the bar marked with a blue line so that both nuts go down through the
8.13 mm opening of the slot, out of sight under the foot, keeps the foot flat on the bar, and turns both screws with the driver. As a
screw starts to turn, its nut swings a quarter turn under the slot's lips and is then drawn up tight.

**Status:** offline tests pass (21 of 21). Simulator runs (the three mechanism tests, the scripted demo, Dream-1) are not yet done; this
line, the numbers under *Runtime* and the `scores` field are filled in from those runs.

## Why this is hard

- **Two nuts, 0.19 mm a side, unseen.** A 3931 nut is 7.75 mm wide and the opening 8.13 mm. Both nuts hang 60 mm apart under an opaque
  foot. The foot's edges show where the bar is to about a millimetre; the last of the alignment is by feel: a nut that misses lands on
  a lip and holds the foot 9.6 mm up.
- **The nut must hang at the right height.** Its flange (17.55 mm corner to corner) is smaller than the slot's cavity (20.57 mm), so the
  cavity never stops it. What stops it at a quarter turn is its boss, whose two square corners meet the walls of the opening. That only
  happens while the foot is flat on the bar: then the flange is 0.7 mm below the lips and 0.8 mm of the boss is still in the opening.
  Lift the foot and the flange itself is in the opening, cannot turn, and the screw goes tight with the nut run up against the foot.
- **The camera end overhangs.** The bracket's centre of mass is 109 mm from the middle of the foot, 89 mm past the bar's edge. Until a
  nut grips, a bracket that is let go tips over that edge and lifts its nuts into the opening. One hand has to hold the arm level
  while the other works the driver, and the two hands are 12 cm apart.
- **A hex key in a 5 mm socket.** The blade has to be over the screw within the socket's play and turned to the socket's hex before it
  goes in, and the wrist gives about half a turn a stroke, so the blade comes out and goes back in several times a screw.

## Files

| file | what it is |
| --- | --- |
| `task.py` | the yam_arena task file: the parts, the per-step thread rule, `success`, `score`, and the scripted demo |
| `tslot_geometry.py` | every size and place, with its source; the profile's section, the nut's outline, the fits (NumPy only) |
| `tslot_rule.py` | the thread rule and the score (NumPy only) |
| `mechkit.py` | parts from USD text: fixed props, free bodies, jointed mechanisms (see *Credits*) |
| `demokit.py` | closed-loop scripted demos on the harness's own embodiment (see *Credits*) |
| `dryrun.py` | a kinematic stand-in for the simulator: runs the real demo script and scorer, and checks the arms' collision hulls along the whole demo |
| `test_task.py` | 21 offline tests: fits, thread rule, score ladder and red-team cases, wiring, the demo's end states, reach and clearances |
| `drawing.py`, `camera_bracket.svg`, `camera_bracket.scad` | the bracket's drawing and printable model, written from the same sizes |
| `task.json` | submission metadata |

## Run and score a trial

Environment: yam_arena (Isaac Lab-Arena, Isaac Sim, PhysX at 240 Hz), 20 Hz control, object jitter off: one fixed layout, no seeds.

    INSTR=$(python3 -c "import json; print(json.load(open('task.json'))['instruction'])")
    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
        -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 3600 --store-frames --log-dir runs/dream

    python demokit.py task.py runs/demo --mode full                          # the scripted demo, recorded: demo.mp4, setup.png, result.json
    python demokit.py task.py runs/check --no-record --mode full,none,held,unheld,one_screw,off_line   # every end state in one session
    PYTHONPATH=<yam-isaac-lab-arena> python3 test_task.py                    # offline tests, no simulator (about 1.5 minutes)

`task_score` gives 1.0 when the episode ends in success, otherwise `score()` at the last step.

## Scoring

Read at the end of the run. Any score above 0 needs both grippers and the driver at least 1 cm clear of the bracket, and the bracket
still for 1 s (it has not moved 0.3 mm or turned 0.5 degrees).

A nut **grips** when it lies crosswise under both lips of the marked bar's slot (the middle of each end of its flange at least 1 mm under its lip:
turned 40 degrees or more), is drawn up against them, and its screw has turned at least a quarter turn past the point where the nut met the lips.

| score | end state |
| --- | --- |
| 1.0 | both nuts grip, the arm points in over the frame, the notch's point within 2 mm of the middle of the line |
| 0.75 | the same, with the notch 2 to 5 mm from the middle of the line |
| 0.5 | exactly one nut grips; or both grip with the notch more than 5 mm off or the arm pointing outward |
| 0.25 | no nut grips, but both nuts are in the marked bar's slot (flange tops at least 1 mm below the bar's face), foot flat or bracket tipped |
| 0 | anything else: a nut on top of the bar, the wrong bar, a hand still on the bracket, the driver still in a screw, nothing done |

Which screw is driven first, which hand does what and where the driver is put down are free.

**Real trial.** Ruler from the middle of the line to the notch. Then each nut is tested by the bracket's own overhang, which is
0.15 N m about the bar's inner edge: take the other screw right out. A nut that grips keeps the foot flat on the bar (a 0.5 mm feeler
goes under no corner); one that does not lets the camera end drop to the table. Put that screw back snug and test the other nut the
same way. Screws are left snug in this task; the final torque (6 N m for an M6 hammer nut in the maker item's data) is applied later
with a torque key and is not part of it.

What the scripted demo's other modes are expected to score (kinematic stand-in; to be confirmed in the simulator):

| mode | what it does | score |
| --- | --- | --- |
| `full` | the whole job | 1.0 |
| `none` | nothing | 0 |
| `held` | the whole job, but never lets go of the arm | 0 |
| `unheld` | lets go as soon as the foot is on the bar, then drives the screws | 0.25 or less |
| `one_screw` | tightens one screw only | 0.5 |
| `off_line` | mounts the bracket 4 mm from the line | 0.75 |

## How the parts are built

Everything is at real size. All geometry is written in `tslot_geometry.py` as coordinates and generated when the task loads; there are
no meshes, CAD files, textures or other third-party assets in this folder. The robot, table and cameras are yam_arena's own.

- **Frame** (fixed): four bars of 80/20 40-4040. Each bar's top slot is seven boxes: two lips (opening 8.13 mm, 4.32 mm thick), two
  upright walls (cavity 20.57 mm wide), two slabs for the cavity's 45 degree lower walls, and the floor 12.32 mm down. The length of the
  upright wall under each lip (2.7 mm) is scaled off the catalog drawing, which does not dimension it. The lips' 2 degree "drop lock"
  slope and the bars' other three slots are not modelled. Four 80/20 40-4302 inside corner brackets are plain boxes.
- **Bracket** (one free body of 168 g with the camera and hardware, friction 0.4): foot 80 x 40 x 6.4 mm, arm 20 x 12 mm to 200 mm from
  the foot's centre, side plate, D405 as a 42 x 42 x 23 mm box of 60 g. The 6.4 mm foot is chosen so that an M6 x 16 screw's end is
  flush with the nut's underside exactly when the nut hangs at the right height: that is how the nuts are set for a trial.
- **Screws** (two hinges in the foot): ISO 4762 heads, 10 mm across and 6 mm high. Each head is a floor and six slabs round a hex socket
  5.1 mm across flats and 3.5 mm deep, so the only thing that turns a screw is a hex blade in its socket. The foot's holes are printed
  6.0 mm, a light push fit on the screws: a screw stays where it is put and turns against 20 mN m (estimated).
- **Nuts** (each a slide along its screw and a hinge on that slide): 80/20 3931, flange 15.75 x 7.75 x 4.57 mm with bevelled ends, boss
  7.82 mm long and 1.5 mm high, both as convex hulls. Two diagonally opposite corners of flange and boss are cut back, as on the part
  (radius 3.6 mm on the boss and 3.0 mm on the flange, estimated from the catalog drawing, which gives no radii): lengthwise in the
  opening the boss can turn 92.7 degrees clockwise and 5.7 degrees the other way before its square corners meet the lips. A nut's play
  on its thread, about 0.15 mm either way at the flange (estimated from the 6H/6g fit), is not a joint here: the colliders are 0.3 mm
  narrower than the part instead, 7.45 mm. The ridges on the flange's top are not modelled; steel and PETG on anodised aluminium have friction 0.4.
- **Driver** (free body, 79 g): Wera 454 HF, 5 x 100 mm. A 4.98 mm hex blade as a convex hull, a neck 30 x 22 x 21 mm that a gripper
  takes by its two flat sides, the 99 mm bar and the 20 mm short arm. The split of the 49 mm handle into neck and bar is ours. The
  blade's tip is chamfered 1 mm over its last 1.5 mm, which leads it into a socket from 1 mm off its axis; a real tip's chamfer is
  about 0.3 mm. The holding ball in the tip is not modelled.
- **Block and supports** (fixed): a beech block 60 x 60 x 40 mm with a 7 mm hole 35 mm deep (square in the model), three EVA foam blocks
  35 mm high.

### The thread rule (the one rule in the task)

PhysX cannot cut an M6 thread. Each nut therefore rides its screw on a rule, run every control step in `task.py` from `tslot_rule.py`:

1. A hinge with 10 mN m of friction joins the nut to its screw, so a free nut goes round with the screw, and a nut that rubs on a lip
   on its way into the slot stays lengthwise. (On the real parts, one turn of thread-seal tape on the screw gives that drag.)
2. Whatever the nut turns on its screw moves it along the screw: 1 mm a turn, through a spring of 40 N/mm. A nut stopped by the slot's
   lips is pulled against them harder with every further turn, and that pull clamps the foot to the bar.
3. The torque the screw takes rises with that pull: 10 mN m + 0.2 x 6 mm x clamp load. Half a turn past contact is 20 N and 34 mN m,
   on top of the 20 mN m of the screw in its hole.

Nothing in the rule knows where the slot is. Whether a nut passes the opening, how far it swings, what stops it and what it is drawn up
against are all contact between the nut's hulls and the bar's boxes. Run up lengthwise, a nut jams against the foot and its screw goes
tight on a loose bracket; hanging too low, it would spin with its screw; on top of the bar, it spins in the air.

### Mechanism tests (200 control steps or fewer each)

| test | start | what it shows |
| --- | --- | --- |
| `TSLOT_START=mounted python demokit.py task.py OUT --fast --mode drive` | bracket on the bar, nuts in the slot, arm propped level | the hex blade goes into a socket and turns the screw; the nut swings with it, stops at a quarter turn on the lips of the opening, and starts up the screw |
| `TSLOT_START=askew python demokit.py task.py OUT --fast --mode settle` | bracket let go 3 mm above the bar, one nut turned 20 degrees | a nut that is off the slot's line lands on the lips and stays out |
| `python demokit.py task.py OUT --fast --mode take` | as the setup says | the grip on the arm, the lift, and how far the arm gives under 168 g |

## The scripted demo

Closed loop on what a policy could observe: the bracket's pose, the screws' angles, the driver's pose, the arms' own joints.

The left hand takes the arm 118 mm from the foot's centre, close to the centre of mass, leaning 30 degrees back along the arm so that
its wrist stands clear of the other hand later. It lifts the bracket off its blocks, carries it a quarter turn round to the marked bar,
squares the foot to the bar and brings its middle over the slot, then lowers it. If the foot stays up, the nuts are on the lips: the hand
lifts a little, steps a fraction of a millimetre across the slot and lowers again, further each time, until the foot is down. It slides
the notch onto the line and holds. The right hand takes the driver by its neck, lifts it out of the block, turns its hex to the socket's,
sets the blade in and turns clockwise through as much of the wrist's window as there is (133 to 193 degrees), lifts out, turns back in
the air and goes in again, until the thread is wound half a turn past where the nut met the lips: three strokes a screw. It stands the
driver back in its block; the left hand lets go last and both arms move clear.

Offline, against the stand-in: success at step 1,614 of 3,600; IK errors under 0.2 mm; no arm link within 4 mm of anything it is not
holding, and the two arms never within 12 mm of each other. Those figures are for the stand-in, not the simulator.

## Real-trial kit

| item | part | source and price (read Nov 2026) |
| --- | --- | --- |
| profile | 80/20 40-4040, about 1.2 m cut into 2 x 300 mm and 2 x 280 mm | about $0.04 a millimetre at tecotechnology.com (80/20 distributor): about $50 |
| corners | 4 x 80/20 40-4302 inside corner bracket, with M8 screws and T-nuts | $5.20 each |
| nuts and screws | 2 x 80/20 3931 drop-in T-nut and 2 x M6 x 16 socket head cap screw (together: 80/20 bolt assembly 75-3595) | nut $1.27 each |
| bracket | printed from `camera_bracket.scad` in PETG, solid | 1 kg of PETG about $20; the bracket takes about 90 g |
| camera | Intel RealSense D405, or a block 42 x 42 x 23 mm weighing 60 g | $325, or nothing |
| driver | Wera 454 HF T-handle hex key 5 x 100 mm, code 05023342001 | about $13 |
| the rest | a beech block with a 7 mm hole, three EVA foam blocks 35 mm high, a blue paint pen, thread-seal tape, a 5 cm offcut of the profile to set the nuts on | a few dollars |

About $120 without the camera. Nothing is consumed in a trial. Reset takes about two minutes (see `reset_notes`).

## Where the sizes come from

- 80/20 40-4040: 80/20 catalog, page "40mm X 40mm T-Slotted Profile - Four Open T-Slots" (8.13, 4.32, 20.57, 12.32, 8.00 mm; 2.364 kg/m).
  The catalog's own numbers differ by 0.26 mm on the slot's depth (12.3 to 12.6 mm); 12.32 is used.
- 80/20 3931: 80/20 catalog p. 3-172, "Standard Drop-In T-Nut" (A .620, B .305, C .308, E .059, F .180, G .239 in; .006 lb; "Loads from
  profile end or side", "Ideal for captivated profiles"), and 80/20's three install drawings for the part (lowered in lengthwise,
  turned, crosswise under the lips). Bolt assembly 75-3595: catalog p. 3-194 on.
- M6 x 16 socket head cap screw: ISO 4762 (head 10 x 6 mm, 5 mm hex, socket at least 3 mm deep).
- Wera 454 HF 5 x 100: Wera's size table (blade 100, handle 49 and 99, short arm 20 mm) and specification sheet (79 g).
- Intel RealSense D405: Intel D400 series datasheet (42 x 42 x 23 mm, 60 g, two M3 back threads, one 1/4-20).
- How such nuts behave: item Industrietechnik, Hammerhead Nut 8 St M6 ("When the screw is tightened, the Hammerhead Nut rotates around
  90° and is clamped in the groove"; 6 N m); FarmBot Genesis assembly guide ("the tee nuts must rotate 90 degrees when you begin
  tightening the screw", "If the nut doesn't grip the slot the first time, simply loosen the screw and try again").
- Cameras on T-slot frames: ALOHA 2 (arXiv 2405.02292, sections 2.4 and 2.5); Trossen Robotics, Stationary AI assembly guide (step 7,
  the worm's-eye camera mount on the front bar); Zivid, stationary mounting ("M6 screws and T-nuts").

## Runtime

Not yet measured in the simulator. Offline: the demo needs about 1,650 control steps (83 s of simulated time) of the 3,600 allowed.

## Credits

- `mechkit.py` is the parts kit of the accepted task `heat-shrink-lug`, with the additions listed at the top of the file; its
  `set_friction` uses the call that the kit of the accepted task `plug-a-night-light-through-a-sliding-outlet-cover` uses.
- `demokit.py` is the demo kit shared by `heat-shrink-lug` and ten other accepted tasks, unchanged.
- `write_pose` in `task.py` follows `partkit.write_pose` of the accepted task `bayonet-mate`.
- The clamp-torque law (0.2 x diameter x load) is the usual nut-factor rule of thumb.
