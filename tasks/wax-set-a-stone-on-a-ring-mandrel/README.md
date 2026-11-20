# Wax-set a stone in a ring on a hand-held mandrel: tweezer it in, press it home, warm and close each wax claw

Wax setting, also called stone-in-place casting: in a jewellery factory the stones are set by hand into each wax pattern before it is cast, and the metal is
cast round them. This task is one claw-set stone. A turquoise wax ring sits on a wooden mandrel. One hand holds the mandrel. The other lays an 8 mm red stone
between the ring's four wax claws with tweezers, presses it home with a pusher, then closes each claw with a warm wax pen: the tip rests on the claw until the
wax softens, the claw is pushed over the stone's edge, and the tip lifts off.

What makes it hard:

- **Three tools, one after another,** each taken from its place and put back: tweezers (a 0.63 g stone carried by its girdle), a pusher, a warm pen.
- **The wax decides the timing.** A claw pushed cold springs back, and breaks if it is forced. One kept under the pen too long melts onto the stone. The
  claw gives only after the tip has rested on it for a moment, and it sets only once it has cooled.
- **The pen touches the claws and nothing else.** Its tip on the ring's shank, on the head or on the stone leaves a melt mark.
- **Order matters.** A claw closed before the stone is pressed home closes over a stone that still stands proud: it does not hold it.
- **The other hand has a job.** The ring overhangs the V-block the mandrel lies in, so that both hands can reach it. Pressing on the ring tips a mandrel
  that nobody holds, and a mandrel that tips over scores 0.
- **Nothing but tools touches the work.** A gripper on the stone or on the wax scores 0.

**The arms.** In the demo the left arm holds the mandrel and the right arm works the tools. The scoring asks for the result only: either arm may do either
job.

## Requirements

- `yam-isaac-lab-arena` with Isaac Lab-Arena 0.3.1 (`source <ARENA_DIR>/yam.env`).
- `mechkit.py` and `demokit.py`, in this folder, from the accepted task `heat-shrink-lug`. Importing `mechkit` hooks `yam_arena.scene` and
  `yam_arena.inspect_embodiment` without modifying them. This copy of `mechkit.py` adds five things, listed at its top: a hinge about any direction, a
  turned box and a round rod as exact colliders, a centre of mass for a free mechanism's frame, and a joint whose spring can be changed while it runs.

## Run a trial

From this folder:

```bash
I="$(python3 -c "import json; print(json.load(open('task.json'))['instruction'])")"
inspect-robots run --embodiment yam_arena -E task=wax_set_a_stone_on_a_ring_mandrel.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --policy dream -P rows_per_request=20 --scorer task_score --max-steps 4200 --instruction "$I"
```

`--scorer task_score` prints the score. The scene is the same for every seed: nothing is randomised. A trial is at most 4,200 control steps, 3.5 minutes of
simulated time.

## The demo

From this folder:

```bash
python demokit.py wax_set_a_stone_on_a_ring_mandrel.py OUT                                                  # the full-score run, with video
python demokit.py wax_set_a_stone_on_a_ring_mandrel.py OUT_checks --fast --mode none,hover,cold,long,early,unheld,shank   # the failure checks
python test_wax_set_a_stone_on_a_ring_mandrel.py                                                            # the offline checks, no simulator
```

The demo is a scripted reference, not a policy. It reads the poses of the mandrel, the stone and the tools and the angle of each claw from the simulator,
all of which the cameras show, and its own arm's joints. It never reads the scorer or the wax's state. It does not sense that a claw has softened: it rests
the tip for 2.4 s, the wax rule's 1.5 s and a margin, as a setter counts.

1. The left hand takes the mandrel by its thick end, fingers either side, and holds it. The gripper leans 30 degrees back along the stick, so that its
   wrist stands clear of the hand that works at the ring.
2. The right hand works the tools. They stand in a row; it comes level with each from the ring's side of the row, 65 mm off, slides onto it, and leaves the
   same way, so it never passes one tool to reach another. It takes the tweezers by their legs, 105 mm above the tips, and squeezes them until the tips
   stand 13 mm apart. It brings the tips over the stone on its block and checks where the stone lies between them; if that is more than 1.5 mm off across
   the tips or 0.6 mm off along them, it goes up and comes down again. It sets the tips down either side of the stone, closes them on the girdle, carries
   the stone over the ring, lowers it between the claws and opens the tips. The tweezers go back in their trough.
3. It takes the pusher by its knob, sets the flat face on the stone's table and presses until the stone stops going down. The pusher goes back in its hole.
4. It takes the pen. For each claw, opposite pairs in turn: the tip comes 7 mm outside the claw and 9 mm above, down level with its place on the claw's
   outside face, and in to 2.5 mm off it (again, if it has not come to rest within 0.8 mm). It closes onto the face at 2 mm/s until it touches, and rests
   there for 2.4 s without leaning. Then it takes the claw round at 60 degrees a second, aiming 0.3 mm into the face where the claw will stand next, until
   the claw is 4 degrees short of lying on the stone, and lifts away. No touch lasts more than 4.2 s. A claw left short of the stone's edge is warmed and
   pushed again, up to three times.
5. The pen goes back in its stand (its handle's end, not the tip that hangs below it, is what is centred on the bore), the left hand lets go, and both
   hands move away.

Every fine placement is closed on what the hand holds, read from its pose: the simulated arm's joints stick until their command is 2 mm or so ahead, so the
tip of a tool lands within about a millimetre of where it is sent, and the script checks and comes again where that is not enough. When a tip arrives the
arm is sent where it stands, which stops it creeping on.

Results in the simulator. The recorded run (the first command above: GPU physics, three cameras) scores 1.0 and ends on success after 2,863 control steps,
143 s of simulated time, of the 4,200 allowed. With `--fast` (CPU physics, no cameras) the same script scores 1.0 in 2,949 steps. The two differ because the
arm's sticking joints make each fine placement land a little differently, and the script comes again where one lands too far off: in the `--fast` run the
tweezers' tips came down 1.5 mm off the stone the first time and were lifted and set down again. The recorded run takes about 17 minutes of wall time on one
data-centre GPU (about 3 steps a second); `--fast` runs at about 19 steps a second. Dream-1, run once with the command under "Run a trial", scored 0.0
over its 4,200 steps and did not refuse: one gripper took the mandrel by its thick end and lifted it off its rests,
the other pulled the tweezers out of their trough; the stone was never picked up, and the mandrel, the tweezers and the pen ended lying on the table.

Failure checks (`--mode`), with the score each reached in the simulator (the second command above):

| mode | what the script does | what happens | score |
|---|---|---|---|
| `none` | nothing | nothing | 0 |
| `hover` | everything, but the pen's tip stays 1.5 mm off each claw for 2.5 s | hovering warms nothing: the stone is pressed home and no claw closes | 0.2 |
| `cold` | pushes each claw at once, without waiting | each cold claw bends and springs back: none closes | 0.2 |
| `long` | keeps the pen on the first claw for 6 s | that claw melts: 0, though the other three are then closed and everything is put back | 0 |
| `early` | closes the claws before the stone is pressed home | the first claw tips the stone, which still stands proud on its seat, and it ends askew between the claws: not in the head, and the pusher cannot seat it | 0 |
| `unheld` | the left hand stays away | the press tips the mandrel over its V-block until its thin end is on the table: 0 from then on, though the stone goes home and four claws are closed | 0 |
| `shank` | lays the pen's tip on the ring's shank for 1.5 s first | a melt mark: 0.5 with everything else done | 0.5 |

`early` was designed to end at 0.2 (the stone pressed home afterwards, the early claws not counted), and the offline tests check that arithmetic. In the
simulator the stone does not survive the first claw, which is the plainer lesson about the order.

Three more modes check nothing and are there for whoever tunes the demo on another arm: `probe` measures how exactly the hand places the pen's tip,
`claws` runs the pen's part alone on an empty head and prints each push step by step (the only place the wax's heat is printed; nothing steers by it),
and `tools` takes each tool out of its place and puts it back, twice. The log also notes whenever a tool standing in its place is knocked.

## Scoring

`score()` and `success()` are in `wax_set_a_stone_on_a_ring_mandrel.py`; the arithmetic is `wax_rule.py`, which needs no simulator. `SUCCESS_HOLD_S = 1`:
success is the full score held for 1 s.

**0, for the rest of the trial,** if at any time:

- a claw melted (the pen's tip on it for 5 s);
- a claw broke (forced 25 degrees while cold);
- a gripper finger touched the wax ring or the stone (came within 2 mm);
- the pen was out of a gripper and out of its stand for 2 s (laid down or dropped);
- the mandrel, with no gripper holding it, tipped over its V-block until its thin end touched the table;
- the stone, a tool or the mandrel left the table.

**At most 0.5** if the pen left a melt mark (its tip on the shank, the head or the stone for 0.5 s).

Otherwise, read at the end of the trial:

- **0.1** if the stone lies in the head, table up (within 1.2 mm of the head's axis and 30 degrees of level).
- **0.1 more** if it is pressed home and level: its table less than 2 mm above the head's rim, within 10 degrees. Unpressed it stands 2.9 mm above the rim,
  pressed 1.4 mm.
- **0.15 for each claw** whose tip lies 1 mm or more over the stone's edge, set (not soft), and closed after the stone was pressed home.
- **0.2 more** when three or more claws are so closed, the pen is in its stand, the tweezers and pusher are in their block, the mandrel lies on its rests,
  every finger is 5 cm or more from all of them and from the stone (surface to surface), and everything is at rest: nothing moves in the last second.
  The simulator's measure of that is that none of the mandrel, the stone and the tools has moved 0.3 mm or turned half a degree in that second. It
  reads rest from where the parts have been, not from their speeds: a stone held under four claws can show a speed there and go nowhere.

Possible scores: 0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.85 and 1.0. Nothing is cleared when parts return to where they started; only a real reset of the
episode clears the latches.

**Real trial.** Score from the trial video and two photos of the ring's head beside a millimetre scale, one from above and one from the side, with a loupe
to hand.

- Melted: a claw's tip has lost its shape and wax lies on the stone 1 mm or more beyond the claw (top photo). Broken: a claw is off, or cracked at its root.
- Melt mark: a pit or glazed patch 1 mm or more across in the wax away from the claws, or wax on the stone's table.
- In the head: the stone sits between the claws, table up. Pressed home: in the side photo its table stands less than 2 mm above the head's rim, and level
  within 10 degrees.
- Closed: in the top photo the claw's tip lies 1 mm or more over the stone's edge. Closed after the press: from the video.
- Put back and clear: from the video's last frames and a ruler: 5 cm between each gripper and the mandrel, the ring, the stone and every tool.
- Touching, laying the pen down, the mandrel tipping and leaving the table: from the video.

## Real trials: the build

Positions in cm from the midpoint between the arm bases (x forward, y left); heights above the table.

| Item | What | Where |
|---|---|---|
| Mandrel | Wooden ring stick, 295 mm, 23 mm tapering to 12 mm, 57 g, loose | Along y, axis 25 mm up, thick end at (24, 3.3) |
| V-block | Steel, 35 x 10 x 30 mm, 90 degree V, fixed | (24, -9): centred 30 mm from the ring toward the thick end |
| Rest | Beech, 40 x 20 x 14.8 mm, fixed | (24, -3.5): 85 mm from the ring |
| Wax ring | Solitaire pattern, US size 7, four-claw head for an 8 mm round, one per trial | On the mandrel at (24, -12), 142 mm from its thin end, head up |
| Stone | 8.0 mm round brilliant cubic zirconia, red, 0.63 g, reused | Table-up in the 4.5 mm square hole of a beech block 40 x 40 x 10 mm at (36, -10) |
| Tool block | Beech, 140 x 40 x 30 mm, its length along x, fixed | (33, -30) |
| Tweezers | Diamond tweezers, grooved medium tips, stainless, 160 mm, about 25 g, tips 16 mm apart | Tips-down in the block's trough at (38, -30) |
| Pusher | Setter's pusher: flat square steel face in a hardwood mushroom handle, 95 mm, about 20 g | Face-down in a 5 mm square hole 22 mm deep at (28, -30) |
| Wax pen | Dial-controlled electric wax pen, straight round-point tip 3 mm across and 25 mm long, tip at 95 C | Tip-down in a steel stand at (18.5, -30) |

The three tools stand in a row along x, 9.5 and 10 cm apart, and each is taken from the ring's side of the row (+y). A gripper that closes its fingers along
x can only slide onto a standing tool along y, and the pen and the tweezers stand taller than the fingers can be lifted over; in a row, no tool stands in the
way of another.

`assets/blocks_and_stand.svg` draws the fixtures. The tweezers stand in a trough 22.4 mm long and 26 mm deep whose floor is 2.6 mm wide and whose two long
walls lean apart to 11.3 mm at the block's top: the tips slide down the walls to the floor, and the walls then hold the legs within 3 degrees of upright. A
V-groove cut with a router bit does the same. The trough's ends, the pusher's hole and the pen's stand have a funnel mouth of steel plate at 45 degrees,
6 mm high, so that a tool a few millimetres off slides in; a chamfer of the same size cut into a taller block does the same.

**Bought parts,** with the prices listed on the makers' reseller pages on Nov 19, 2026 (US dollars):

| Item | Product | Price |
|---|---|---|
| Mandrel | JTS wood ring stick, SKU 1302-1050 | 9.95 |
| Tweezers | GEMS Switzerland diamond tweezers, grooved medium tip, 160 mm, SKU 1418-231 | 31.00 |
| Pusher | JTS prong and bezel pusher, square flat steel face, SKU 0710-401 | 11.95 |
| Wax pen | EuroTool Electric Wax Worker, dial control, SKU 2305-7000, with a straight round-point tip | 289.95 |
| Injection wax | Freeman Turquoise flakes, 1 lb, SKU 2311-2031; about 2 g a ring | 25.95 a pound |
| Stone | 8.0 mm round red cubic zirconia | not confirmed; a few dollars |
| V-block, thermocouple meter, loupe, millimetre scale | ordinary bench items | not priced |

The pusher's face, the pen's handle and tip and the tweezers' mass are modelled from typical tools of these kinds; measure the bought ones and adjust
`ring_geometry.py` if they differ by more than a millimetre or a few grams.

**The wax ring, one per trial.** `assets/wax_ring.stl` is the pattern (millimetres) and `assets/wax_ring_drawing.svg` its drawing; `assets/make_ring.py`
writes both from the task's numbers. Two ways to make it:

- **Injected, the factory's way.** Print the STL once as a master, pour a two-part silicone mould round it, and inject the named wax with a small hand-pump
  injector. Such injectors were listed at 389 to 830 dollars on Nov 19, 2026. A ring then takes a few minutes and about 0.11 dollars of wax.
- **Ordered.** Send the STL to a casting house or print bureau that prints in castable wax. No price was confirmed. Printed wax is more brittle than
  injection wax, so calibrate the wax rule on the wax used.

Push the ring by hand onto the mandrel up to a pencil mark 142 mm from the thin end, head up, and lay the mandrel in the V-block and on the rest with the
ring 25 mm beyond the V-block. The stick's taper moves the ring 27 mm for each millimetre of bore, so a ring that stops more than 2 mm short of the mark
or slides past it is not used: take another, and correct the master if they all do. Photograph the new ring's head from above before the trial, to tell
melt marks from moulding flaws afterwards.

**Calibration.** The wax rule's times and forces are estimates. With the real pen at its setting and a real ring on the mandrel, find: how long the tip
must rest on a claw before a light touch turns it and it stays turned (the model's 1.5 s); how long before the claw slumps without being pushed (the
model's 5 s); the push that breaks a cold claw (the model's 0.9 N at the tip); and the push that seats the stone (the model's 0.6 N). Scale the constants
at the top of `wax_rule.py` to them.

**Safety.** The pen's tip is at about 95 C and its handle stays cool; it stands tip-down in a steel tube. The supervisor holds the pen's power switch. The
cord hangs from above and leaves the gripper between the fingers. Stop the trial if the tip comes within 2 cm of a gripper pad or a cable. Only the pen's tip touches wax, so no molten wax reaches the grippers.

**Reset.** A new wax ring, photographed from above. Wipe the stone and check it under the loupe. Stone table-up in its block, tweezers and pusher in the tool block, pen in its
stand and at temperature for five minutes. Room at 20 to 25 C. Consumed per trial: one wax ring.

## The model behind it

Everything is a rigid body; the wax follows a rule, declared in `task.json` and written in `wax_rule.py`. The rule changes only how stiff the wax is. Every
motion is the robot's own.

- **The seat.** The stone's pavilion first rests on a sliding seat, 1.5 mm above home. The seat's drive gives at 0.6 N and stays where it is pressed, so the
  stone goes down under the pusher and stays down. Home is a fixed bearing in the head.
- **A claw** is a hinge with a spring. Cold, its drive rests where the claw was moulded or last set: pushed, it springs back, and it breaks if it is forced
  25 degrees. Once the pen's tip has rested on it for 1.5 s it is soft: its spring is switched off, so it turns where it is pushed against a little
  damping and stays where it is left. With the tip away it cools as fast as it warmed, and sets where it stands. After 5 s of touch it melts. Its hinge
  ends 0.6 degrees past where it lies on a seated stone, so a claw cannot be driven into the stone.
- **Touch** is geometry: the last 4 mm of the pen's tip within 0.25 mm of a claw. Near is not touching. The same test against the shank, the head and the
  stone, for 0.5 s, is a melt mark.
- **A soft claw looks wet** (darker and glossy); a melted one is replaced by a puddle on the stone's edge.

Simplified, and said so in `task.json`: the ring does not slide on the mandrel; the pen's cord is not simulated; the tweezers' legs are sprung hinges
2.5 mm thick (real ones are about 1.2 mm); the claw's spring is softer than real cold wax, which hardly bends before it breaks, while the force that breaks
it is of the real order; the times are estimates.

Colliders. Hull cooking may drop points that stand less than about 0.7 mm out of a hull, so every shape whose exact size matters is a box or a round rod:
the claws, the seat and the bearing (four blocks each), the head's rim (twelve thin blocks), the mandrel (five rods, each as thick as the stick is where it
works), the pen's tip (a rod with a round end), the tweezers and all the fixtures. The stone collides as 13 corners: an eight-sided girdle at the stone's
true radius, a point culet and a square table. It is drawn as a round brilliant.

## Design notes

- **Why the hand must hold.** The mandrel's weight sits 10 mm toward the rest from the V-block's near edge and the ring 25 mm the other way: 0.24 N on
  the ring lifts the thick end, and seating the stone takes 0.6 N. Left alone the mandrel lies steady, and the stone's own weight is 40 times too little
  to tip it. Pressed with nobody holding it, it turns over the V-block's edge until its thin end is on the table, 19 mm down: the work has been dropped,
  and the trial scores 0 from then on. The V-block is short, 10 mm, so that the edge the stick turns about is 25 mm from the ring.
- **Why the tip is steered by sight.** The arm gives about 1 cm for each newton at the pen's tip, five times a claw's height, and its joints stick until
  their command is 2 mm ahead. The demo therefore reads where the tip's ball lies on the claw at every step and aims a little into the face where the claw
  will stand next. The pen's point is 3 mm across for the same reason: a 2 mm point slid off a claw 1.8 mm wide.
- **Why the holding hand leans.** Upright, the holding arm's wrist stood 3 mm from the working hand's fingers at the claw on its side. Leaning 30 degrees
  back along the stick, the two arms keep more than 5 cm apart. The offline tests check this, and the working arm's way past every standing tool, against
  the robot's own collision shapes.
- **Unintended strategies, all in the offline tests:** hovering the pen near a claw; pushing a claw without waiting; forcing it cold; touching for one
  physics substep in each control step; warming the wrong place; closing the claws before pressing the stone; laying the pen on the table; pinching the
  stone with a gripper; putting everything back after a zero.
- **Not punished:** which arm does what, the order of the claws, retries, regrasps, a claw closed in several touches, how the tools are held.
- **No speed or force is scored.** The times in the wax rule belong to the wax, not to the robot.

## Files

| File | What |
|---|---|
| `wax_set_a_stone_on_a_ring_mandrel.py` | The task file: scorer wiring, props, demo |
| `wax_rule.py` | The wax model and the score, no simulator |
| `ring_geometry.py` | Every number and the geometry the scorer and the demo share |
| `test_wax_set_a_stone_on_a_ring_mandrel.py` | Offline checks: scoring, loopholes, geometry, USD, reach, text against code |
| `mechkit.py`, `demokit.py` | Kits from `heat-shrink-lug` |
| `assets/` | `make_ring.py`, `wax_ring.stl`, `wax_ring_drawing.svg`, `blocks_and_stand.svg`: generated by `make_ring.py`, no third-party assets |
| `task.json` | The task's details |

## Attribution and sources

`mechkit.py` and `demokit.py` come from the accepted task `heat-shrink-lug` in this repository, with the additions listed at the top of `mechkit.py`.
Everything else in this folder was written for this task.

- Gerry Lewy, "Setting Stones in Wax" and "Setting Gemstones in Wax", Ganoksin (2016): https://www.ganoksin.com/article/setting-stones-wax/ and
  https://www.ganoksin.com/article/setting-gemstones-wax/ . The wax on a tapered wooden spindle in the hand, never in the fingers.
- Ajit Menon and Julie Martin, "Casting Gemstones in Place", Santa Fe Symposium 1996:
  https://santafesymposium.org/1996-santa-fe-symposium-papers/1996-casting-gemstones-in-place
- Hubert Schuster, "Platinum Casting with Gemstones", Ganoksin (2004): https://www.ganoksin.com/article/platinum-casting-with-gemstones/ . Labour saved by
  setting in wax.
- The cost of a stone set in wax against one set in metal: a US refiner's technical article on casting with gemstones (2008).
- The 1.7 billion stones set by hand in a year: a mass-market maker's sustainability report for 2018, as reported by The Nation (Thailand), 19 June 2019:
  https://www.nationthailand.com/business/30371407 . The report does not say how many of them were set in wax.
- National Skill Development Corporation, India: the qualification "Cast and Diamonds set Jewellery, Wax Setter".
- That wax claws are softened with a heated tool and pressed over the girdle, and that the beginner's fault is holding the tool on too long: a 2026 guide
  to stone-in-place casting, and a setter's account given to the authors.
