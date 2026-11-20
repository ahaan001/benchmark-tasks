# Deal four cards face down off the top of a deck lying on the felt, one into each box

A standard 52-card deck lies face down on a felt cloth with four card boxes taped around it. The robot deals four
cards, one into each box, face down. A playing card is 0.29 mm thick: no gripper can pinch it off the pack. It has to
be moved the way a dealer moves it in European-style ("slide") dealing: a fingertip on its back, friction, and a
slide across the felt.

Category: Societal relatability. Two arms, no tools.

## Files

| File | What it is |
|---|---|
| `task.py` | Scene, scoring hooks, demo hook. The task file to run |
| `layout.py` | Every number of the scene and of the scoring |
| `scoring.py` | The scoring rules on card poses, pure numpy |
| `cardkit.py` | Thin plates with their own friction and contact settings, fingertip positions, reset and demo hooks |
| `deal_demo.py` | The closed-loop scripted demo |
| `test_task.py` | Offline tests: 19 cases, numpy only |
| `task.json` | The submission text, generated from `layout.py` |

## Where the difficulty lies

1. **The card cannot be grasped.** It is 0.29 mm thick and lies on 51 others. The only way to move it is friction
   from above. Cards are finished to slip on each other (the maker's test: a card slides off the next on a 7 degree
   slope, friction about 0.12), so a fingertip that grips the top card better than that takes the top card alone.
2. **The fingertips have to land on the top face and nowhere else.** The pack is 15 mm tall. Fingertips that come
   down beside it and push on its edge shear it: seven cards came off together in the simulator (table below).
3. **A stiff card cannot simply be dragged over the pack's edge.** Once its middle passes the edge it tips, and a
   fingertip pressing on the overhang levers it on the cards below: the second card comes out with it and the next
   ones are pushed back. The card has to be eased off: pushed until most of it overhangs, let go so it tips onto the
   felt by its own weight, then taken by its felt end.
4. **Steering and leaving it.** The card is steered by friction alone into a box with 1.5 cm to spare, and the
   fingertips have to leave straight up or they drag it out again.
5. **Four times, with the pack one card lower each time.**

Simulator runs of poor deals (200 steps each, `DEAL_MODE`):

| Run | What the fingertips do | Result |
|---|---|---|
| `one` | The demo's deal of one card | Top card in box 2, 1.3 and 2.5 mm from its centre; second card moved 0.4 mm; pack 0.0 mm; score 0.25 |
| `light` | First touch 1 mm above the card; then the card is taken by its front end and dragged straight over the pack's edge | The push carried nothing. The straight drag took the second card out with the first (left leaning on the pack, a stray) and pushed the third and fourth back 6 and 5 mm: score 0 when the run ended |
| `edge` | Come down 1 cm behind the pack, 1.5 mm below its top, and push | Seven cards sheared off the pack together: 7 strays, score 0 |
| `rest` | Nothing | Ten loose cards 0.29 mm apart, no drift in 200 steps; score 0 |

What the simulation does not test: with plain Coulomb friction and identical cards, a fingertip that lands cleanly on
the top face always takes exactly one card. A real pack gives up a second card more easily (air film, cling, warp).
See Fidelity.

## Scoring

Rules (numbers in `layout.py`, code in `scoring.py`):

| Term | Rule | On a real trial |
|---|---|---|
| In a box | Flat on the felt (within 5 degrees and 2 mm), all four corners inside the outer edge of the box's tape | By eye; ruler on a corner in doubt |
| Dealt | The only card in its box, face down, and its face never showed upward | Lift the card: no second card under it. Video for a face shown |
| In the pack | Centre within 1 cm of the pack's centre along x and along y, on the pack, face down | Ruler: no edge sticks out more than 1 cm |
| Stray | Any other loose card; in a shared box, the second and further cards | Count them |
| Trial spoiled | The pack's bottom moved more than 2 cm from its place, or a card fell off the table. 0 whatever follows | Video, against the pencil outline |
| Hands off | Tip and pad of all four fingers 3 cm or more from every dealt card | 3 cm spacer |
| At rest | Nothing moved more than 3 mm in the last second | Last second of video |

Score: 0.25 for each box with a dealt card, minus 0.25 for each stray, never below 0. With four boxes dealt and no
stray: 1.0 if hands off and at rest, otherwise 0.75. Any box, any order, either arm. Doing nothing scores 0.
`success()` is true only at 1.0, held for 1 s. Whole-trial memory (a face shown, the pack shoved) clears only when the
embodiment resets.

Checked offline (`test_task.py`): nothing 0; the ladder 0 to 1.0; any box in any order; a finger at 2.9 cm (0.75) and
3.1 cm (1.0); a card 1.4 mm over the tape (stray); a card askew 20 degrees but inside (counts); two cards in one box;
a face shown and turned back (never counts); strays; a pack fanned past 1 cm; a pack shoved 2.1 cm and pushed back
(0); a card off the table (0); the whole pack pushed into a box (0); memory cleared only by reset; a card left
leaning on the pack (not dealt); the full demo on a kinematic stand-in.

## Setup, real build and reset

Everything in the scene is in `task.json` `setup`. Kit:

| Item | Source | Price |
|---|---|---|
| Bicycle Standard playing cards, poker size (UPC 073854008089) | any retailer; WebstaurantStore item 1151001512 | $3.99 |
| Poker table speed cloth, 100% polyester, 60 in wide, 1 yard | upholstery or gaming suppliers | about $14 |
| White 3 mm (1/8 in) graphic tape, pencil, ruler, 3 cm spacer | stationery | about $6 |

Tape the cloth flat. Mark the deck's outline in pencil at x 20, y 0 (long side along x) and tape the four boxes at
the coordinates in `task.json`. Reset: gather the cards, square the deck, set it face down in its outline. Use a new
deck every 50 trials or when a card is bent, creased or sticky. Nothing sharp, wet or powered is involved.

If the robot's fingertips are slicker on a card than cards are on felt, the card will not follow them across the
cloth. The order needed is fingertip on card above card on felt above card on card (0.31, 0.21 and 0.12 here). Speed
cloth is made for cards to slide on.

## Fidelity

- Cards are rigid plates at true size (63.5 x 88.9 x 0.29 mm, 1.55 g). They do not bend. Sliding a card off a 15 mm
  pack bends a real card very little; drawing one from a dealing shoe bends it a lot, which is why there is no shoe.
- The top 10 cards are separate bodies; the 42 below are one block of the same size, weight and friction.
- Friction is Coulomb friction, averaged over the two surfaces by the simulator: card 0.12 (static) and 0.10, felt
  0.30 and 0.26 (est), the robot's stock fingers 0.5. No air film, cling or warp.
- Under the fingertips' press (0.4 mm commanded) the ten loose cards close up by about 0.6 mm in total and spring
  back when released. It is a solver effect, smaller than one pixel, and no card is carried by it.
- Card backs are a plain red panel with a white border. No artwork, no third-party asset.

## Run a trial

In the simulator's environment (the Robocurve Brev launchable), from this folder:

```bash
I="Deal four cards from the deck, face down, one at a time: one card into each white box. Leave the rest of the deck where it is."
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "$I" --scorer task_score --max-steps 2400 --store-frames \
  --log-dir logs/dream
# The scripted demo (DEAL_MODE=one, light, edge or rest for the other runs above)
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "$I" --scorer task_score --max-steps 2400 --store-frames \
  --log-dir logs/demo
# Offline tests (numpy only)
PYTHONPATH=<yam-isaac-lab-arena> python3 test_task.py
```

`TASK_DEBUG=1` prints a line every 10 steps: each top card's position, the pack's, the fingertips' height and the
tally.

## The demo

Closed loop on the cards' poses (`deal_demo.py`); it never moves a card except through the fingertips. Fingers 4 cm
apart and pointing straight down, so two fingertips ride on the card and hold its heading. Per card: both tips come
down on the back half of the top card, commanded 0.4 mm into it; push it forward at 4 cm/s until the tips are 8 mm
short of the pack's front edge, the card then overhanging by about 7 cm; lift, and the card tips onto the felt and
leans on the pack; come down on the card 1.5 cm behind its front edge; slide it at 6 cm/s until its centre is within
2 mm of the box's centre (its back end leaves the pack on the way and it drops flat); lift straight up. The left arm
deals boxes 1 and 2, the right arm boxes 3 and 4.

## Results

The 200-step runs are in the table under "Where the difficulty lies". Full-length demo and Dream-1 results are added with the submission.

One fixed layout (`object_jitter=0`); no seeds.

## Credits and licences

`cardkit.py` is adapted from `jengakit.py` of the accepted task `jenga-push-and-stack` in this repository (the
spawner for boxes with their own contact settings, and the reset and scripted-policy hooks). The YAM robot model
comes with yam_arena (I2RT Robotics, MIT License). No third-party meshes, textures or fonts.
