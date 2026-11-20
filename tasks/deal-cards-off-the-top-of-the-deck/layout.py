"""Every number of the scene (metres; x forward from the midpoint between the arm bases, y to the robot's left)."""

# Poker-size playing card (Bicycle Standard): 2.5 x 3.5 in; 52 cards measure 15.15 mm; 1.55 g a card.
CARD_L, CARD_W, CARD_T = 0.0889, 0.0635, 0.00029
CARD_MASS = 0.00155
DECK_N = 52
LOOSE_N = 10  # the top cards that are separate bodies; the rest of the pack is one block
CARD_GAP = 1e-5  # spawn gap between neighbours
DECK_XY = (0.20, 0.0)  # the pack's centre; its long side runs along x

# Friction (static, dynamic). The simulator averages the two surfaces in contact; the stock fingers are 0.5.
CARD_MU = (0.12, 0.10)  # card on card: the maker's slip test, a card slides off the next on a 7 degree slope
FELT_MU = (0.30, 0.26)  # with a card's: 0.21 card on felt (est)
FINGER_MU = 0.5

FELT_X, FELT_Y, FELT_T = (0.05, 0.65), (-0.35, 0.35), 0.002

# Card boxes: the clear space inside the tape, the tape's width, and each box's centre (box 1 on the robot's left).
BOX_L, BOX_W, TAPE = 0.12, 0.095, 0.003
BOXES = ((0.30, 0.25), (0.41, 0.10), (0.41, -0.10), (0.30, -0.25))

FELT_GREEN = (0.05, 0.36, 0.17)
CARD_WHITE = (0.96, 0.96, 0.94)
CARD_RED = (0.74, 0.08, 0.10)
TAPE_WHITE = (0.97, 0.97, 0.97)

# Scoring.
PACK_MOVED = 0.02  # the pack's bottom moved this far from its place: 0 for the trial
IN_PACK = 0.01  # a card counts as still in the pack within this of the pack's centre, along x and along y
FLAT_DEG = 5.0  # a dealt card lies within this of level
ON_FELT = 0.002  # ... and this close to the felt
CLEAR = 0.03  # every finger this far from every dealt card
REST_S, REST_M = 1.0, 0.003
OFF_TABLE = 0.05  # below the table top
RUN_LIMIT = 2400
