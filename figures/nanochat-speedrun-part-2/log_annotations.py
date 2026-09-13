"""
Figure: the head of the run log, annotated.

One idea: four of the expectations from Part 1 are settled before the first
training step, by lines the script prints on its way up.

Every line below is copied from blog/evidence/part-1/speedrun.log of run
d24-4xh100-full with its real line number, abridged only to drop a leading
status glyph and two parenthetical asides so the text stays ASCII. The asserts
re-derive the arithmetic the log reports, so a mis-transcribed number fails the
build instead of being drawn.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib import (Fig, BLUE, GOLD, TEAL, PURPLE, BROWN, MUTED, TEXT,
                    FS_TITLE, FS_BODY, FS_META, PAD, output_to)

# (log line no., text, expectation settled)
LINES = [
    (342, "Using Flash Attention 3 (Hopper GPU detected)", "6"),
    (348, '  "n_layer": 24,', "1"),
    (349, '  "n_head": 12,', "1"),
    (351, '  "n_embd": 1536,', "1"),
    (354, "converted 145/158 linear layers, skipped 13 (too small)", "3"),
    (363, "Auto-computed optimal batch size: 1,048,576 tokens", "1"),
    (367, "Calculated number of iterations ... ratio: 5,568", "1"),
    (368, "Total number of training tokens: 5,838,471,168", "4"),
    (373, "Total batch size 1,048,576 => gradient accum steps: 8", "2"),
]

EXPECTATIONS = {
    "1": ("depth 24 sets every other dimension", BLUE),
    "2": ("world size changes accumulation, nothing else", TEAL),
    "3": ("FP8 is applied selectively", PURPLE),
    "4": ("the token budget is exact, not rounded", GOLD),
    "6": ("FA3 is detected, not assumed", BROWN),
}

DEPTH, HEADS, DIM = 24, 12, 1536
ITERS, BATCH, MICRO = 5568, 1_048_576, 131_072
TOKENS = 5_838_471_168


def check():
    assert DIM == DEPTH * 64, f"dim {DIM} should be depth x 64"
    assert DIM % HEADS == 0, "head dimension is not an integer"
    assert DIM // HEADS == 128, f"head dim {DIM // HEADS}, expected 128"
    assert ITERS * BATCH == TOKENS, (
        f"{ITERS:,} x {BATCH:,} = {ITERS * BATCH:,}, log says {TOKENS:,}")
    assert BATCH == 2 ** 20, "batch size should be a clean power of two"
    assert BATCH // MICRO == 8, "gradient accumulation should be 8 on 4 GPUs"
    assert 145 + 13 == 158, "FP8 layer counts do not add up"
    nums = [n for n, _, _ in LINES]
    assert nums == sorted(nums), "log lines are out of order"
    assert set(e for _, _, e in LINES) == set(EXPECTATIONS), "legend mismatch"
    return TOKENS


def build():
    check()

    W = 660
    x_ln = PAD
    x_txt = PAD + 44
    y0 = 76
    ROWH = 25
    H = y0 + len(LINES) * ROWH + 134

    f = Fig(W, H, "The opening lines of the training log, each printed value "
            "tied to the Part 1 expectation it settles. Four of them resolve "
            "before the first training step.")

    f.o.append(f'<text x="{PAD}" y="24" font-size="{FS_TITLE}" fill="{TEXT}">'
               f'Four expectations settle in '
               f'<tspan fill="{GOLD}" font-weight="600">31 lines</tspan> '
               f'before step one</text>')
    f.text(PAD, 44, "speedrun.log lines 342 to 373, abridged", FS_META, MUTED)

    for i, (ln, txt, exp) in enumerate(LINES):
        y = y0 + i * ROWH
        colour = EXPECTATIONS[exp][1]
        f.rect(x_ln - 6, y - 13, W - 2 * PAD + 12, ROWH - 3,
               colour, r=3, opacity=0.10)
        f.rect(x_ln - 6, y - 13, 3, ROWH - 3, colour, r=1)
        f.text(x_ln, y, f"{ln}", FS_META, BROWN, mono=True)
        f.text(x_txt, y, txt, FS_BODY, TEXT, mono=True)

    ly = y0 + len(LINES) * ROWH + 24
    f.line(PAD, ly - 12, W - PAD, ly - 12, BROWN, 1, opacity=0.4)
    for j, key in enumerate(sorted(EXPECTATIONS)):
        label, colour = EXPECTATIONS[key]
        y = ly + j * 19
        f.rect(PAD, y - 8, 9, 9, colour, r=2)
        f.text(PAD + 17, y, f"Expectation {key}", FS_META, colour, weight="600")
        f.text(PAD + 104, y, label, FS_META, MUTED)

    f.text(PAD, H - 12,
           f"{ITERS:,} x {BATCH:,} = {TOKENS:,} tokens exactly. "
           f"No remainder, no padding, no truncated final batch.",
           FS_META, BROWN)
    return f


if __name__ == "__main__":
    output_to(__file__)
    tokens = check()
    path, size = build().save("log-annotations.svg")
    print(f"  log-annotations.svg {size/1024:5.1f} KB  ")
    print(f"  {len(LINES)} log lines, {len(EXPECTATIONS)} expectations, "
          f"{tokens:,} tokens re-derived from the printed factors")
