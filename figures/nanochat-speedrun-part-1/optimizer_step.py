"""
Figure: one optimizer step across four ranks.

One idea: gradients stay rank-local right up to the point where an ordinary
DDP-wrapped model would already have been all-reducing.

Static swimlane rather than an animation. The process is inherently parallel,
and an animation forces the reader to watch a parallel thing serially.

Shapes are from the real run:
    Tokens / micro-batch / rank: 16 x 2048 = 32,768
    Total batch size 1,048,576 => gradient accumulation steps: 8
    Distributed world size: 4
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib import (Fig, BLUE, GOLD, PURPLE, BROWN, MUTED, TEXT, MONO,
                    FS_TITLE, FS_META, PAD, output_to)


WORLD = 4
DEVICE_BATCH = 16
SEQ = 2048
GRAD_ACCUM = 8
TOTAL_BATCH = 1_048_576

assert DEVICE_BATCH * SEQ * WORLD * GRAD_ACCUM == TOTAL_BATCH


def build():
    W = 640
    PAD_L, PAD_R = 62, PAD
    track = W - PAD_L - PAD_R
    LANE_H, LANE_GAP = 22, 6
    y0 = 82

    # phase widths as fractions of the step
    phases = [
        ("forward + backward", 0.52, BLUE, f"{GRAD_ACCUM} accumulation, no communication"),
        ("reduce_scatter", 0.14, PURPLE, "async"),
        ("local update", 0.18, GOLD, "owned shard only"),
        ("all_gather", 0.16, PURPLE, "async"),
    ]

    H = y0 + WORLD * (LANE_H + LANE_GAP) + 84
    f = Fig(W, H, "One optimizer step across four ranks. Gradients stay local "
            "through eight accumulation steps, then reduce-scatter, "
            "local update and all-gather.")

    f.o.append(f'<text x="{PAD}" y="22" font-size="{FS_TITLE}" fill="{TEXT}">'
               f'The model is never DDP-wrapped \u2014 '
               f'<tspan fill="{GOLD}" font-weight="600">the optimizer owns '
               f'synchronisation</tspan></text>')
    f.text(PAD, 40,
           f"{WORLD} ranks x {DEVICE_BATCH} x {SEQ:,} tokens x {GRAD_ACCUM} "
           f"accumulation steps = {TOTAL_BATCH:,} tokens per step",
           FS_META, MUTED, mono=True)

    # phase headings, alternating height so the narrow ones do not collide
    x = PAD_L
    for i, (name, frac, colour, sub) in enumerate(phases):
        w = track * frac
        f.text(x + w / 2, 68 if i % 2 == 0 else 56, name, FS_META, colour, anchor="middle", weight="600")
        x += w

    # lanes
    for r in range(WORLD):
        y = y0 + r * (LANE_H + LANE_GAP)
        f.text(PAD_L - 9, y + LANE_H / 2 + 4, f"rank {r}", FS_META, MUTED, anchor="end")
        x = PAD_L
        for name, frac, colour, sub in phases:
            w = track * frac
            f.rect(x + 1, y, w - 2, LANE_H, colour, r=4)
            x += w

    y_end = y0 + WORLD * (LANE_H + LANE_GAP)

    # the boundary that matters
    x_sync = PAD_L + track * phases[0][1]
    f.line(x_sync, y0 - 10, x_sync, y_end + 4, TEXT, 2, dash="4 4")
    f.text(x_sync + 10, y_end + 20,
           "an ordinary DDP model would already be all-reducing at the dashed line",
           FS_META, TEXT)

    f.line(PAD, y_end + 36, W - PAD_R, y_end + 36, BROWN, 1, opacity=0.4)
    f.o.append(f'<text x="{PAD}" y="{y_end + 56}" font-size="{FS_META}" fill="{MUTED}">'
               f'<tspan fill="{PURPLE}">communication</tspan> is an explicit async '
               f'phase; <tspan fill="{GOLD}">each rank updates only its own shard</tspan>'
               f', so optimizer state is never duplicated</text>')

    return f


if __name__ == "__main__":
    output_to(__file__)
    path, size = build().save("optimizer-step.svg")
    print(f"  optimizer-step.svg   {size/1024:5.1f} KB    "
          f"{WORLD} ranks x {DEVICE_BATCH}x{SEQ} x {GRAD_ACCUM} = {TOTAL_BATCH:,} "
          f"-- asserted")
