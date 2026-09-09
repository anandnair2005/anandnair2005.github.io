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

from figlib import Fig, BLUE, GOLD, PURPLE, BROWN, MUTED, TEXT, MONO


WORLD = 4
DEVICE_BATCH = 16
SEQ = 2048
GRAD_ACCUM = 8
TOTAL_BATCH = 1_048_576

assert DEVICE_BATCH * SEQ * WORLD * GRAD_ACCUM == TOTAL_BATCH


def build():
    W = 1000
    PAD_L, PAD_R = 96, 40
    track = W - PAD_L - PAD_R
    LANE_H, LANE_GAP = 40, 10
    y0 = 108

    # phase widths as fractions of the step
    phases = [
        ("forward + backward", 0.52, BLUE, f"{GRAD_ACCUM} accumulation, no communication"),
        ("reduce_scatter", 0.14, PURPLE, "async"),
        ("local update", 0.18, GOLD, "owned shard only"),
        ("all_gather", 0.16, PURPLE, "async"),
    ]

    H = y0 + WORLD * (LANE_H + LANE_GAP) + 118
    f = Fig(W, H, "One optimizer step across four ranks. Gradients stay local "
            "through eight accumulation steps, then reduce-scatter, "
            "local update and all-gather.")

    f.o.append(f'<text x="{PAD_L - 56}" y="30" font-size="17" fill="{TEXT}">'
               f'The model is never DDP-wrapped \u2014 '
               f'<tspan fill="{GOLD}" font-weight="600">the optimizer owns '
               f'synchronisation</tspan></text>')
    f.text(PAD_L - 56, 56,
           f"{WORLD} ranks x {DEVICE_BATCH} x {SEQ:,} tokens x {GRAD_ACCUM} "
           f"accumulation steps = {TOTAL_BATCH:,} tokens per step",
           13, MUTED, mono=True)

    # phase headings
    x = PAD_L
    for name, frac, colour, sub in phases:
        w = track * frac
        f.text(x + w / 2, 88, name, 13, colour, anchor="middle", weight="600")
        x += w

    # lanes
    for r in range(WORLD):
        y = y0 + r * (LANE_H + LANE_GAP)
        f.text(PAD_L - 14, y + LANE_H / 2 + 5, f"rank {r}", 13, MUTED, anchor="end")
        x = PAD_L
        for name, frac, colour, sub in phases:
            w = track * frac
            f.rect(x + 1, y, w - 2, LANE_H, colour, r=4)
            x += w

    y_end = y0 + WORLD * (LANE_H + LANE_GAP)

    # the boundary that matters
    x_sync = PAD_L + track * phases[0][1]
    f.line(x_sync, y0 - 14, x_sync, y_end + 6, TEXT, 2, dash="4 4")
    f.text(x_sync + 10, y_end + 26,
           "an ordinary DDP model would already be all-reducing, here",
           13, TEXT)

    f.line(PAD_L - 56, y_end + 48, W - PAD_R, y_end + 48, BROWN, 1, opacity=0.4)
    f.o.append(f'<text x="{PAD_L - 56}" y="{y_end + 74}" font-size="14" fill="{MUTED}">'
               f'<tspan fill="{PURPLE}">communication</tspan> is an explicit async '
               f'phase; <tspan fill="{GOLD}">each rank updates only its own shard</tspan>'
               f', so optimizer state is never duplicated</text>')

    return f


if __name__ == "__main__":
    path, size = build().save("optimizer-step.svg")
    print(f"  optimizer-step.svg   {size/1024:5.1f} KB    "
          f"{WORLD} ranks x {DEVICE_BATCH}x{SEQ} x {GRAD_ACCUM} = {TOTAL_BATCH:,} "
          f"-- asserted")
