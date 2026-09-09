"""
Figure: BOS-aligned best-fit packing.

One idea: the row is filled exactly, and the only way to do that is to crop.

Static rather than animated. The algorithm is a loop, so the figure shows
successive states of one row stacked vertically - reader-paced by nature,
and legible in print, RSS and email where an animation is not.

Document lengths are simulated. The point is the mechanism, not the numbers.
"""

from figlib import Fig, BLUE, TEAL, GOLD, RED, BROWN, MUTED, TEXT, MONO

T = 2048
ROW_CAPACITY = T + 1

# One representative row, taken from a faithful replay of the loader's inner
# loop over a warmed buffer: two documents placed whole, then a crop that
# fills the remainder exactly.
PLACED = [1177, 602]
CROP_DOC = 533
CROP_TAKE = ROW_CAPACITY - sum(PLACED)
CROP_WASTE = CROP_DOC - CROP_TAKE

assert CROP_TAKE > 0 and sum(PLACED) + CROP_TAKE == ROW_CAPACITY
assert (CROP_TAKE, CROP_WASTE) == (270, 263)


def build():
    W = 900
    PAD_L, PAD_R = 40, 40
    WASTE_LANE = 120        # reserved so the discarded tail has room to sit
    track = W - PAD_L - PAD_R - WASTE_LANE
    ROW_H, STEP_GAP = 44, 30
    y0 = 96
    px = track / ROW_CAPACITY

    steps = []
    pos = 0
    for i, d in enumerate(PLACED):
        steps.append(("fit", list(PLACED[:i + 1]), None,
                      f"largest that fits in {ROW_CAPACITY - pos:,}: {d:,}"))
        pos += d
    steps.append(("crop", list(PLACED), CROP_TAKE,
                  f"nothing fits {ROW_CAPACITY - pos:,} \u2014 crop the shortest"))

    H = y0 + len(steps) * (ROW_H + STEP_GAP) + 92
    f = Fig(W, H, "BOS-aligned best-fit packing: documents placed whole until "
            "none fits, then one document cropped to fill the row exactly.")

    f.o.append(f'<text x="{PAD_L}" y="30" font-size="17" fill="{TEXT}">'
               f'Every row comes out '
               f'<tspan fill="{GOLD}" font-weight="600">exactly full</tspan>'
               f' \u2014 which is only possible if something gets cropped</text>')
    f.text(PAD_L, 58, f"row capacity = T + 1 = {ROW_CAPACITY:,} tokens, "
           f"not {T:,}: inputs and targets are the same row offset by one",
           13, MUTED)

    palette = [BLUE, TEAL, BLUE, TEAL]

    for si, (kind, docs, take, caption) in enumerate(steps):
        y = y0 + si * (ROW_H + STEP_GAP)
        # empty capacity behind everything. Documents in a row sit flush
        # against each other, so these keep a tighter radius than the library
        # default: a larger one opens gaps between adjacent blocks and works
        # against the point of the figure, which is that the row is exactly full.
        f.rect(PAD_L, y, track, ROW_H, "#3d3d3d", r=3)

        x = PAD_L
        for di, d in enumerate(docs):
            w = d * px
            f.rect(x, y, w, ROW_H, palette[di % len(palette)], r=3)
            f.rect(x, y, 3, ROW_H, TEXT, r=0)      # the BOS token
            if w > 60:
                f.text(x + w / 2, y + ROW_H / 2 + 5, f"{d:,}", 14, "#1a1a1a",
                       anchor="middle", mono=True)
            x += w

        if kind == "crop":
            w = take * px
            f.rect(x, y, w, ROW_H, GOLD, r=3)
            f.rect(x, y, 3, ROW_H, TEXT, r=0)
            if w > 60:
                f.text(x + w / 2, y + ROW_H / 2 + 5, f"{take:,}", 14, "#1a1a1a",
                       anchor="middle", mono=True)
            # the discarded tail, at the same token scale, in its own lane
            wx = x + w + 10
            f.rect(wx, y + 11, CROP_WASTE * px, ROW_H - 22, RED, r=2, opacity=0.4)
            f.text(wx, y + ROW_H + 16, f"{CROP_WASTE:,} discarded", 13, RED)

        f.text(PAD_L, y - 9, f"{si + 1}. {caption}", 13, MUTED)

    y_end = y0 + len(steps) * (ROW_H + STEP_GAP)
    f.line(PAD_L, y_end + 2, W - PAD_L, y_end + 2, BROWN, 1, opacity=0.4)
    f.o.append(f'<text x="{PAD_L}" y="{y_end + 30}" font-size="14" fill="{MUTED}">'
               f'<tspan font-family="{MONO}">'
               f'{" + ".join(f"{d:,}" for d in PLACED)} + {CROP_TAKE:,}'
               f' = {ROW_CAPACITY:,}</tspan>'
               f' \u2014 no padding, so every token is trained on</text>')
    f.text(PAD_L, y_end + 54,
           "The cropped document is always the shortest in the buffer, "
           "which minimises what is thrown away at that step.", 13, BROWN)
    return f


if __name__ == "__main__":
    path, size = build().save("bestfit-packing.svg")
    print(f"  bestfit-packing.svg  {size/1024:5.1f} KB    "
          f"{' + '.join(str(d) for d in PLACED)} + {CROP_TAKE} = {ROW_CAPACITY}, "
          f"{CROP_WASTE} discarded")
