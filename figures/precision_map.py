"""
Figure: which linear layers FP8 converts, and which it refuses.

One idea: the 13 skipped layers are exactly the ones that cannot work, and
the count resolves without any hand-waving.

Counts are DERIVED from the model shape and the filter rules, then asserted
against the line the real run printed:

    converted 145/158 linear layers, skipped 13 (too small)
"""

from figlib import Fig, BLUE, GOLD, BROWN, MUTED, TEXT, MONO


DEPTH = 24
MATS_PER_BLOCK = 6          # qkv, proj, and the MLP matrices
LOGGED = "converted 145/158 linear layers, skipped 13 (too small)"


def counts():
    """Apply the FP8 eligibility rules to the depth-24 model."""
    block_mats = DEPTH * MATS_PER_BLOCK
    converted = block_mats + 1                # + lm_head
    # ve_gate is Linear(12, 12) on alternating layers: 12 % 16 != 0
    ve_gates = DEPTH // 2
    # smear_gate is Linear(24, 1): far below the minimum dimension
    skipped = ve_gates + 1
    total = converted + skipped
    assert (converted, skipped, total) == (145, 13, 158), \
        f"derivation drifted: {converted}/{total}, skipped {skipped}"
    return converted, skipped, total, ve_gates


def build():
    converted, skipped, total, ve_gates = counts()

    W = 1000
    PAD_L, PAD_T = 40, 40
    H = 430

    f = Fig(W, H, f"FP8 layer eligibility: {converted} of {total} linear "
            f"layers converted, {skipped} skipped.")

    f.o.append(f'<text x="{PAD_L}" y="26" font-size="17" fill="{TEXT}">'
               f'<tspan fill="{BLUE}" font-weight="600">{converted}</tspan> of {total}'
               f' linear layers run in FP8. The other '
               f'<tspan fill="{GOLD}" font-weight="600">{skipped}</tspan> cannot.</text>')

    # --- the converted block, drawn as one cell per matrix ---------------
    cell, gap = 26, 4
    cols = 24
    y0 = PAD_T + 34
    f.text(PAD_L, y0 - 10, f"{DEPTH} blocks x {MATS_PER_BLOCK} matrices + lm_head",
           13, MUTED)
    for i in range(converted):
        r, c = divmod(i, cols)
        f.rect(PAD_L + c * (cell + gap), y0 + r * (cell + gap), cell, cell, BLUE)

    rows = (converted + cols - 1) // cols
    y1 = y0 + rows * (cell + gap) + 34

    # --- the skipped ones, at the same cell size so the ratio is honest ------
    f.text(PAD_L, y1 - 10, "skipped by the eligibility filter", 13, MUTED)
    for i in range(skipped):
        f.rect(PAD_L + i * (cell + gap), y1, cell, cell, GOLD)

    y2 = y1 + cell + 40
    f.line(PAD_L, y2 - 16, W - PAD_L, y2 - 16, BROWN, 1, opacity=0.4)

    f.o.append(f'<text x="{PAD_L}" y="{y2 + 6}" font-size="14" fill="{MUTED}">'
               f'<tspan fill="{GOLD}">{ve_gates} x ve_gate</tspan> '
               f'<tspan font-family="{MONO}">Linear(12, 12)</tspan>'
               f' &#8212; 12 is not divisible by 16</text>')
    f.o.append(f'<text x="{PAD_L}" y="{y2 + 30}" font-size="14" fill="{MUTED}">'
               f'<tspan fill="{GOLD}">1 x smear_gate</tspan> '
               f'<tspan font-family="{MONO}">Linear(24, 1)</tspan>'
               f' &#8212; far below the minimum dimension</text>')

    f.text(PAD_L, y2 + 58, "the run logged: " + LOGGED, 13, BROWN, mono=True)
    f.h = int(y2 + 58 + 24)
    f.o[1] = f'<rect width="{W}" height="{f.h}" fill="#333333"/>'
    f.o[0] = f.o[0].replace(f"0 0 {W} {H}", f"0 0 {W} {f.h}")
    return f


if __name__ == "__main__":
    path, size = build().save("precision-map.svg")
    c, s, t, _ = counts()
    print(f"  precision-map.svg    {size/1024:5.1f} KB    "
          f"{c}/{t} converted, {s} skipped -- asserted against the run log")
