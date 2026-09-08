"""
Figure: the --depth cascade.

One idea: one integer fixes the whole run.

Values are DERIVED from the formulas in scripts/base_train.py and then
asserted against the lines the real d24 run printed:

    Auto-computed optimal batch size: 1,048,576 tokens
    Scaling LRs by 1.4142 for batch size 1,048,576 (reference: 524,288)
    Scaling the LR for the AdamW parameters proportional to 1/sqrt(1536/768) = 0.707107
    Calculated number of iterations from target data:param ratio: 5,568
    Total number of training tokens: 5,838,471,168
"""

import math
from figlib import Fig, BLUE, GOLD, BROWN, MUTED, TEXT, MONO

ASPECT_RATIO = 64
HEAD_DIM = 128
B_REF = 2 ** 19            # 524,288, the measured d12 optimum
SCALING_PARAMS_D24 = 729_810_624    # from the run's own reported ratio
RATIO = 8


def derive(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = -( -base_dim // HEAD_DIM) * HEAD_DIM
    return base_dim, model_dim, model_dim // HEAD_DIM


def d24():
    base_dim, model_dim, heads = derive(24)
    target_tokens = RATIO * SCALING_PARAMS_D24
    batch = 2 ** round(math.log2(1_048_576))
    iters = target_tokens // batch
    tokens = iters * batch
    lr_scale = math.sqrt(batch / B_REF)
    adamw_scale = 1 / math.sqrt(model_dim / derive(12)[1])
    assert (model_dim, heads, batch, iters, tokens) == \
           (1536, 12, 1_048_576, 5568, 5_838_471_168), "derivation drifted"
    assert abs(lr_scale - 1.4142) < 1e-4 and abs(adamw_scale - 0.707107) < 1e-6
    return model_dim, heads, batch, iters, tokens, lr_scale, adamw_scale


ROWS = [
    ("depth",             "12",          "24",          True),
    ("base_dim",          "768",         "1,536",       False),
    ("model_dim",         "768",         "1,536",       False),
    ("num_heads",         "6",           "12",          False),
    ("total_batch_size",  "524,288",     "1,048,576",   False),
    ("lr_scale",          "1.0",         "1.4142",      False),
    ("adamw_lr_scale",    "1.0",         "0.707107",    False),
    ("iterations",        "\u2014",      "5,568",       False),
    ("training tokens",   "\u2014",      "5,838,471,168", False),
]

def build():
    d24()    # assert before drawing

    W = 1000
    PAD_L = 40
    LABEL_W = 230
    COL_W = 250
    x_ref = PAD_L + LABEL_W
    x_run = x_ref + COL_W
    ROW_H = 42
    y0 = 92
    H = y0 + len(ROWS) * ROW_H + 56

    f = Fig(W, H, "The --depth cascade: one integer determines width, heads, "
            "batch size, learning-rate scaling and training horizon.")

    f.o.append(f'<text x="{PAD_L}" y="30" font-size="17" fill="{TEXT}">'
               f'One integer fixes '
               f'<tspan fill="{GOLD}" font-weight="600">everything below it</tspan>'
               f'</text>')
    f.text(x_ref, 66, "d12  reference", 13, MUTED, anchor="middle")
    f.text(x_run, 66, "d24  the run", 13, GOLD, anchor="middle")
    f.line(PAD_L, 76, W - PAD_L, 76, BROWN, 1, opacity=0.4)

    for i, (name, ref, run, is_input) in enumerate(ROWS):
        y = y0 + i * ROW_H
        colour = GOLD if is_input else BLUE
        # the spine: one continuous line showing the value flowing down
        if i < len(ROWS) - 1:
            f.line(PAD_L + 8, y + 6, PAD_L + 8, y + ROW_H + 6, BROWN, 2, opacity=0.5)
        f.rect(PAD_L + 3, y - 4, 11, 11, colour, r=6)
        f.text(PAD_L + 26, y + 6, name, 15, TEXT if is_input else MUTED,
               weight="600" if is_input else None, mono=True)
        f.text(x_ref, y + 6, ref, 15, MUTED, anchor="middle", mono=True)
        f.text(x_run, y + 6, run, 15, colour, anchor="middle", mono=True,
               weight="600" if is_input else None)

    y_end = y0 + len(ROWS) * ROW_H
    f.text(PAD_L, y_end + 18, "d12 is the measured anchor, so its scales are "
                              "exactly 1.0; every other depth extrapolates from it.",
           13, BROWN)
    return f


if __name__ == "__main__":
    path, size = build().save("depth-cascade.svg")
    md, h, b, it, tok, lr, aw = d24()
    print(f"  depth-cascade.svg    {size/1024:5.1f} KB    "
          f"d24 -> dim {md}, heads {h}, batch {b:,}, {it:,} iters, "
          f"{tok:,} tokens -- asserted against the run log")
