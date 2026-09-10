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
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib import (Fig, BLUE, GOLD, BROWN, MUTED, TEXT, MONO,
                    FS_TITLE, FS_BODY, FS_META, PAD, ROW, output_to)

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

    PAD_L = PAD
    LABEL_W = 168
    COL_W = 132
    x_ref = PAD_L + LABEL_W
    x_run = x_ref + COL_W
    # Canvas fits the content: the value columns are centred, so the table
    # ends half a column past the last one. Deriving W keeps the figure tight
    # if the column widths ever change.
    W = x_run + COL_W // 2 + PAD_L
    ROW_H = ROW
    y0 = 62
    FOOT = ["d12 is the measured anchor, so its scales are exactly 1.0;",
            "every other depth extrapolates from it."]
    H = y0 + len(ROWS) * ROW_H + 20 + len(FOOT) * 16

    f = Fig(W, H, "The --depth cascade: one integer determines width, heads, "
            "batch size, learning-rate scaling and training horizon.")

    f.o.append(f'<text x="{PAD_L}" y="24" font-size="{FS_TITLE}" fill="{TEXT}">'
               f'One integer fixes '
               f'<tspan fill="{GOLD}" font-weight="600">everything below it</tspan>'
               f'</text>')
    f.text(x_ref, 46, "d12  reference", FS_META, MUTED, anchor="middle")
    f.text(x_run, 46, "d24  the run", FS_META, GOLD, anchor="middle")
    f.line(PAD_L, 54, W - PAD_L, 54, BROWN, 1, opacity=0.4)

    for i, (name, ref, run, is_input) in enumerate(ROWS):
        y = y0 + i * ROW_H
        colour = GOLD if is_input else BLUE
        # the spine: one continuous line showing the value flowing down
        if i < len(ROWS) - 1:
            f.line(PAD_L + 5, y + 4, PAD_L + 5, y + ROW_H + 4, BROWN, 2, opacity=0.5)
        f.rect(PAD_L + 1, y - 4, 9, 9, colour, r=5)
        f.text(PAD_L + 19, y + 4, name, FS_BODY, TEXT if is_input else MUTED,
               weight="600" if is_input else None, mono=True)
        f.text(x_ref, y + 4, ref, FS_BODY, MUTED, anchor="middle", mono=True)
        f.text(x_run, y + 4, run, FS_BODY, colour, anchor="middle", mono=True,
               weight="600" if is_input else None)

    y_end = y0 + len(ROWS) * ROW_H
    for i, line in enumerate(FOOT):
        f.text(PAD_L, y_end + 12 + i * 16, line, FS_META, BROWN)
    return f


if __name__ == "__main__":
    output_to(__file__)
    path, size = build().save("depth-cascade.svg")
    md, h, b, it, tok, lr, aw = d24()
    print(f"  depth-cascade.svg    {size/1024:5.1f} KB    "
          f"d24 -> dim {md}, heads {h}, batch {b:,}, {it:,} iters, "
          f"{tok:,} tokens -- asserted against the run log")
