"""
Figure: the shape of the repo.

One idea: speedrun.sh is the whole project's table of contents, and almost
everything expensive hangs off one script.

Stage wall-times are from the real run and are used only to size the stage
markers, so the reader can see where the time goes before Part 2 says so.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib import (Fig, BLUE, GOLD, TEAL, BROWN, MUTED, TEXT, MONO,
                    FS_TITLE, FS_BODY, FS_META, PAD, output_to)


# speedrun.sh, in order. Minutes are from the d24-4xH100-full run.
STAGES = [
    ("dataset download",  None,   []),
    ("tok_train",         None,   ["nanochat/tokenizer"]),
    ("tok_eval",          None,   []),
    ("base_train",       196.5,   ["gpt.py", "optim.py", "dataloader.py",
                                   "flash_attention.py", "fp8.py"]),
    ("base_eval",         11.0,   []),
    ("chat_sft",          27.0,   ["gpt.py", "optim.py"]),
    ("chat_eval",         13.0,   ["engine.py"]),
    ("report generate",    2.8,   []),
]


MODULES = [
    ("gpt.py",             "model, attention, window pattern"),
    ("optim.py",           "DistMuonAdamW"),
    ("dataloader.py",      "BOS-aligned best-fit packing"),
    ("flash_attention.py", "FA3 on Hopper, SDPA elsewhere"),
    ("fp8.py",             "tensorwise scaling"),
    ("engine.py",          "KV-cache inference"),
]


def build():
    PAD_L = PAD_R = PAD
    LEFT_W = 246
    x_mod = PAD_L + LEFT_W + 92
    ROW_H = 37            # two lines of text per module row, so wider than ROW
    y0 = 74
    FOOT_W = 604
    W = max(x_mod + 210 + PAD_R, FOOT_W + PAD_L + PAD_R)
    H = y0 + max(len(STAGES), len(MODULES)) * ROW_H + 58

    base_mins = next(s for n, s, _ in STAGES if n == "base_train")
    other_mins = sum(s for n, s, _ in STAGES if s and n != "base_train")

    f = Fig(W, H, "The NanoChat repo: speedrun.sh stages on the left, library "
            "modules on the right. Base training dominates.")

    f.o.append(f'<text x="{PAD_L}" y="24" font-size="{FS_TITLE}" fill="{TEXT}">'
               f'<tspan font-family="{MONO}">runs/speedrun.sh</tspan>'
               f' is the whole project\u2019s table of contents</text>')
    f.text(PAD_L, 50, "pipeline stage", FS_META, MUTED)
    f.text(x_mod, 50, "library module", FS_META, MUTED)
    f.line(PAD_L, 58, W - PAD_R, 58, BROWN, 1, opacity=0.4)

    mod_y = {}
    for i, (name, desc) in enumerate(MODULES):
        y = y0 + i * ROW_H
        mod_y[name] = y
        f.rect(x_mod - 11, y - 11, 6, 17, BLUE, r=3)
        f.text(x_mod, y, name, FS_BODY, TEXT, mono=True)
        f.text(x_mod, y + 16, desc, FS_META, MUTED)

    busiest = max(s for _, s, _ in STAGES if s)
    for i, (name, mins, uses) in enumerate(STAGES):
        y = y0 + i * ROW_H
        hot = name == "base_train"
        colour = GOLD if hot else BLUE
        # bar length shows the wall clock actually goes
        w = 108 * (mins / busiest) if mins else 5
        f.rect(PAD_L, y - 10, w, 15, colour, r=4, opacity=1.0 if hot else 0.55)
        f.text(PAD_L + max(w, 5) + 9, y + 2, name, FS_BODY,
               TEXT if hot else MUTED, mono=True,
               weight="600" if hot else None)
        if mins:
            f.text(PAD_L + LEFT_W, y + 2, f"{mins:.0f}m", FS_META,
                   GOLD if hot else MUTED, anchor="end", mono=True)
        for m in uses:
            key = m.split("/")[-1]
            if key in mod_y:
                f.line(PAD_L + LEFT_W + 10, y, x_mod - 16, mod_y[key],
                       GOLD if hot else BROWN, 1.2,
                       opacity=0.7 if hot else 0.3)

    y_end = y0 + max(len(STAGES), len(MODULES)) * ROW_H
    ratio = base_mins / other_mins
    f.line(PAD_L, y_end + 2, W - PAD_R, y_end + 2, BROWN, 1, opacity=0.4)
    f.o.append(f'<text x="{PAD_L}" y="{y_end + 24}" font-size="{FS_META}" fill="{MUTED}">'
               f'<tspan fill="{GOLD}" font-weight="600">base_train</tspan>'
               f' is {ratio:.1f}x every other stage combined, and touches five of '
               f'the six modules. Everything after this is a zoom into it.</text>')

    return f


if __name__ == "__main__":
    output_to(__file__)
    path, size = build().save("repo_map.svg")
    total_other = sum(s for n, s, _ in STAGES if s and n != "base_train")
    base = next(s for n, s, _ in STAGES if n == "base_train")
    print(f"  repo-map.svg         {size/1024:5.1f} KB    "
          f"base_train {base}m vs {total_other:.1f}m for everything else "
          f"= {base/total_other:.1f}x")
