"""
Figure: how the run was orchestrated.

One idea: the rented host is the only disposable part, so every artifact has
to leave it continuously rather than at the end.

The asymmetry is the point. Restore happens once, at the start, and moves a
small amount of setup. Sync happens throughout, and moves everything the run
produces. A host that vanishes mid-run costs time, not results.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib import (Fig, BLUE, GOLD, TEAL, PURPLE, RED, BROWN, MUTED, TEXT,
                    FS_TITLE, FS_BODY, FS_META, PAD, output_to)

LOCAL = ["scripted launch", "config and secrets", "log tailing"]
HOST = ["tokenizer", "base pretrain", "SFT", "evaluations"]
DRIVE = ["checkpoints, 9.3 GiB", "speedrun.log", "eval reports"]

WALL_CLOCK = "4h 36m"
COST = "$41.16"


def check():
    assert LOCAL and HOST and DRIVE, "a column is empty"
    # the host column is the one that disappears; it must own the compute
    assert any("pretrain" in s for s in HOST), "compute is not on the host"
    assert not any("checkpoint" in s for s in HOST), (
        "checkpoints must live on the durable side, not the ephemeral host")
    return True


def build():
    check()

    W = 660
    BOXW, BOXH = 176, 150
    y0 = 92
    gap = (W - 2 * PAD - 3 * BOXW) / 2
    xs = [PAD + i * (BOXW + gap) for i in range(3)]
    H = y0 + BOXH + 132

    f = Fig(W, H, "The orchestration: a local driver, an ephemeral rented "
            "host that does the compute, and cloud storage as the durable "
            "layer. Results are synced off the host continuously.")

    f.o.append(f'<text x="{PAD}" y="24" font-size="{FS_TITLE}" fill="{TEXT}">'
               f'Only the middle column is '
               f'<tspan fill="{RED}" font-weight="600">disposable</tspan>'
               f'</text>')
    f.text(PAD, 44, "so results have to leave it while the run is still going",
           FS_META, MUTED)

    cols = [
        ("Local machine", "orchestrator", LOCAL, BLUE, "persistent"),
        ("Vast.ai host", "4 x H100", HOST, GOLD, "ephemeral"),
        ("Google Drive", "durable store", DRIVE, TEAL, "persistent"),
    ]

    for (title, sub, items, colour, life), x in zip(cols, xs):
        dashed = life == "ephemeral"
        f.rect(x, y0, BOXW, BOXH, colour, r=6, opacity=0.12,
               stroke=colour if dashed else None, sw=1.5)
        if dashed:
            f.o.append(
                f'<rect x="{x}" y="{y0}" width="{BOXW}" height="{BOXH}" '
                f'rx="6" fill="none" stroke="{RED}" stroke-width="1.5" '
                f'stroke-dasharray="5 4"/>')
        f.text(x + 12, y0 + 22, title, FS_BODY, colour, weight="600")
        f.text(x + 12, y0 + 38, sub, FS_META, MUTED, mono=True)
        for k, it in enumerate(items):
            yy = y0 + 62 + k * 19
            f.rect(x + 12, yy - 7, 5, 5, colour, r=1)
            f.text(x + 24, yy, it, FS_META, TEXT)
        f.text(x + BOXW / 2, y0 + BOXH + 18,
               "vanishes on exit" if dashed else life,
               FS_META, RED if dashed else MUTED, anchor="middle")

    # --- the two flows, deliberately unequal ---------------------------
    mid = y0 + 46
    f.arrow(xs[0] + BOXW + 4, mid, xs[1] - 4, mid, BLUE, 1.4)
    f.text((xs[0] + BOXW + xs[1]) / 2, mid - 9, "launch", FS_META, BLUE,
           anchor="middle")

    low = y0 + 108
    f.arrow(xs[2] - 4, low, xs[1] + BOXW + 4, low, TEAL, 1.4)
    f.text((xs[1] + BOXW + xs[2]) / 2, low - 9, "restore", FS_META, TEAL,
           anchor="middle")
    f.text((xs[1] + BOXW + xs[2]) / 2, low + 14, "once, small", FS_META, MUTED,
           anchor="middle")

    f.arrow(xs[1] + BOXW + 4, mid, xs[2] - 4, mid, GOLD, 2.4)
    f.text((xs[1] + BOXW + xs[2]) / 2, mid - 9, "sync", FS_META, GOLD,
           anchor="middle", weight="600")
    f.text((xs[1] + BOXW + xs[2]) / 2, mid + 14, "continuous, everything",
           FS_META, GOLD, anchor="middle")

    fy = H - 34
    f.line(PAD, fy - 14, W - PAD, fy - 14, BROWN, 1, opacity=0.4)
    f.text(PAD, fy, f"One run: {WALL_CLOCK} wall clock, {COST} of rented time.",
           FS_META, BROWN)
    f.text(PAD, fy + 16,
           "A host lost mid-run costs the hours since the last sync, not the run.",
           FS_META, MUTED)
    return f


if __name__ == "__main__":
    output_to(__file__)
    check()
    path, size = build().save("orchestration.svg")
    print(f"  orchestration.svg    {size/1024:5.1f} KB    "
          f"3 tiers, 1 ephemeral, restore/sync asymmetry shown by arrow weight")
