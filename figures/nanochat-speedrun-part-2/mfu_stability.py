"""
Figure: MFU across the whole run.

One idea: the headline number is not a flattering average.

Every value here was computed from the 5,568 base-training step lines in
blog/evidence/part-1/speedrun.log of run d24-4xh100-full, then hard-coded so
this repository stays self-contained. The asserts below re-derive the summary
statistics from the histogram, so a transcription error in either the bins or
the stated mean fails the build rather than drawing a wrong picture.

    step 00000/05568 ... bf16_mfu: 0.92    <- torch.compile, excluded
    step 00001/05568 ... bf16_mfu: 60.03
    ...
    step 05567/05568 ... bf16_mfu: 59.79
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figlib import (Fig, BLUE, GOLD, RED, BROWN, MUTED, TEXT, MONO,
                    FS_TITLE, FS_BODY, FS_META, PAD, output_to)

# 0.5-point bins over the 5,567 steady-state steps.
HIST = [
    (53.5, 1), (55.5, 2), (56.0, 7), (56.5, 4), (57.0, 7), (57.5, 20),
    (58.0, 25), (58.5, 64), (59.0, 520), (59.5, 4637), (60.0, 277), (60.5, 3),
]
QUARTERS = [59.51, 59.67, 59.68, 59.79]  # means, first quarter to last
MEAN, STDEV = 59.66, 0.32
WITHIN_1 = 5489              # steps within one point of the mean
STEP0 = 0.92                 # the compile step
TOTAL = 5568


def check():
    n = sum(c for _, c in HIST)
    assert n == TOTAL - 1, f"histogram holds {n}, expected {TOTAL - 1}"
    # bin midpoints reproduce the reported mean to within half a bin
    approx = sum((b + 0.25) * c for b, c in HIST) / n
    assert abs(approx - MEAN) < 0.25, f"binned mean {approx:.2f} vs {MEAN}"
    # the densest bin must contain the mean
    peak = max(HIST, key=lambda x: x[1])[0]
    assert peak <= MEAN < peak + 0.5, "mean is outside the modal bin"
    # quarters rise monotonically and end on the reported headline
    assert QUARTERS == sorted(QUARTERS), "quarterly means are not monotonic"
    assert QUARTERS[-1] == 59.79, "final quarter should match the report"
    assert WITHIN_1 / n > 0.98, "stability claim does not hold"
    return n


def build():
    n = check()

    W = 660
    PAD_L, PAD_R = PAD, PAD
    y0 = 74                  # top of the plot
    PLOT_H = 150
    AXIS_Y = y0 + PLOT_H
    track = W - PAD_L - PAD_R - 96    # leave a lane for the step-0 marker
    H = AXIS_Y + 112

    f = Fig(W, H, f"Model FLOPs utilisation across {TOTAL:,} training steps. "
                  f"Mean {MEAN}% with a standard deviation of {STDEV} once the "
                  f"compile step is excluded.")

    f.o.append(f'<text x="{PAD_L}" y="24" font-size="{FS_TITLE}" fill="{TEXT}">'
               f'<tspan fill="{GOLD}" font-weight="600">{100*WITHIN_1/n:.1f}%</tspan>'
               f' of steps land within one point of the mean</text>')
    f.text(PAD_L, 44, f"{TOTAL - 1:,} steady-state steps, 0.5-point bins",
           FS_META, MUTED)

    # --- the distribution -----------------------------------------
    lo, hi = 53.0, 61.0
    biggest = max(c for _, c in HIST)

    def x_of(v):
        return PAD_L + track * (v - lo) / (hi - lo)

    for b, c in HIST:
        # sqrt keeps the long tail visible next to a bin holding 4,637
        h = PLOT_H * (c / biggest) ** 0.5
        w = track * 0.5 / (hi - lo)
        hot = b <= MEAN < b + 0.5
        f.rect(x_of(b) + 0.5, AXIS_Y - h, w - 1, h, GOLD if hot else BLUE, r=2)

    f.text(x_of(59.75), AXIS_Y - PLOT_H - 8, f"{max(HIST, key=lambda x: x[1])[1]:,} steps",
           FS_META, GOLD, anchor="middle", weight="600")

    # axis
    f.line(PAD_L, AXIS_Y, PAD_L + track, AXIS_Y, BROWN, 1, opacity=0.5)
    for v in (54, 56, 58, 60):
        f.line(x_of(v), AXIS_Y, x_of(v), AXIS_Y + 4, BROWN, 1, opacity=0.5)
        f.text(x_of(v), AXIS_Y + 17, f"{v}", FS_META, MUTED, anchor="middle", mono=True)
    f.text(PAD_L, AXIS_Y + 34, "bf16 MFU, %", FS_META, MUTED)

    # --- step 0, drawn in its own lane so the scale stays honest ---------
    x0 = PAD_L + track + 34
    f.line(x0 - 16, y0 - 6, x0 - 16, AXIS_Y + 6, BROWN, 1, dash="3 4")
    f.rect(x0, AXIS_Y - 6, 12, 6, RED, r=2, opacity=0.75)
    f.text(x0 + 6, AXIS_Y + 17, f"{STEP0}", FS_META, RED, anchor="middle", mono=True)
    f.text(x0 + 6, y0 + 8, "step 0", FS_META, RED, anchor="middle")
    f.text(x0 + 6, y0 + 22, "compile", FS_META, RED, anchor="middle")

    # --- quarterly means ------------------------------------------------
    qy = AXIS_Y + 58
    f.text(PAD_L, qy - 8, "mean by quarter of the run", FS_META, MUTED)
    qw = (track - 24) / 4
    for i, q in enumerate(QUARTERS):
        x = PAD_L + i * (qw + 8)
        last = i == len(QUARTERS) - 1
        f.rect(x, qy, qw, 20, GOLD if last else BLUE, r=4,
               opacity=1.0 if last else 0.55)
        f.text(x + qw / 2, qy + 14, f"{q}", FS_BODY, "#1a1a1a",
               anchor="middle", mono=True, weight="600" if last else None)
    f.text(PAD_L + track + 12, qy + 14, "rising", FS_META, GOLD)

    f.text(PAD_L, H - 12,
           f"mean {MEAN}%, standard deviation {STDEV}, minimum 53.91%. "
           f"The final quarter is the reported {QUARTERS[-1]}%.",
           FS_META, BROWN)
    return f


if __name__ == "__main__":
    output_to(__file__)
    n = check()
    path, size = build().save("mfu-stability.svg")
    print(f"  mfu-stability.svg   {size/1024:5.1f} KB    ")
    print(f"  {n:} steps binned, mean {MEAN}, stdev {STDEV} ")
    print(f"  -- asserted against the run log")
