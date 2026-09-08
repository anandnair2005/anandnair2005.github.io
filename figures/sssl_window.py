"""
Figure: the SSSL sliding-window pattern.

One idea: most layers only ever see a quarter of the context.

The pattern is COMPUTED with the same arithmetic as
nanochat/gpt.py::_compute_window_sizes, so the figure cannot drift from the
code it describes.
"""

from figlib import Fig, BLUE, GOLD, BROWN, MUTED, TEXT, MONO


def window_sizes(n_layer=24, sequence_len=2048, pattern="SSSL"):
    """Mirror of nanochat/gpt.py::_compute_window_sizes."""
    long_window = sequence_len
    short_window = -(-long_window // 4 // 128) * 128      # ceil to FA3 tile size
    sizes = [long_window if pattern[i % len(pattern)] == "L" else short_window
             for i in range(n_layer)]
    sizes[-1] = long_window                               # final layer always full
    return sizes, short_window, long_window


def build():
    sizes, short_w, long_w = window_sizes()
    n = len(sizes)
    n_short = sum(1 for s in sizes if s == short_w)

    W = 1000
    PAD_L, PAD_R, PAD_T = 96, 40, 44
    BAR_H, GAP = 16, 6
    track = W - PAD_L - PAD_R
    H = PAD_T + n * (BAR_H + GAP) + 56

    f = Fig(W, H, f"Attention window per layer. {n_short} of {n} layers "
            f"attend to only {short_w} tokens.")

    f.o.append(f'<text x="{PAD_L}" y="26" font-size="17" fill="{TEXT}">'
               f'<tspan fill="{BLUE}" font-weight="600">{n_short} of {n}</tspan>'
               f' layers attend to only '
               f'<tspan font-family="{MONO}">{short_w}</tspan> tokens</text>')

    # bar width IS the window size, so the ratio is the message
    for i, s in enumerate(sizes):
        y = PAD_T + i * (BAR_H + GAP)
        f.rect(PAD_L, y, track * s / long_w, BAR_H,
               GOLD if s == long_w else BLUE)

    for i, label in ((0, "layer 0"), (n - 1, f"layer {n - 1}")):
        y = PAD_T + i * (BAR_H + GAP) + BAR_H - 4
        f.text(PAD_L - 14, y, label, 13, MUTED, anchor="end")

    base_y = PAD_T + n * (BAR_H + GAP) + 4
    for value in (short_w, long_w):
        x = PAD_L + track * value / long_w
        f.line(x, PAD_T - 8, x, base_y, BROWN, 1, dash="3 4")
        f.text(x, base_y + 20, f"{value:,}", 14, MUTED, anchor="middle", mono=True)
    f.text(PAD_L, base_y + 42, "attention window, in tokens", 13, MUTED)

    return f, sizes, short_w, long_w, n_short


if __name__ == "__main__":
    f, sizes, short_w, long_w, n_short = build()
    path, size = f.save("sssl-window.svg")
    long_idx = [i for i, s in enumerate(sizes) if s == long_w]
    print(f"  sssl-window.svg      {size/1024:5.1f} KB    "
          f"short={short_w} (comment says 768), {n_short}/{len(sizes)} short, "
          f"full-context layers {long_idx}")
