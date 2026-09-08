"""
Shared drawing helpers for the blog figures.

Palette and design rules follow 3Blue1Brown's manim defaults, read from
manimlib/default_config.yml. The discipline matters more than the colours:

  * one idea per figure
  * no panels, borders, legends or footnotes baked into the image
  * at most three semantic colours; grey for everything de-emphasised
  * blue is the default case, gold is whatever the figure is actually about
  * red only for genuine loss, never as a category alongside green

Output is SVG. Text stays text, so it scales, stays sharp at any zoom, and
is readable by screen readers and search.
"""

import os

BG = "#333333"       # manim camera.background_color: grey, not black
BLUE = "#58C4DD"     # blue_c   - the default case
GOLD = "#F0AC5F"     # gold_c   - the point of the figure
TEAL = "#5CD0B3"     # teal_c   - second category
PURPLE = "#9A72AC"   # purple_c - third category, communication
BROWN = "#736357"    # grey_brown - the "1 brown", de-emphasised
RED = "#FC6255"      # red_c    - discarded, used sparingly
TEXT = "#DDDDDD"     # grey_a
MUTED = "#888888"    # grey_c
DIM = "#555555"

SANS = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
        "'Helvetica,Arial,sans-serif'")
MONO = ("ui-monospace,SFMono-Regular,Consolas,'Liberation Mono',"
        "'Menlo,monospace'")

OUT = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("'", "&quot;"))


class Fig:
    """Minimal SVG builder. Coordinates are user units; 1000 wide is typical."""

    def __init__(self, w, h, label):
        self.w, self.h = w, h
        self.o = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{SANS}" role="img" aria-label="{esc(label)}">',
            f'<rect width="{w}" height="{h}" fill="{BG}"/>',
        ]

    def rect(self, x, y, w, h, fill, r=2, opacity=None, stroke=None, sw=2):
        a = f' opacity="{opacity}"' if opacity is not None else ""
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                      f'height="{h:.1f}" rx="{r}" fill="{fill}"{a}{s}/>')

    def text(self, x, y, s, size=14, fill=TEXT, anchor="start",
             weight=None, mono=False, opacity=None):
        w = f' font-weight="{weight}"' if weight else ""
        f = f' font-family="{MONO}"' if mono else ""
        a = f' opacity="{opacity}"' if opacity is not None else ""
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                      f'fill="{fill}" text-anchor="{anchor}"{w}{f}{a}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=MUTED, sw=1, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = f' opacity="{opacity}"' if opacity is not None else ""
        self.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                      f'y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{a}/>')

    def arrow(self, x1, y1, x2, y2, stroke=MUTED, sw=1.6, head=7):
        """Straight arrow. Used only where direction genuinely carries meaning."""
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        self.line(x1, y1, bx, by, stroke, sw)
        p = [(x2, y2),
             (bx - head * 0.55 * math.sin(-ang), by - head * 0.55 * math.cos(-ang)),
             (bx + head * 0.55 * math.sin(-ang), by + head * 0.55 * math.cos(-ang))]
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in p)
        self.o.append(f'<polygon points="{pts}" fill="{stroke}"/>')

    def save(self, name):
        self.o.append("</svg>")
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.o))
        return path, os.path.getsize(path)
