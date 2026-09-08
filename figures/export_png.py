"""
Export every figure SVG to a PNG for Medium syndication.

    pip install playwright
    playwright install chromium
    python figures/export_png.py

The site itself serves SVG and needs none of this. Medium, however, accepts
only .JPG / .JPEG / .GIF / .PNG - not SVG - and wants images at least 1192px
wide before it offers all placement options. The figures are authored at
1000px, so everything is rendered at 2x (2000px) to clear that bar.

Playwright is used rather than a native SVG rasteriser because it is the same
engine that renders the live site: text metrics, font fallback and wrapping
come out identical to what a reader sees. Lighter converters (svglib, cairosvg)
re-implement SVG text layout and can silently shift labels.

On a managed/corporate machine the browser Playwright downloads often cannot
execute ("This program is blocked by group policy"). The script therefore
prefers an already-installed system browser (Edge, then Chrome) and only falls
back to Playwright's bundled Chromium, so `playwright install` may be
unnecessary.

Only needs re-running when a figure changes; figures/png/ is committed.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "png"
SCALE = 2
MEDIUM_MIN_WIDTH = 1192

# Tried in order. None = Playwright's own bundled Chromium.
CHANNELS = ["msedge", "chrome", None]


def launch(pw):
    """Return a browser, preferring system installs over the bundled build."""
    errors = []
    for channel in CHANNELS:
        try:
            browser = pw.chromium.launch(channel=channel) if channel \
                      else pw.chromium.launch()
            print(f"  using {channel or 'bundled chromium'}")
            return browser
        except Exception as exc:
            first = str(exc).strip().splitlines()[0]
            errors.append(f"  {channel or 'bundled chromium'}: {first}")
    raise RuntimeError("no usable browser found:\n" + "\n".join(errors))


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. It is needed only for PNG export;\n"
            "building the SVGs (python figures/build_all.py) needs nothing.\n\n"
            "    pip install playwright\n"
            "    playwright install chromium",
            file=sys.stderr,
        )
        return 1

    svgs = sorted(HERE.glob("*.svg"))
    if not svgs:
        print("no SVGs found - run build_all.py first", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    print(f"exporting {len(svgs)} figures at {SCALE}x")
    failures = []

    with sync_playwright() as pw:
        browser = launch(pw)
        page = browser.new_page()
        for svg in svgs:
            # An .svg opened directly is an SVG document with no document.body,
            # so host the markup inside a real HTML page instead.
            markup = svg.read_text(encoding="utf-8")
            page.set_content(
                "<!doctype html><meta charset='utf-8'>"
                "<style>html,body{margin:0;padding:0}svg{display:block}</style>"
                + markup
            )
            size = page.evaluate(
                """(scale) => {
                    const s = document.querySelector('svg');
                    const [, , w, h] = s.getAttribute('viewBox').split(' ').map(Number);
                    s.setAttribute('width', w * scale);
                    s.setAttribute('height', h * scale);
                    return { w: w * scale, h: h * scale };
                }""",
                SCALE,
            )
            w, h = size["w"], size["h"]
            page.set_viewport_size({"width": w, "height": h})
            dest = OUT / (svg.stem + ".png")
            page.screenshot(
                path=str(dest),
                clip={"x": 0, "y": 0, "width": w, "height": h},
            )
            ok = w >= MEDIUM_MIN_WIDTH
            if not ok:
                failures.append(svg.stem)
            print(f"  {dest.name:22s} {w}x{h:<5d} {dest.stat().st_size/1024:5.1f} KB"
                  f"    {'ok' if ok else '<- under Medium 1192px minimum'}")
        browser.close()

    if failures:
        print(f"\ntoo narrow for Medium: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
