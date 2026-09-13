"""
Shared implementation behind each post's build_all.py and export_png.py.

Figures live per post, under figures/<post-slug>/, because a post's figures
are only ever rebuilt with that post. The entry points sit in the post folder
so you can rebuild one post without touching the other nine, but the logic
lives here so a fix lands once rather than being copy-pasted into every post.

Nothing here is post-specific: both functions work off the directory handed
to them.
"""

import importlib
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCALE = 2                # PNG export scale
MEDIUM_MIN_WIDTH = 1192   # Medium wants at least this before offering all layouts

# Entry points and tooling, not figure generators.
NOT_GENERATORS = {"build_all", "export_png", "figlib", "figtools"}


def generators(d):
    """Generator modules in a post folder, in a stable order."""
    return sorted(
        f[:-3] for f in os.listdir(d)
        if f.endswith(".py") and f[:-3] not in NOT_GENERATORS
    )


def _load(d, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_dir(d):
    """Rebuild every SVG in one post folder. Returns the number built."""
    import figlib
    figlib.output_to(d)
    names = generators(d)
    if not names:
        print(f"no figure generators found in {d}", file=sys.stderr)
        return 0

    print(f"building {len(names)} figures in {os.path.basename(d)}")
    total = 0
    for name in names:
        mod = _load(d, name)
        result = mod.build()
        fig = result[0] if isinstance(result, tuple) else result
        out = name.replace("_", "-") + ".svg"
        _, size = fig.save(out)
        total += size
        print(f"  {out:22s} {size/1024:5.1f} KB")
    print(f"  {'TOTAL':22s} {total/1024:5.1f} KB")
    return len(names)


def export_dir(d):
    """Render every SVG in a post folder to PNG at 2x, for syndication."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. It is needed only for PNG export;\n"
            "building the SVGs needs nothing.\n\n"
            "    pip install playwright\n"
            "    playwright install chromium",
            file=sys.stderr,
        )
        return 1

    svgs = sorted([f for f in os.listdir(d) if f.endswith(".svg")])
    if not svgs:
        print("no SVGs found - run build_all.py first", file=sys.stderr)
        return 1

    out = os.path.join(d, "png")
    os.makedirs(out, exist_ok=True)
    print(f"exporting {len(svgs)} figures at {SCALE}x")
    narrow = []

    with sync_playwright() as pw:
        browser = _launch(pw)
        page = browser.new_page()
        for name in svgs:
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                markup = fh.read()
            # An .svg opened directly is an SVG document with no document.body,
            # so host the markup inside a real HTML page instead.
            page.set_content(
                '<!doctype html><meta charset="utf-8">'
                '<style>html,body{margin:0;padding:0}svg{display:block}</style>'
                + markup
            )
            size = page.evaluate(
                """(scale) => {
                    const s = document.querySelector('svg');
                    const [ , , w, h] = s.getAttribute('viewBox').split(' ').map(Number);
                    s.setAttribute('width', w * scale);
                    s.setAttribute('height', h * scale);
                    return { w: w * scale, h: h * scale };
                }""",
                SCALE,
            )
            w, h = size["w"], size["h"]
            page.set_viewport_size({"width": w, "height": h})
            dest = os.path.join(out, name[:-4] + ".png")
            page.screenshot(path=dest, clip={"x": 0, "y": 0, "width": w, "height": h})
            ok = w >= MEDIUM_MIN_WIDTH
            if not ok:
                narrow.append(name)
            print(f"  {os.path.basename(dest):24s} {w}x{h:<5d} "
                  f"{os.path.getsize(dest)/1024:6.1f} KB"
                  f"{'' if ok else ' <- under Medium 1192px minimum'}")
        browser.close()

    if narrow:
        print(f"\ntoo narrow for Medium: {', '.join(narrow)}", file=sys.stderr)
        return 1
    return 0


# Tried in order. None = Playwright's own bundled Chromium, which on a managed
# machine is often blocked by group policy even though it downloads fine.
CHANNELS = ["msedge", "chrome", None]


def _launch(pw):
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
