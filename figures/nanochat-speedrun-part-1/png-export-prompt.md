# Exporting figures to PNG for Medium

The site serves the SVGs in this folder directly. Medium cannot: it accepts
only JPG/JPEG/GIF/PNG, and wants images at least **1192px wide** before it
offers all placement options. The figures are authored at 1000px, so the
syndication copies in `png/` are rendered at **2x (2000px)**.

There are two ways to regenerate them.

## Option A - hand the prompt below to a coding agent

Preferred on a machine with VS Code and an agent that has integrated browser
tools. Needs no Playwright install and no 114 MB browser download, and avoids
corporate group-policy restrictions on running downloaded browser binaries.

## Option B - run the script

```bash
python build_all.py      # no dependencies, pure stdlib
pip install playwright
playwright install chromium      # skip if Edge or Chrome is installed
python export_png.py
```

`export_png.py` prefers a system browser (Edge, then Chrome) and only falls
back to Playwright's bundled Chromium, because on managed machines the
downloaded binary is often blocked by group policy.

---

## The prompt

Copy everything in the block below.

```markdown
# Task: export blog figure SVGs to PNG using the VS Code integrated browser

## Context
Repo: my GitHub Pages site (`anandnair2005.github.io`), branch `feature/pages-init`.

This folder (`figures/<post-slug>/`) holds one post's SVG figures plus the Python
generators that build them. The site serves SVG. I also syndicate the post to
Medium, which **does not accept SVG** - only JPG/JPEG/GIF/PNG - and which
wants images **at least 1192px wide** before it offers all image placement
options. The figures are authored at 1000px wide.

So: render each SVG at **2x scale** and save as PNG into
`png/`, overwriting what's there.

The six figures are:
repo-map, depth-cascade, sssl-window, precision-map, optimizer-step, bestfit-packing

## Do this
1. Run `python build_all.py` first so the SVGs are current. It needs
   **no dependencies** - pure Python 3 stdlib, no venv, no pip install.
2. Make sure `png/` exists (create it from the shell, not the browser).
3. Use the **integrated browser / Playwright-style tools** (e.g.
   `openBrowserPage` then `runPlaywrightCode`) to render and screenshot each
   figure at 2x.
4. `export-png.html` already exists in the repo for exactly this. Open
   it and it exposes `window.renderAt2x(name)` which fetches `<name>.svg`,
   inlines it, sets width/height to 2x, hides its own toolbar, and returns
   `{w, h, x, y}`. Then set the viewport to `{w, h}` and screenshot with
   `clip: {x, y, width: w, height: h}` using the returned `x/y`.

## Five things that will waste your time if you don't know them

1. **Clip to the returned x/y, never a hardcoded 0,0.** The harness has a
   toolbar laid out above the figure. If it is visible, the SVG starts 38px
   down the page, so a clip at 0,0 captures the toolbar and silently cuts 38px
   off the bottom of the figure. The PNG still has the exact right
   dimensions, so this passes a size check and looks fine unless you inspect
   the pixels. `renderAt2x` now hides the toolbar, but clip to the returned
   coordinates anyway.

2. **Do NOT navigate directly to a `.svg` file.** An SVG document has no
   `document.body`, so any script touching `document.body.style` dies with
   `Cannot read properties of null (reading 'style')`. Always host the SVG
   markup inside an HTML page - which is what `export-png.html` does.

3. **No Node APIs inside the browser-eval tool.** `require` is not defined, so
   you cannot use `fs` to mkdir or stat files from inside `runPlaywrightCode`.
   Create directories and check file sizes from the shell instead.

4. **Get a real page ID first.** Call `openBrowserPage` and use the ID it
   returns. Reusing a guessed or stale ID fails with `Page ... not found`.
   Viewport size can also silently revert; set it *after* `goto`, then assert
   the rendered width before screenshotting.

5. **If `fetch()` of the local `.svg` is blocked** by file:// CORS on your
   host, fall back to: read the SVG text with your file-read tool, then
   `page.setContent()` with the markup wrapped in
   `<!doctype html><meta charset='utf-8'><style>html,body{margin:0;padding:0}svg{display:block}</style>` + markup.

## Verification - I cannot view images (vision is disabled)
Verify programmatically only. Do not ask me to look at anything.

- Confirm each PNG's real dimensions by parsing the **IHDR** header: width and
  height are big-endian uint32 at byte offsets 16 and 20. Every figure must be
  exactly 2000px wide (>= 1192 for Medium).
- Confirm **colour type** at byte offset 25 is `2` (opaque RGB). If it is `6`
  (RGBA) the background may be transparent, which would make light text
  invisible on Medium's white page - that's a bug, fix it.
- Confirm all six files exist and are non-trivial in size (tens of KB).
- Optionally use the accessibility snapshot to confirm expected label text is
  present in the rendered SVG.

Report a table of filename, dimensions, file size, and colour type.
```

---

## Expected result

Six PNGs in `png/`, all exactly 2000px wide:

| Figure | Dimensions |
| --- | --- |
| repo-map | 2000 x 1024 |
| depth-cascade | 2000 x 1052 |
| sssl-window | 2000 x 1256 |
| precision-map | 2000 x 932 |
| optimizer-step | 2000 x 852 |
| bestfit-packing | 2000 x 820 |

**File sizes will vary by browser and that is fine.** VS Code's bundled
Chromium produced roughly 27 KB per figure; system Edge produced roughly 70 KB
from identical SVGs. Same pixels, different PNG encoder settings. Size is not
a correctness signal, which is why the prompt asks for header parsing instead.

The check that matters is colour type `2`. Colour type `6` means an alpha
channel, which risks a transparent background - light text on a transparent
background disappears against Medium's white page.
