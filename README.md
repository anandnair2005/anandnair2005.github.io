# anandnair.dev

Source for [anandnair.dev](https://anandnair.dev) – writing about AI and the
software systems underneath it. Built with Jekyll and served by GitHub Pages.

Posts are also syndicated to
[Medium](https://medium.com/indistinguishable-from-magic-ai); this site is the
canonical copy.

## Layout

```
_posts/                     one Markdown file per post
_layouts/  _includes/       page structure
assets/css/style.css        all styling, no framework
figures/
  figlib.py                 shared drawing primitives and palette
  figtools.py               shared build and PNG-export logic
  <post-slug>/              one folder per post
    build_all.py            rebuild that post's figures
    export_png.py           render them to PNG for Medium
    *.py                    one generator per figure
    *.svg                   what the site serves
    png/                    raster copies, syndication only
preview.py                  render a post to a local HTML preview
```

## Figures

Diagrams are generated, not drawn. Each generator derives its numbers and then
**asserts them against the original run log** before drawing anything, so a
figure fails loudly rather than quietly showing something wrong. The generators
are published alongside the images so any number in a diagram can be traced
back to the run that produced it.

Rebuild one post's figures:

```bash
cd figures/<post-slug>
python build_all.py
```

Python 3, standard library only – no virtualenv, no dependencies.

PNG export is separate. It is needed only for Medium, which does not accept
SVG, and it requires Playwright:

```bash
pip install playwright
playwright install chromium     # skip if Edge or Chrome is installed
python export_png.py
```

## Previewing a post

There is no local Jekyll here, so `preview.py` renders a post to a standalone
HTML file using the real stylesheet and page structure:

```bash
python preview.py                    # newest post
python preview.py _posts/<file>.md   # a specific one
python preview.py --check            # verify the renderer itself
```

It is a layout check, not a rendering check: it handles the Markdown subset
this blog uses and ignores the rest. Where it disagrees with the published
site, the site is right.

## Licence

Post text and figures © Anand Nair. Code in this repository is MIT.