"""
Export this post's figures to PNG for Medium syndication.

    pip install playwright
    playwright install chromium    # skip if Edge or Chrome is installed
    python export_png.py

The site itself serves SVG and needs none of this. Medium accepts only
.JPG / .JPEG / .GIF / .PNG - not SVG - and wants images at least 1192px wide
before it offers all placement options, so everything is rendered at 2x.

The shared implementation lives in figures/figtools.py; see
png-export-prompt.md in this folder for the agent-driven alternative.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # figures/, for figtools

from figtools import export_dir  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(export_dir(HERE))
