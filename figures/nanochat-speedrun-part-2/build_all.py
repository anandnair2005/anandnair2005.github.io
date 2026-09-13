"""
Rebuild the figures for this post, and only this post.

    python build_all.py

Needs Python 3 and nothing else - no venv, no pip install.

Every generator in this folder asserts its numbers against the real run log
before drawing, so a figure fails loudly rather than quietly showing something
wrong. The shared implementation lives in figures/figtools.py.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # figures/, for figlib and figtools

from figtools import build_dir  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(0 if build_dir(HERE) else 1)
