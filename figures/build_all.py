"""
Regenerate every figure for the blog.

    python figures/build_all.py

Requires Python 3 only - no third-party packages.

Each generator asserts its numbers against the real run log before drawing,
so a figure fails loudly rather than quietly showing something wrong.

SVG is what the site serves. Medium does not accept SVG, so PNG copies for
syndication live in figures/png/ - see figures/export-png.html to rebuild them.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODULES = [
    "repo_map",
    "depth_cascade",
    "sssl_window",
    "precision_map",
    "optimizer_step",
    "bestfit_packing",
]

if __name__ == "__main__":
    print("building figures")
    for name in MODULES:
        m = importlib.import_module(name)
        result = m.build()
        fig = result[0] if isinstance(result, tuple) else result
        out = name.replace("_", "-") + ".svg"
        path, size = fig.save(out)
        print(f"  {out:22s} {size/1024:5.1f} KB")
    total = sum(os.path.getsize(os.path.join(HERE, f))
                for f in os.listdir(HERE) if f.endswith(".svg"))
    print(f"  {'TOTAL':22s} {total/1024:5.1f} KB")
