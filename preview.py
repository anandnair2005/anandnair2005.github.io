"""
Render a post to a standalone HTML preview, for checking layout locally.

    python preview.py                     # newest post
    python preview.py _posts/2026-08-10-nanochat-speedrun-part-1.md
    python preview.py --check             # verify the renderer still works

Writes preview-<slug>.html next to this script. Needs Python 3 and nothing
else. The output is gitignored: it is a throwaway view, never a source.

Why this exists
---------------
Ruby will not install on this machine, so there is no local `jekyll serve`.
This reproduces the parts of the page that determine layout - the real
stylesheet, the real header/byline/footer structure, the real reading column -
so figure sizes, table alignment and vertical rhythm can be measured without
pushing first.

What it is not
--------------
It is not kramdown. It handles the subset of Markdown this blog actually uses
and ignores the rest. Treat it as a layout check, not a rendering check: if
the preview and the live site disagree, the live site is right.

Deliberately generated rather than hand-written, so it can never drift from
the post the way a copied HTML file does.
"""

import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POSTS = os.path.join(HERE, "_posts")

# Mirrors _config.yml. Only what the preview actually shows.
SITE = {
    "title": "Anand Nair",
    "author": "Anand Nair",
    "github_username": "anandnair2005",
    "medium_publication": "https://medium.com/indistinguishable-from-magic-ai",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------- front matter

def split_front_matter(text):
    if not text.startswith("---"):
        return {}, text
    end = text.index("\n---", 3)
    raw, body = text[3:end], text[end + 4:]
    meta, key = {}, None
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            meta[key] = value.strip('"') if value else []
        elif line.lstrip().startswith("-") and key:
            if not isinstance(meta.get(key), list):
                meta[key] = []
            meta[key].append(line.lstrip()[1:].strip().strip('"'))
    return meta, body


# ------------------------------------------------------------------- inline

def inline(s):
    """Span-level markdown. Code spans are protected from other rules."""
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
               lambda m: f'<img src="{m.group(2).lstrip("/")}" alt="{m.group(1)}">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![\*])", r"<em>\1</em>", s)

    def pop(m):
        return f"<code>{html.escape(spans[int(m.group(1))])}</code>"

    return re.sub(r"\x00(\d+)\x00", pop, s)


def slugify(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&[a-z]+;", " ", s)
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# -------------------------------------------------------------------- blocks

def render(body):
    lines = body.split("\n")
    out, i = [], 0
    pending_ial = None

    def take_ial():
        nonlocal pending_ial
        v, pending_ial = pending_ial, None
        return v

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # kramdown inline attribute list: applies to the block just emitted
        m = re.match(r"^\{:\s*([^}]+)\}$", stripped)
        if m:
            attr = m.group(1).strip()
            if out and attr.startswith("."):
                cls = attr[1:]
                for j in range(len(out) - 1, -1, -1):
                    tag = re.match(r"^<(table|figure|p|blockquote)", out[j])
                    if tag:
                        out[j] = out[j].replace(f"<{tag.group(1)}",
                                                f'<{tag.group(1)} class="{cls}"', 1)
                        break
            elif attr.startswith("#"):
                pending_ial = attr[1:]
                # heading was already emitted; retro-fit its id
                for j in range(len(out) - 1, -1, -1):
                    if out[j].startswith("<h"):
                        out[j] = re.sub(r'id="[^"]*"', f'id="{attr[1:]}"', out[j], count=1)
                        break
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{lang}"' if lang else ""
            out.append(f'<div class="highlight"><pre><code{cls}>'
                       f'{html.escape(chr(10).join(buf))}</code></pre></div>')
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level, text = len(m.group(1)), inline(m.group(2))
            out.append(f'<h{level} id="{slugify(text)}">{text}</h{level}>')
            i += 1
            continue

        # table
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            head, body_rows = cells[0], cells[2:]
            t = "<table><thead><tr>"
            t += "".join(f"<th>{inline(c)}</th>" for c in head)
            t += "</tr></thead><tbody>"
            for r in body_rows:
                t += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</tbody></table>")
            continue

        # blockquote
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(buf))}</p></blockquote>")
            continue

        # list
        if re.match(r"^[-*]\s+|^\d+\.\s+", stripped):
            ordered = bool(re.match(r"^\d+\.", stripped))
            tag = "ol" if ordered else "ul"
            buf = []
            while i < len(lines) and re.match(r"^\s*([-*]\s+|\d+\.\s+)", lines[i]):
                buf.append(re.sub(r"^\s*([-*]\s+|\d+\.\s+)", "", lines[i]))
                i += 1
            items = "".join(f"<li>{inline(b)}</li>" for b in buf)
            out.append(f"<{tag}>{items}</{tag}>")
            continue

        # A standalone image. kramdown wraps this in a paragraph rather than a
        # <figure>, and the preview must match: emitting <figure> here once
        # hid a CSS bug that only affected the real, <p>-wrapped markup.
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m:
            alt, src = m.group(1), m.group(2).lstrip("/")
            out.append(f'<p><img src="{src}" alt="{html.escape(alt)}"></p>')
            i += 1
            continue

        # paragraph
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(\||>|#{1,6}\s|```|\{:|[-*]\s+|\d+\.\s+)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            out.append(f"<p>{inline(' '.join(buf))}</p>")
        else:
            i += 1

    return "\n".join(out)


def read_time(rendered):
    """Mirrors the layout: strip_html | number_of_words | divided_by 265 | plus 1."""
    text = re.sub(r"<[^>]+>", " ", rendered)
    return len(text.split()) // 265 + 1


# ---------------------------------------------------------------------- page

PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} &middot; PREVIEW</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
      /* preview-only banner, not part of the site */
      .preview-note {{
        position: fixed; right: 12px; bottom: 12px; z-index: 99;
        background: #242424; color: #fff; border-radius: 6px;
        padding: 6px 10px; font: 12px/1.4 system-ui, sans-serif; opacity: .85;
      }}
    </style>
  </head>
  <body>
    <header class="site-head">
      <div class="wrap">
        <a class="brand" href="#">
          <img src="assets/img/avatar.png" alt="">
          <span>{site_title}</span>
        </a>
        <nav class="nav">
          <a href="#">Archive</a><a href="#">Topics</a><a href="#">About</a>
        </nav>
      </div>
    </header>
    <main>
      <article class="post">
        <div class="post-head">
          {series}
          <h1>{title}</h1>
          {subtitle}
          <div class="byline">
            <img src="assets/img/avatar.png" alt="">
            <div>
              <div>{author}</div>
              <div><time>{date}</time> &middot; {minutes} min read</div>
            </div>
          </div>
        </div>
    
        <div class="post-body">
{body}
        </div>
    
        <div class="post-foot">
          {tags}
        </div>
      </article>
    </main>
    <footer class="site-foot">
      <div class="wrap">
        &copy; 2026 {author} &middot;
        <a href="#">RSS</a> &middot; <a href="#">GitHub</a> &middot;
        <a href="#">Medium</a>
      </div>
    </footer>
    <div class="preview-note">local preview &middot; not kramdown</div>
  </body>
</html>
"""


def build(path):
    with open(path, encoding="utf-8") as fh:
        meta, body = split_front_matter(fh.read())

    rendered = render(body)

    date = meta.get("date", "")
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(date))
    nice = f"{MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}" if m else str(date)

    series = ""
    if meta.get("series"):
        series = (f'<div class="series-flag">{meta["series"]} &middot; '
                  f'Part {meta.get("part", "1")}</div>')

    subtitle = ""
    if meta.get("subtitle"):
        subtitle = f'<p class="sub">{meta["subtitle"]}</p>'

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.strip("[]").split(",") if t.strip()]
    tag_html = "".join(f'<a class="pill" href="#">{t}</a>' for t in tags)

    page = PAGE.format(
        title=meta.get("title", "Untitled"),
        site_title=SITE["title"],
        author=SITE["author"],
        date=nice,
        minutes=read_time(rendered),
        series=series,
        subtitle=subtitle,
        body=rendered,
        tags=tag_html,
    )

    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", os.path.basename(path))[:-3]
    out = os.path.join(HERE, f"preview-{slug}.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out, rendered

# ----------------------------------------------------------------------------- self-check

def check(path):
    """Assert the renderer still behaves, and report what failed.

    This file is edited by hand and has twice come back from a round trip with
    its indentation shifted, which silently disables whole block handlers. Each
    assertion below corresponds to a way it has actually broken, so run this
    after any change you did not make yourself.
    """
    with open(path, encoding="utf-8") as fh:
        meta, body = split_front_matter(fh.read())
    r = render(body)

    src_images = len(re.findall(r"^!\[", body, re.M))
    src_tables = len(re.findall(r"^\{:\s*\.wide\}", body, re.M))
    src_caps = len(re.findall(r"^\s*\{:\s*\.caption\}", body, re.M))

    src_h2 = len(re.findall(r"^## ", body, re.M))
    src_fences = len(re.findall(r"^```", body, re.M)) // 2
    src_quotes = len(re.findall(r"^>", body, re.M))

    # Paragraphs that open with an inline code span or an emphasised phrase.
    # Each of these has been broken by a one-character change to a
    # block-detection regex, and none shows up in a simple count, so compare
    # the rendered text against the source line instead. List items are
    # excluded: they are block-level and legitimately render as <li>.
    def first_words(line):
        plain = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", r"\1", line)
        plain = re.sub(r"[`*_]", "", plain)
        return " ".join(plain.split()[:6])

    plain_render = re.sub(r"<[^>]+>", "", r)
    truncated = []
    in_fence = False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or len(line.split()) < 8:
            continue
        if re.match(r"^\s*(`|\*\*)", line):
            head = first_words(line)
            if head and head not in plain_render:
                truncated.append(head)

    results = [
        ("front matter parsed", bool(meta.get("title"))),
        (f"images rendered as <p><img ({src_images})",
         r.count("<p><img") == src_images and "<figure" not in r),
        (f"tables rendered ({src_tables})", r.count("<table") == src_tables),
        (f"table .wide class applied ({src_tables})",
         r.count('<table class="wide"') == src_tables),
        (f"caption .caption class applied ({src_caps})",
         r.count('<p class="caption"') == src_caps),
        ("every image followed by a caption",
         src_caps == src_images),
        ("no IAL text leaked into output", "{:" not in r),
        (f"h2 headings ({src_h2})", len(re.findall(r"<h2 ", r)) == src_h2),
        ("h2 headings carry ids", all(
            'id="' in m for m in re.findall(r"<h2[^>]*>", r))),
        ("appendix ids from IAL", 'id="appendix-a"' in r and 'id="appendix-b"' in r),
        (f"code blocks ({src_fences})", r.count('<div class="highlight">') == src_fences),
        ("blockquotes rendered", src_quotes == 0 or "<blockquote" in r),
        ("code spans rendered", "<code>" in r),
        ("links rendered", '<a href="http' in r),
        ("bold rendered", "<strong>" in r),
        ("no unclosed paragraphs",
         len(re.findall(r"<p\b[^>]*>", r)) == r.count("</p>")),
        ("no raw markdown leaked", not re.search(r"^\s*\|", r, re.M)),
        ("entities left for the browser",
         "&mdash;" in r or "\u2014" in r),
        ("no paragraph truncated at an inline marker", not truncated),
    ]

    width = max(len(n) for n, _ in results)
    bad = 0
    for name, ok in results:
        if not ok:
            bad += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:{width}s}")
    print(f"\n  {len(results) - bad}/{len(results)} checks passed")
    return bad


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}

    if args:
        target = args[0]
    else:
        posts = sorted(f for f in os.listdir(POSTS) if f.endswith(".md"))
        if not posts:
            sys.exit("no posts found in _posts/")
        target = os.path.join(POSTS, posts[-1])

    if "--check" in flags:
        print(f"  checking against {os.path.basename(target)}\n")
        raise SystemExit(1 if check(target) else 0)

    out, rendered = build(target)
    print(f"  {os.path.basename(target)}")
    print(f"  -> {os.path.basename(out)}  ({os.path.getsize(out) / 1024:.1f} KB)")
    for pattern, label in (("<p><img", "images"), ("<table", "tables"),
                           ("<h2", "sections"), ('<div class="highlight"', "code blocks")):
        print(f"     {rendered.count(pattern):3d} {label}")
