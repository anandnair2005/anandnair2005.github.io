---
title: "Placeholder: second post in a series"
subtitle: "The sibling of part one. Its only job is to prove that series navigation links resolve in both directions."
date: 2026-08-16
series: "A Placeholder Series"
part: 2
tags: [placeholder, layout]
image: /assets/img/placeholder-wide.png
---

The green strip above the title should read "Part 2 of 2" and link back to
part one. Nothing in this file names its sibling: the include matches on
the `series` field, so adding a part three would wire itself up.

## Why this matters for the real posts

The nanochat articles are a two-part series today, but later writing may be
unrelated to nanochat entirely. Because the series relationship lives in
front matter rather than in a folder name, a post about streaming
multiprocessors just omits the `series` field and sits on its own.
