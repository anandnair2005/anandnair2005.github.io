---
title: "Placeholder: first post in a series"
subtitle: "A stand-in article used to check the layout. Series navigation, tags, figure breakout and code blocks all appear below."
date: 2026-08-10
series: "A Placeholder Series"
part: 1
tags: [placeholder, layout]
image: /assets/img/placeholder-wide.png
---

This post exists only to exercise the design. Delete it once the real
articles land.

The paragraph you are reading is set in the reading measure, roughly 680
pixels, which is the same width Medium uses for body copy. Long-form
technical writing is easier to follow at this width than at full page
width, so the text column stays narrow even on a wide monitor.

## A second-level heading

Headings switch to the sans-serif face while body copy stays serif. That
contrast is the main thing that makes a long article feel like an article
rather than documentation.

Inline code such as `row_capacity = T + 1` sits inside a paragraph, and a
fenced block gets its own panel:

```python
for i, doc in enumerate(doc_buffer):
    doc_len = len(doc)
    if doc_len <= remaining and doc_len > best_len:
        best_idx, best_len = i, doc_len
```

### A third-level heading

> A blockquote, for pulling out a claim worth pausing on.

Figures deliberately break out past the text column, because the generated
diagrams are 1400 pixels wide and would be unreadable if squeezed into the
reading measure:

![Placeholder figure](/assets/img/placeholder-wide.png)

Tables break out the same way, since they tend to be wide:

| Column | Meaning | Value |
| --- | --- | --- |
| `depth` | main size dial | 24 |
| `model_dim` | rounded width | 1536 |
| `num_heads` | attention heads | 12 |

A list, to confirm spacing:

- First item
- Second item, which runs on a little longer so it wraps onto a second
  line and shows how leading behaves inside list items
- Third item

That covers every element the real posts use.
