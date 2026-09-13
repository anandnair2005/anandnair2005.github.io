---
layout: page
title: Topics
permalink: /tags/
---

{%- comment -%}
One page listing every tag with its posts, anchored by tag slug.
This is why no per-tag files are needed: `jekyll-archives` is not on
GitHub Pages' plugin allowlist, and this gives clickable topics with
zero per-tag maintenance. Adding a new tag to a post is enough.
{%- endcomment -%}

{%- assign tags = site.tags | sort %}

{%- if tags.size > 0 %}
<p>
  {%- for t in tags %}
  <a class="pill" href="#{{ t[0] | slugify }}">{{ t[0] }} ({{ t[1] | size }})</a>
  {%- endfor %}
</p>

{%- for t in tags %}
<h2 id="{{ t[0] | slugify }}">{{ t[0] }}</h2>
<ul>
  {%- for p in t[1] %}
  <li>
    <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
    <span class="muted">{{ p.date | date: "%b %Y" }}</span>
  </li>
  {%- endfor %}
</ul>
{%- endfor %}

{%- else %}
<p class="muted">No topics yet.</p>
{%- endif %}
