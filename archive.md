---
layout: page
title: Archive
permalink: /archive/
---

{%- assign all_series = site.posts | map: "series" | compact | uniq %}

{%- for s in all_series %}
<h2>{{ s }}</h2>
<ul>
  {%- assign parts = site.posts | where: "series", s | sort: "part" %}
  {%- for p in parts %}
  <li>
    <strong>Part {{ p.part }}</strong> &mdash;
    <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
    <span class="muted">{{ p.date | date: "%b %Y" }}</span>
  </li>
  {%- endfor %}
</ul>
{%- endfor %}

{%- assign standalone = site.posts | where_exp: "p", "p.series == nil" %}
{%- if standalone.size > 0 %}
<h2>Other writing</h2>
<ul>
  {%- for p in standalone %}
  <li>
    <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
    <span class="muted">{{ p.date | date: "%b %Y" }}</span>
  </li>
  {%- endfor %}
</ul>
{%- endif %}

{%- if site.posts.size == 0 %}
<p class="muted">No posts yet.</p>
{%- endif %}
