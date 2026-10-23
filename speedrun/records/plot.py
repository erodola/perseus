#!/usr/bin/env python3
"""Draw the record progression of both tracks from records/progress.csv.

    python3 records/plot.py

Writes progress-light.svg and progress-dark.svg (the README's <picture>) and progress.html (the same
chart with hover details) next to this script. Standard library only.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS = [("prefill", "GPT-2 Prefill-128", "time to first token", "s"),
          ("decode", "GPT-2 Decode-8", "time per token", "s/token")]
# Orchard blues, as in the Perseus figures.
THEMES = {
    "light": dict(text="#24292f", muted="#59636e", grid="#d1d9e0", line="#3d5a80", bg="#ffffff"),
    "dark": dict(text="#e6edf3", muted="#9198a1", grid="#3d444d", line="#98c1d9", bg="#0d1117"),
}
W, PH, GAP = 900, 250, 16      # one full-width panel per track, stacked
H = 2 * PH + GAP
LEFT, RIGHT, TOP, BOTTOM = 52, 20, 52, 36
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["record"] = int(r["record"])
            r["date"] = dt.date.fromisoformat(r["date"])
            r["value"] = float(r["value"])
            rows.append(r)
    return rows


def log_span(values):
    lo = 10 ** math.floor(math.log10(min(values)))
    hi = 10 ** math.ceil(math.log10(max(values)))
    return lo, hi if hi > lo else lo * 10


def log_ticks(lo, hi):
    ticks, d = [], lo
    while d <= hi * 1.0001:
        ticks += [d * m for m in (1, 2, 5) if lo <= d * m <= hi * 1.0001]
        d *= 10
    return ticks


def months(d0, d1):
    m = dt.date(d0.year, d0.month, 1)
    while m <= d1:
        if m >= d0:
            yield m
        m = dt.date(m.year + m.month // 12, m.month % 12 + 1, 1)


def fmt(v):
    return f"{v:g}"


def panel(oy, track, title, quantity, unit, rows, d0, d1, a):
    pts = sorted((r for r in rows if r["track"] == track), key=lambda r: (r["date"], r["record"]))
    x0, x1, y0, y1 = LEFT, W - RIGHT, oy + TOP, oy + PH - BOTTOM
    out = [f'<text {a["t"]} x="{x0}" y="{oy + 22}" font-size="15" font-weight="600">{title}</text>',
           f'<text {a["m"]} x="{x0}" y="{oy + 40}" font-size="12">{quantity} ({unit}), log scale</text>']
    if not pts:
        out.append(f'<line {a["g"]} x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
        out.append(f'<text {a["m"]} x="{(x0 + x1) / 2:.0f}" y="{(y0 + y1) / 2:.0f}" font-size="13" '
                   f'text-anchor="middle">No records yet</text>')
        return out
    lo, hi = log_span([r["value"] for r in pts])

    def x(d):
        return x0 + (d - d0).days / max(1, (d1 - d0).days) * (x1 - x0)

    def y(v):
        return y0 + (math.log10(hi) - math.log10(v)) / (math.log10(hi) - math.log10(lo)) * (y1 - y0)

    for v in log_ticks(lo, hi):
        out.append(f'<line {a["g"]} x1="{x0}" x2="{x1}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text {a["m"]} x="{x0 - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end">{fmt(v)}</text>')
    for i, m in enumerate(months(d0, d1)):
        label = m.strftime("%b %Y" if i == 0 or m.month == 1 else "%b")
        out.append(f'<text {a["m"]} x="{x(m):.1f}" y="{y1 + 20}" font-size="11" text-anchor="middle">{label}</text>')
    out.append(f'<line {a["g"]} x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')

    path = f"M{x(pts[0]['date']):.1f},{y(pts[0]['value']):.1f}"
    for r in pts[1:]:
        path += f"H{x(r['date']):.1f}V{y(r['value']):.1f}"
    out.append(f'<path {a["ln"]} d="{path}"/>')
    out.append(f'<path {a["hold"]} d="M{x(pts[-1]["date"]):.1f},{y(pts[-1]["value"]):.1f}H{x1}"/>')

    for r in pts:
        cx, cy = x(r["date"]), y(r["value"])
        tip = {"title": f"#{r['record']} {r['label']}", "value": f"{fmt(r['value'])} {unit}",
               "date": r["date"].isoformat()}
        out.append(f'<circle {a["pt"]} cx="{cx:.1f}" cy="{cy:.1f}" r="5" tabindex="0" '
                   f'data-tip="{html.escape(json.dumps(tip))}"><title>{html.escape(tip["title"])}, '
                   f'{tip["value"]}, {tip["date"]}</title></circle>')
        out.append(f'<text {a["t"]} x="{cx + 10:.1f}" y="{cy - 9:.1f}" font-size="12">'
                   f'#{r["record"]} · {fmt(r["value"])} {unit}</text>')
    return out


def svg(rows, a):
    dates = [r["date"] for r in rows]
    d0 = d1 = None
    if dates:
        d0 = min(dates) - dt.timedelta(days=10)
        d1 = max(max(dates) + dt.timedelta(days=30), d0 + dt.timedelta(days=60))
    body = []
    for i, (track, title, quantity, unit) in enumerate(TRACKS):
        body += panel(i * (PH + GAP), track, title, quantity, unit, rows, d0, d1, a)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="{FONT}" role="img" aria-label="Record progression of both tracks">'
            + "".join(body) + "</svg>")


def paint(c):
    """Presentation attributes per role, so the SVG renders the same in any viewer."""
    line = f'stroke="{c["line"]}" stroke-width="2.5"'
    return {"t": f'fill="{c["text"]}"', "m": f'fill="{c["muted"]}"', "g": f'stroke="{c["grid"]}" stroke-width="1"',
            "ln": f'fill="none" {line}',
            "hold": f'fill="none" stroke="{c["line"]}" stroke-width="2" stroke-dasharray="4 4" opacity="0.55"',
            "pt": f'fill="{c["line"]}" {line}'}


CLASSES = {k: f'class="{v}"' for k, v in
           {"t": "t", "m": "m", "g": "g", "ln": "ln", "hold": "hold", "pt": "pt"}.items()}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Encrypted GPT-2 Speedrun records</title>
<style>
:root{{color-scheme:light dark;{light}}}
@media (prefers-color-scheme:dark){{:root{{{dark}}}}}
body{{margin:0;padding:24px max(16px,4vw);background:var(--bg);color:var(--text);font:14px/1.5 {font}}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:var(--muted);margin:0 0 16px}}
svg{{width:100%;max-width:{w}px;height:auto;overflow:visible}}
.t{{fill:var(--text)}}.m{{fill:var(--muted)}}.g{{stroke:var(--grid)}}
.ln{{fill:none;stroke:var(--line);stroke-width:2.5}}
.hold{{fill:none;stroke:var(--line);stroke-width:2;stroke-dasharray:4 4;opacity:.55}}
.pt{{fill:var(--line);stroke:var(--line);stroke-width:2.5;cursor:pointer}}
.pt:hover,.pt:focus{{stroke-width:4;outline:none}}
#tip{{position:fixed;pointer-events:none;background:var(--text);color:var(--bg);padding:6px 9px;border-radius:6px;
font-size:12.5px;line-height:1.4;max-width:280px}}
</style></head><body>
<h1>Encrypted GPT-2 Speedrun records</h1>
<p>Hover or focus a point for details.</p>
{svg}
<div id="tip" hidden></div>
<script>
const tip = document.getElementById("tip");
function show(e) {{
  const d = JSON.parse(e.target.dataset.tip), r = e.target.getBoundingClientRect();
  tip.innerHTML = "<b></b><br><span></span>";
  tip.querySelector("b").textContent = d.title;
  tip.querySelector("span").textContent = d.value + " \\u00b7 " + d.date;
  tip.hidden = false;
  tip.style.left = Math.min(r.right + 10, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = (r.top - tip.offsetHeight - 6) + "px";
}}
for (const p of document.querySelectorAll(".pt")) {{
  p.querySelector("title").remove();
  p.addEventListener("mouseenter", show); p.addEventListener("focus", show);
  p.addEventListener("mouseleave", () => tip.hidden = true); p.addEventListener("blur", () => tip.hidden = true);
}}
</script></body></html>
"""


def main():
    rows = load(os.path.join(HERE, "progress.csv"))
    for name, colors in THEMES.items():
        with open(os.path.join(HERE, f"progress-{name}.svg"), "w", encoding="utf-8") as f:
            f.write(svg(rows, paint(colors)) + "\n")
    tokens = {k: ";".join(f"--{n}:{v}" for n, v in c.items()) for k, c in THEMES.items()}
    with open(os.path.join(HERE, "progress.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.format(light=tokens["light"], dark=tokens["dark"], font=FONT, w=W, svg=svg(rows, CLASSES)))


if __name__ == "__main__":
    main()
