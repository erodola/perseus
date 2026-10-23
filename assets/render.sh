#!/bin/bash
# Render assets/<name>.tex to <name>-{light,dark}.png ($WIDTH px wide, default 4800); a multi-page file (a `% pages:
# a b ...` line, one name per page) to <name>-<page>-{light,dark}.png. Default name: results-table.
#   TECTONIC=<tectonic> PDFPY=<python with pymupdf> bash assets/render.sh [name ...]
set -euo pipefail
cd "$(dirname "$0")"
T="${TECTONIC:-tectonic}"; PY="${PDFPY:-python}"; WIDTH="${WIDTH:-4800}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for name in "${@:-results-table}"; do
    pages=$(sed -n 's/^% pages: //p' "$name.tex")
    cp "$name.tex" "$tmp/light.tex"
    { echo '\def\darkmode{}'; cat "$name.tex"; } > "$tmp/dark.tex"
    for m in light dark; do
        "$T" --chatter minimal --outdir "$tmp" "$tmp/$m.tex"
        "$PY" - "$tmp/$m.pdf" "$name" "$m" "$WIDTH" $pages <<'PY'
import sys, pymupdf
pymupdf.mupdf.fz_set_graphics_aa_level(0)   # fills and rules pixel-exact (no seams between table cells)
pymupdf.mupdf.fz_set_text_aa_level(8)       # text smooth (the call above also turns text antialiasing off)
pdf, name, mode, width, pages = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
for i, page in enumerate(pymupdf.open(pdf)):
    out = f"{name}-{pages[i]}-{mode}.png" if pages else f"{name}-{mode}.png"
    page.get_pixmap(matrix=pymupdf.Matrix(width / page.rect.width, width / page.rect.width)).save(out)
PY
    done
done
