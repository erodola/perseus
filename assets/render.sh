#!/bin/bash
# Render assets/<name>.tex to <name>-{light,dark}.png (3200 px wide); a multi-page file (a `% pages: a b ...` line, one
# name per page) to <name>-<page>-{light,dark}.png. Default name: results-table.
#   TECTONIC=<tectonic> PDFPY=<python with pymupdf> bash assets/render.sh [name ...]
set -euo pipefail
cd "$(dirname "$0")"
T="${TECTONIC:-tectonic}"; PY="${PDFPY:-python}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for name in "${@:-results-table}"; do
    pages=$(sed -n 's/^% pages: //p' "$name.tex")
    cp "$name.tex" "$tmp/light.tex"
    { echo '\def\darkmode{}'; cat "$name.tex"; } > "$tmp/dark.tex"
    for m in light dark; do
        "$T" --chatter minimal --outdir "$tmp" "$tmp/$m.tex"
        "$PY" - "$tmp/$m.pdf" "$name" "$m" $pages <<'PY'
import sys, pymupdf
pdf, name, mode, pages = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
for i, page in enumerate(pymupdf.open(pdf)):
    out = f"{name}-{pages[i]}-{mode}.png" if pages else f"{name}-{mode}.png"
    page.get_pixmap(matrix=pymupdf.Matrix(3200 / page.rect.width, 3200 / page.rect.width)).save(out)
PY
    done
done
