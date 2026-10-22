#!/bin/bash
# Pack the recorded graphs that ship as a release artifact (graphs/RELEASE.txt, one directory name per line) into
# out/perseus-graphs.tar.zst and print its sha256: the file to attach to a release.
#   tools/release_graphs.sh            # pack
#   tar --zstd -xf perseus-graphs.tar.zst -C .    # unpack, from the repository root
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mapfile -t names < <(sed -n 's/^\([a-z0-9_]\+\)\b.*/\1/p' graphs/RELEASE.txt)
for n in "${names[@]}"; do [ -e "graphs/$n" ] || { echo "missing graphs/$n" >&2; exit 1; }; done
mkdir -p out
tar --zstd -cf out/perseus-graphs.tar.zst "${names[@]/#/graphs/}"
ls -l out/perseus-graphs.tar.zst
sha256sum out/perseus-graphs.tar.zst
