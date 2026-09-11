#!/usr/bin/env bash
# Pre-download the pinned ML wheels with curl -C - (resumable), for networks that
# drop transfers before pip can finish (IncompleteRead). Versions are read from
# pyproject.toml [project.optional-dependencies].npu — the same declaration the
# install paths use, so this file cannot drift.
#
# Resulting ./wheels can be fed to pip/docker builds on an offline host, e.g.:
#   pip install --find-links wheels -e ".[npu]"
set -u
cd "$(dirname "$0")/.."
mkdir -p wheels
cd wheels

# Resolve each [npu] pin (== or bare name) to its cp311 aarch64 (or py3-any)
# wheel URL via the PyPI JSON API.
URLS=$(
  python3 - <<'PY'
import json, urllib.request

import tomllib

with open("../pyproject.toml", "rb") as f:
    pins = tomllib.load(f)["project"]["optional-dependencies"]["npu"]

for pin in pins:
    name, _, version = pin.partition("==")
    if not version:  # unpinned (pandas, pyyaml): take the latest
        meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json"))
        files = meta["urls"]
    else:
        meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json"))
        files = meta["urls"]
    wheel = next(
        (f["url"] for f in files
         if "cp311" in f["filename"]
         and ("aarch64" in f["filename"] or f["filename"].endswith("any.whl"))),
        None,
    )
    print(wheel if wheel else f"WARN no cp311/aarch64 wheel for {pin}")
PY
)

for url in ${URLS}; do
  [[ "$url" == WARN* ]] && { echo "$url"; continue; }
  f=$(basename "$url")
  echo "=== $f ==="
  for i in $(seq 1 200); do
    if curl -fsSL -C - --retry 10 --retry-delay 2 -m 600 -o "$f" "$url"; then
      echo "OK $f ($(stat -c%s "$f") bytes)"
      break
    fi
    echo "  retry $i: $(stat -c%s "$f" 2>/dev/null || echo 0) bytes so far"
    sleep 1
  done
done

echo "=== wheels/ contents ==="
ls -l
