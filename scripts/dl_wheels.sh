#!/usr/bin/env bash
# Pre-download the large ML wheels with curl -C - (resumable), because the build proxy
# drops connections before pip can finish big wheels (IncompleteRead ~12MiB). curl resumes
# across each drop; pip cannot. Resulting ./wheels is consumed by the Dockerfile via
# --find-links so the build doesn't re-fetch these from the flaky proxy.
set -u
cd "$(dirname "$0")/.."
mkdir -p wheels
cd wheels

declare -a URLS=(
  "https://files.pythonhosted.org/packages/58/fe/334225e6330e672b36aef23d77451fa906ea12881570c08638a91331a212/torch-2.9.0-cp311-cp311-manylinux_2_28_aarch64.whl"
  "https://files.pythonhosted.org/packages/df/91/ea9a22a34ca108e70d9752097228091cccdb28d20af7657653cfd9dc6508/torch_npu-2.9.0-cp311-cp311-manylinux_2_28_aarch64.whl"
  "https://files.pythonhosted.org/packages/79/f9/b6f60f978397c616fd8dacca2305759fe4f80d397b20ef72534803244bd5/triton-3.5.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
  "https://files.pythonhosted.org/packages/09/01/dafaeccae2cffddd7cf0c7abdeec650b430ecda8fb6ca349d29fea118a6f/triton_ascend-3.2.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
)

# Resolve latest cp311/aarch64 (or py3-any) wheels for the rest dynamically.
for pkg in numpy pandas pyyaml; do
  u=$(curl -sS -m 30 "https://pypi.org/pypi/$pkg/json" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
cands = [x['url'] for x in d['urls'] if 'cp311' in x['filename'] and ('aarch64' in x['filename'] or 'any' in x['filename'])]
print(cands[0] if cands else '')
" 2>/dev/null)
  [ -n "$u" ] && URLS+=("$u") || echo "WARN: no URL resolved for $pkg"
done

for url in "${URLS[@]}"; do
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
