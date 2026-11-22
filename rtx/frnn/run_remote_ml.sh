#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$HERE/data/multires_fair_remote_$STAMP.txt"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 1; fi
SSHO=(-o BatchMode=yes -o ConnectTimeout=20)
R(){ ssh "${SSHO[@]}" bpgcjump "ssh -i ~/.ssh/hri_rsa_1 -o BatchMode=yes -o ConnectTimeout=20 HRI@10.1.19.30 '$1'"; }

D=$(R 'd=~/sph_dir1_20261120; k=0; while [ -e "$d" ]; do k=$((k+1)); d=~/sph_dir1_20261120_$k; done; mkdir -p "$d" && echo "$d"')
echo "remote dir: $D"

tar -C "$ROOT" -czf - rtx/frnn/frnn_ml.cu rtx/frnn/frnnPrograms.cu rtx/frnn/frnnShared.h rtx/frnn/build_frnn_ml.sh rtx/frnn/remote_sweep_ml.sh rtx/third_party/optix/include \
  | ssh "${SSHO[@]}" bpgcjump "ssh -i ~/.ssh/hri_rsa_1 -o BatchMode=yes -o ConnectTimeout=20 HRI@10.1.19.30 'tar -xzf - -C $D'"

R "cd $D/rtx/frnn && timeout 3300 flock -w 900 /tmp/sph_bench.lock timeout 2400 bash remote_sweep_ml.sh > results.txt 2>&1; echo rc=\$?" || true
R "cat $D/rtx/frnn/results.txt" > "$OUT"
echo "results: $OUT"
grep -E "^(INFO|FRNNML|RUNFAIL|BUILD)" "$OUT" | cut -c1-200
