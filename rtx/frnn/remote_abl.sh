#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
HOST="${HOST:-pcproj@10.1.19.63}"
SSHO="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20"
TAG="${TAG:-20261120}"
SETS="${SETS:-all all2}"
CAP_S="${CAP_S:-3000}"
OUT="$HERE/data/abl_remote_ada_${TAG}.txt"
CLK="$HERE/data/abl_remote_ada_${TAG}_clocks.txt"
for f in "$OUT" "$CLK"; do
    if [ -e "$f" ]; then echo "refusing to overwrite $f" >&2; exit 1; fi
done
remote(){ ssh $SSHO "$HOST" "$1"; }

RD="sph_abl_$TAG"
i=1
while remote "test -e ~/$RD"; do i=$((i+1)); RD="sph_abl_${TAG}_$i"; done
echo "remote dir: ~/$RD"
tar czf - -C "$ROOT" rtx/frnn/abl.cu rtx/frnn/ablPrograms.cu rtx/frnn/ablShared.h rtx/frnn/build_abl.sh rtx/frnn/run_abl.sh rtx/third_party/optix/include \
  | ssh $SSHO "$HOST" "mkdir ~/$RD && tar xzf - -C ~/$RD"

PAYLOAD=$(cat <<EOF
set -uo pipefail
cd ~/$RD/rtx/frnn
CC=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
export ARCH=sm_\$CC CUDA_HOME=/usr/local/cuda
bash build_abl.sh > build.log 2>&1 || { echo BUILD_FAILED; tail -30 build.log; exit 1; }
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu --format=csv,noheader -lms 1000 > clocks.txt 2>&1 &
LP=\$!
timeout $CAP_S flock -w 600 /tmp/sph_bench.lock bash -c 'for s in $SETS; do bash run_abl.sh build_abl results_\$s.txt \$s; done' > run.log 2>&1
echo "rc=\$?"
kill \$LP
cat results_*.txt > results.txt
echo RUN_DONE
EOF
)
echo "$PAYLOAD" | ssh $SSHO "$HOST" "bash -s"
remote "cat ~/$RD/rtx/frnn/results.txt" > "$OUT"
remote "cat ~/$RD/rtx/frnn/clocks.txt" > "$CLK"
echo "pulled: $OUT ($(grep -c '^ABL' "$OUT") rows)"
