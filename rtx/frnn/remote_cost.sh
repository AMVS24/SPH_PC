#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
JUMP="${JUMP:-bpgcjump}"
INNER="ssh -i ~/.ssh/hri_rsa_1 -o BatchMode=yes -o ConnectTimeout=20 HRI@10.1.19.30"
SSHO="-o BatchMode=yes -o ConnectTimeout=20"
TAG="${TAG:-20261120}"
REPS="${REPS:-7}"
SET="${SET:-all}"
CAP_S="${CAP_S:-3300}"

remote(){ ssh $SSHO "$JUMP" "$INNER '$1'"; }

remote "nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader" > /dev/null

RD="sph_dir2_$TAG"
i=1
while remote "test -e $RD" ; do i=$((i+1)); RD="sph_dir2_${TAG}_$i"; done
echo "remote dir: ~/$RD"

tar czf - -C "$ROOT" \
    rtx/frnn/cost.cu rtx/frnn/costPrograms.cu rtx/frnn/costShared.h rtx/frnn/build_cost.sh rtx/frnn/run_cost.sh \
    rtx/third_party/optix/include \
  | ssh $SSHO "$JUMP" "$INNER 'mkdir $RD && tar xzf - -C $RD'"

PAYLOAD=$(cat <<EOF
set -uo pipefail
cd ~/$RD/rtx/frnn
CC=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
export ARCH=sm_\$CC
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME=/usr/local/cuda; else export CUDA_HOME=\$(ls -d /usr/local/cuda-* | sort -V | tail -1); fi
{
  echo "date: \$(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,power.limit --format=csv
  "\$CUDA_HOME/bin/nvcc" --version | tail -2
  echo "ARCH=\$ARCH CUDA_HOME=\$CUDA_HOME"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} > gpu_info.txt 2>&1
bash build_cost.sh > build.log 2>&1 || { echo BUILD_FAILED; tail -30 build.log; exit 1; }
(nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu --format=csv,noheader -lms 1000 > clocks.txt 2>&1 &)
flock /tmp/sph_bench.lock timeout $CAP_S bash run_cost.sh build_cost cost_results.txt $SET $REPS > run.log 2>&1
pkill -u \$(id -u) -f "nvidia-smi --query-gpu=timestamp" || true
echo RUN_DONE
EOF
)
echo "$PAYLOAD" | ssh $SSHO "$JUMP" "$INNER 'bash -s'"

GPU=$(remote "cd ~/$RD/rtx/frnn && sed -n 3p gpu_info.txt" | cut -d, -f1 | tr ' ' '_' )
OUT="$HERE/data/cost_remote_${GPU}_${TAG}.txt"
INFO="$HERE/data/cost_remote_${GPU}_${TAG}_info.txt"
CLK="$HERE/data/cost_remote_${GPU}_${TAG}_clocks.txt"
for f in "$OUT" "$INFO" "$CLK"; do
    if [ -e "$f" ]; then echo "refusing to overwrite $f; results stay in ~/$RD on the remote" >&2; exit 1; fi
done
remote "cat ~/$RD/rtx/frnn/cost_results.txt" > "$OUT"
remote "cat ~/$RD/rtx/frnn/gpu_info.txt" > "$INFO"
remote "cat ~/$RD/rtx/frnn/clocks.txt" > "$CLK"
echo "pulled: $OUT"
echo "fit with: python -I $HERE/fit_cost.py --sweep $OUT --orcs $OUT --ada $HERE/data/frnn_box1000.json"
