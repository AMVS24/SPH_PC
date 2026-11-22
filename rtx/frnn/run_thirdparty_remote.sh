#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/data/thirdparty_remote_ada_20261120.txt"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 1; fi
SSH="ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes pcproj@10.1.19.63"

$SSH 'bash -s' > "$OUT" 2>&1 <<'REMOTE'
set -u
cd ~/sph_ext_20261120 || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
PTS=~/sph_ext_20261120/extdrv/pts_disordered_1M.txt

echo "== build cuNSearch"
for P in ON OFF; do
  B=cuns_build_$P
  if [ ! -e $B ]; then
    mkdir $B && cd $B
    cmake ../cuNSearch -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89 -DCUNSEARCH_USE_DOUBLE_PRECISION=$P > cmake.log 2>&1
    make -j16 cuNSearch > make.log 2>&1; echo "cuns_$P make rc=$?"; tail -3 make.log
    DEF=""; [ $P = ON ] && DEF=-DCUNSEARCH_USE_DOUBLE_PRECISION
    nvcc -O3 -arch=sm_89 -std=c++14 $DEF -I../cuNSearch/include -I. -I./cuNSearch -I../cuNSearch/Utils ../extdrv/cuns_driver.cu -o cuns_driver libcuNSearch.a > drv.log 2>&1
    echo "cuns_$P driver rc=$?"; grep -i error drv.log | head -5
    cd ..
  fi
done

echo "== build RTNN"
if [ ! -e rtnn/src/build ]; then
  mkdir rtnn/src/build && cd rtnn/src/build
  cmake -DKNN=5 -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DCUDA_NVCC_FLAGS="-arch;sm_89" .. > cmake.log 2>&1
  echo "rtnn cmake rc=$?"; tail -3 cmake.log
  make -j16 > make.log 2>&1; echo "rtnn make rc=$?"; grep -iE "error" make.log | head -10
  cd ~/sph_ext_20261120
fi
ls -la rtnn/src/build/bin/optixNSearch cuns_build_ON/cuns_driver cuns_build_OFF/cuns_driver

echo "== runs"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
timeout 2400 flock -w 600 /tmp/sph_bench.lock bash -c '
PTS=$HOME/sph_ext_20261120/extdrv/pts_disordered_1M.txt
export LD_LIBRARY_PATH=$HOME/sph_ext_20261120/rtnn/src/build/lib:${LD_LIBRARY_PATH:-}
for r in 2.754 1.0; do
  for P in ON OFF; do
    for l in 1 2 3; do echo "RUN cuns prec=$P r=$r launch=$l"; timeout 600 $HOME/sph_ext_20261120/cuns_build_$P/cuns_driver $PTS $r 5 | grep ^CUNS; done
  done
  for K in 50 256; do
    for l in 1 2 3; do echo "RUN rtnn r=$r K=$K launch=$l"; timeout 600 $HOME/sph_ext_20261120/rtnn/src/build/bin/optixNSearch -f $PTS -r $r -k $K -d 0 2>&1 | grep -iE "total search time|search compute|result copy|build|sort|Caught|error"; done
  done
done
echo "RUN rtnn sanity r=2.754 K=256"
timeout 900 $HOME/sph_ext_20261120/rtnn/src/build/bin/optixNSearch -f $PTS -r 2.754 -k 256 -d 0 -c 1 2>&1 | tail -15
'
echo "rc=$?"
REMOTE
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
