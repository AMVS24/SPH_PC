#!/usr/bin/env bash
set -uo pipefail

BIN="${1:?usage: run_cost.sh <bin_dir> <out_file> <set: sweep|orcs|all> [reps]}"
OUT="${2:?out file}"
SET="${3:-all}"
REPS="${4:-7}"
EXT=""
[ -f "$BIN/cost.exe" ] && EXT=".exe"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-300}"

if [ -e "$OUT" ]; then
    echo "refusing to write: $OUT already exists" >&2
    exit 1
fi
: > "$OUT"

run(){
    for b in ${BUILDS:-cost_instr cost}; do
        r=$REPS
        w="${WARM_S:-1.0}"
        [ "$b" = cost_instr ] && r=1 && w=0
        line=$(timeout "$PER_RUN_TIMEOUT" "$BIN/$b$EXT" "$@" --reps "$r" --warm-s "$w" 2>&1 | grep '^COST' || true)
        if [ -z "$line" ]; then
            echo "FAIL build=$b args=$*" | tee -a "$OUT"
        else
            echo "$line" >> "$OUT"
            echo "$b $* -> ok"
        fi
    done
}

if [ "$SET" = sweep ] || [ "$SET" = all ]; then
    for n in 200000 1000000; do
        for k in 1 4 16 32 64 80 128 256 512; do
            run --dist disordered --box 103 --n "$n" --k "$k"
        done
    done
fi

if [ "$SET" = orcs ] || [ "$SET" = all ]; then
    sph_r(){ python3 -c "import math;n=$1;print(f'{(80.0/(n/1000.0**3*4.0/3.0*math.pi))**(1/3):.4f}')" 2>/dev/null || python -c "import math;n=$1;print(f'{(80.0/(n/1000.0**3*4.0/3.0*math.pi))**(1/3):.4f}')"; }
    for n in 50000 200000 1000000; do
        sr=$(sph_r "$n")
        for d in lattice disordered cluster; do
            for rad in "const:1" "const:$sr" "const:160" "uniform:1:160" "lognormal:1:2:1:330"; do
                if [ "$n" = 1000000 ] && [ "$d" = cluster ] && [ "$rad" != "const:1" ] && [ "$rad" != "lognormal:1:2:1:330" ]; then continue; fi
                if [ "$n" = 1000000 ] && [ "$rad" = "uniform:1:160" ]; then continue; fi
                if [ "$n" = 1000000 ] && [ "$rad" = "const:160" ]; then continue; fi
                run --dist "$d" --box 1000 --n "$n" --radius "$rad"
            done
        done
    done
fi
