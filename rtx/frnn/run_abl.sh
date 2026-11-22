#!/usr/bin/env bash
set -uo pipefail
BIN="${1:?bin dir}"
OUT="${2:?out file}"
SET="${3:-ablate}"
LAUNCHES="${LAUNCHES:-3}"
REPS="${REPS:-5}"
N="${N:-1000000}"
EXT=""
[ -f "$BIN/abl.exe" ] && EXT=".exe"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 1; fi
: > "$OUT"

one(){
    for b in abl_instr abl; do
        L=$LAUNCHES; R=$REPS; W=1.0
        [ "$b" = abl_instr ] && L=1 && R=1 && W=0
        for l in $(seq 1 $L); do
            line=$(timeout 300 "$BIN/$b$EXT" "$@" --reps "$R" --warm-s "$W" 2>&1 | grep '^ABL' || true)
            if [ -z "$line" ]; then echo "FAIL build=$b args=$*" | tee -a "$OUT"; else echo "$line launch=$l" >> "$OUT"; fi
        done
    done
    echo "done $*"
}

if [ "$SET" = ablate ] || [ "$SET" = all ]; then
    for m in count density force; do
        for v in base report fasttrace unsorted spherefront aabbseg tiny; do
            one --variant $v --mode $m --n "$N" --k 80
        done
        one --variant base --mode $m --n "$N" --k 80 --grid-div 2
    done
    one --variant empty --mode count --n "$N" --k 80
fi

if [ "$SET" = ksweep ] || [ "$SET" = all ]; then
    for k in 1 4 16 32 80 128 256 512; do
        for v in base spherefront empty tiny; do
            one --variant $v --mode count --n "$N" --k "$k"
        done
        one --variant base --mode density --n "$N" --k "$k"
        one --variant spherefront --mode density --n "$N" --k "$k"
    done
fi

if [ "$SET" = dists ] || [ "$SET" = all ]; then
    for d in lattice; do
        for m in count force; do
            for v in base spherefront; do
                one --variant $v --mode $m --n "$N" --k 80 --dist $d
            done
        done
    done
fi

if [ "$SET" = profile ] || [ "$SET" = all2 ]; then
    for core in 0.01 0.001 0.0001; do
        for v in base spherefront empty tiny; do
            one --variant $v --mode count --n "$N" --k 80 --dist profile --alpha 3 --core $core --box 100 --no-grid
        done
    done
fi

if [ "$SET" = sparse ] || [ "$SET" = all2 ]; then
    for d in disordered lattice; do
        one --variant base --mode count --n "$N" --radius const:1 --box 1000 --dist $d
        one --variant base --mode count --n "$N" --radius const:1 --box 1000 --dist $d --grid-compact
    done
    one --variant base --mode count --n "$N" --k 80 --grid-compact
    one --variant base --mode count --n "$N" --k 1 --grid-compact
fi

if [ "$SET" = mech ] || [ "$SET" = all2 ]; then
    for k in 1 4 16 80 256; do
        one --variant refit --mode count --n "$N" --k "$k"
        one --variant aabbseg --mode count --n "$N" --k "$k"
    done
    one --variant refit --mode force --n "$N" --k 80
fi

if [ "$SET" = bridge ]; then
    for spec in "--dist file --file $PTS --radius const:2.754" "--dist file --file $PTS --radius const:1.0" "--dist surface --radius const:1" "--dist surface --radius const:0.5"; do
        one --variant base --mode count --n "$N" $spec --xfer
        one --variant base --mode count --n "$N" $spec --grid-fp64 --xfer
        one --variant base --mode count --n "$N" $spec --grid-compact
        one --variant kcap --kcap 50 --mode count --n "$N" $spec
        one --variant refit --mode count --n "$N" $spec
    done
fi
