#!/usr/bin/env bash
set -uo pipefail
B="${1:?binary}"
OUT="${2:?out}"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 1; fi
: > "$OUT"
for n in 50000 200000; do
  for spec in lognormal:1:2:1:330 uniform:1:160; do
    for l in 1 2 3; do timeout 600 "$B" --dist randrad --box 103 --n $n --rspec $spec --budget-s 3 --single-cap-s 30 | grep ^FRNNML | sed "s/\$/ launch=$l/" >> "$OUT" || echo "RUNFAIL n=$n $spec" >> "$OUT"; done
  done
done
for spec in uniform:0.5:1.5 uniform:0.2:2.0 lognormal:0:0.5:0.3:3; do
  for l in 1 2 3; do timeout 600 "$B" --dist randrad --box 103 --n 1000000 --rspec $spec --budget-s 3 --single-cap-s 30 | grep ^FRNNML | sed "s/\$/ launch=$l/" >> "$OUT" || echo "RUNFAIL n=1M $spec" >> "$OUT"; done
done
