#!/usr/bin/env bash
# Copy everything the run has produced so far (CSV / JSON / PNG + log tail) to results/<run>/ on the results branch and push it.
cd "$(dirname "$0")"
WT="$(cat .results_wt)"; BR="$(cat .results_branch)"; RUN="$(cat .run_name)"; SRC="$PWD"
DEST="$WT/results/$RUN"; mkdir -p "$DEST/data" "$DEST/plots"
cp -f data/*.csv data/*.json "$DEST/data/" 2>/dev/null || true
cp -f plots/*.png "$DEST/plots/" 2>/dev/null || true
tail -n 3000 run.log > "$DEST/run_tail.log" 2>/dev/null || true
cp -f run_status.txt "$DEST/" 2>/dev/null || true
cd "$WT"
git add -f "results/$RUN"
git diff --cached --quiet || git commit -q -m "run results $RUN $(date -Is)"
for t in 1 2 3 4; do
    git pull -q --rebase origin "$BR" 2>/dev/null; git push -q -u origin "HEAD:$BR" && break || sleep $((2 ** t))
done
