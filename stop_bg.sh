#!/usr/bin/env bash
cd "$(dirname "$0")"
for f in run.pid autopush.pid; do [ -f "$f" ] && kill "$(cat $f)" 2>/dev/null || true; done
echo "stopped"; ./push_results.sh || true
