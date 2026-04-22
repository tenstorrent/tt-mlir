#!/usr/bin/env bash
# Filter chisel results where status is not "ok"
# Usage: ./chisel_filter_failures.sh <results.jsonl> [results2.jsonl ...]

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <results.jsonl> [results2.jsonl ...]" >&2
    exit 1
fi

jq -c 'select(.status != "ok")' "$@"
