#!/usr/bin/env bash
# Aggregate chisel failures per op and dump per-op JSONL files.
# Usage: ./chisel_aggregate_failures.sh [-o <output_dir>] <results.jsonl> [results2.jsonl ...]

set -euo pipefail

OUTPUT_DIR="chisel_failures"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [-o <output_dir>] <results.jsonl> [results2.jsonl ...]" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Collect all failures into a temp file
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

jq -c 'select(.status != "ok")' "$@" > "$TMP"

if [[ ! -s "$TMP" ]]; then
    echo "No failures found."
    exit 0
fi

# Dump per-op JSONL files (op name sanitized: dots and slashes -> underscores)
jq -r '.op' "$TMP" | sort -u | while read -r op; do
    safe_op="${op//[.\/]/_}"
    jq -c --arg op "$op" 'select(.op == $op)' "$TMP" > "$OUTPUT_DIR/${safe_op}.jsonl"
done

# Print summary: count per op, sorted descending
echo "=== Failure counts per op ==="
jq -r '.op' "$TMP" | sort | uniq -c | sort -rn | awk '{printf "%6d  %s\n", $1, $2}'
echo ""
echo "Per-op files written to: $OUTPUT_DIR/"
