#!/usr/bin/env bash
# Download every perf-reports artifact (perf JSON + shlo/ttir/ttnn dumps) for one run.
set -u
RUN=$1; VAR=$2
gh api "repos/tenstorrent/tt-xla/actions/runs/$RUN/jobs?per_page=100" --paginate \
  -q '.jobs[] | select(.name|contains("perf ")) | "\(.id)\t\(.name)"' \
| awk -F'\t' '{n=$2; sub(/^.*\/ perf /,"",n); sub(/ \(n150-perf\)$/,"",n); print $1"\t"n}' \
| while IFS=$'\t' read -r id name; do
    [ -n "$id" ] || continue
    key=$(printf '%s' "$name" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
    d=raw/$VAR/$key
    [ -f "$d/.done" ] && { echo "skip $key"; continue; }
    mkdir -p "$d"
    if gh run download "$RUN" --repo tenstorrent/tt-xla --name "perf-reports-$id" --dir "$d" >/dev/null 2>&1; then
      echo "$id" > "$d/.done"; echo "ok   $key ($id)"
    else
      echo "FAIL $key ($id)"
    fi
  done
