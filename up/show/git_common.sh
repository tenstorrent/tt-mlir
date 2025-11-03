#!/bin/bash

# Common git operations for uplift scripts
TT_MLIR_DIR="/Users/achoudhury/code/tt-mlir"
TT_METAL_DIR="/Users/achoudhury/code/tt-metal"
LOGS_DIR="$(dirname "$0")/../scratch"

init() {
    mkdir -p $LOGS_DIR
    fetch_branches
}

# Fetch latest branches
fetch_branches() {
    git -C "$TT_MLIR_DIR" fetch origin main --prune
    git -C "$TT_MLIR_DIR" fetch origin uplift --prune
    git -C "$TT_METAL_DIR" fetch origin main --prune
}

# Get metal commit from tt-mlir branch
get_metal_commit() {
    local branch="$1"
    git -C "$TT_MLIR_DIR" show "origin/${branch}:third_party/CMakeLists.txt" | grep "set(TT_METAL_VERSION" | sed 's/.*TT_METAL_VERSION "\([a-f0-9]*\)".*/\1/'
}

# Get git log command
# e.g. output: "git log 8141206..origin/main --format=\"%cd %h by %an|||: %s\" --date=short-local | column -t -s \"|||\""
log_cmd() {
    local commit_range="$1"
    local path="$2"

    FORMAT="--format=\"%cd %h by %an|||: %s\""
    DATE="--date=short-local"
    POST_PROCESS="column -t -s \"|||\""

    if [ $# -eq 1 ]; then
        echo "git log $commit_range $FORMAT $DATE | $POST_PROCESS"
    else
        echo "git -C $path log $commit_range $FORMAT $DATE | $POST_PROCESS"
    fi
}

# Get git log output
get_git_log() {
    local commit_range="$1"
    eval $(log_cmd "$commit_range" "$TT_METAL_DIR")
}

# Get commit count
get_commit_count() {
    local commit_range="$1"
    git -C "$TT_METAL_DIR" rev-list --count "$commit_range"
} 
