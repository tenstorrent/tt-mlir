#!/bin/bash
# Usage: ./metal_to_mlir.sh <tt-metal-log-range>
# Example: ./metal_to_mlir.sh abc..def

# Exit early on error
set -e

# Relative paths
TT_METAL_DIR="third_party/tt-metal/src/tt-metal"
CMAKELISTS="third_party/CMakeLists.txt"

# Global settings 
VERBOSE=""


print_usage() {
  cat >&2 <<EOF
Usage: $0 [OPTIONS] <tt-metal-log-range>

Required positional arguments:
  <tt-metal-log-range>    Range of tt-metal commits to bisect (e.g. abc..def)

Options:
  -v, --verbose           Enable verbose output
  -l, --local-only        Do not push new branch to origin (local branch only)
  -p, --patches PATCHES   MLIR patch commit range (string)
  -h, --help              Show this help message and exit
EOF
}

parse_and_validate_args() {
  # Use GNU getopt to parse options and positional arguments in any order
  local TEMP
  TEMP=$(getopt -o vhlp: --long verbose,help,local-only,patches: -n "$0" -- "$@")
  if [ $? != 0 ]; then
    print_usage
    exit 1
  fi
  eval set -- "$TEMP"

  MLIR_PATCHES=""
  LOCAL_ONLY=""
  # Parse options
  while true; do
    case "$1" in
      -v|--verbose)
        VERBOSE=1; shift ;;
      -l|--local-only)
        LOCAL_ONLY=1; shift ;;
      -p|--patches)
        MLIR_PATCHES="$2"; shift 2 ;;
      -h|--help)
        print_usage; exit 0 ;;
      --)
        shift ; break ;;
      *)
        echo "Internal error in getopt parsing"; exit 1 ;;
    esac
  done

  # The first positional argument is required: bisect window
  if [ $# -lt 1 ]; then
    print_usage
    exit 1
  fi
  METAL_BISECT_WINDOW="$1"
  shift

  # If there are any more positional arguments, error
  if [ $# -gt 0 ]; then
    echo "Unexpected extra positional arguments: $*" >&2
    exit 1
  fi

  # Validate values (bisect window and patches must be valid git commit hashes)
  if [ -n "$MLIR_PATCHES" ]; then
    fetch_and_validate_range "$MLIR_PATCHES"

    build_mlir_patch_map "$MLIR_PATCHES"
  fi
  fetch_and_validate_range "$METAL_BISECT_WINDOW" "$TT_METAL_DIR"
}

# Split a commit range of the form abc..def into SPLIT_LEFT and SPLIT_RIGHT
# Usage: split_commit_range <range>
split_commit_range() {
  local range="$1"
  if [[ "$range" == *..* ]]; then
    SPLIT_LEFT="${range%%..*}"
    SPLIT_RIGHT="${range##*..}"
  else
    echo "Invalid range format: $range. Only abc..def format supported." >&2
    exit 1
  fi
}

# Build MLIR_PATCH_MAP: metal short sha -> mlir patch commit for a given patch range
# Usage: build_mlir_patch_map <patch_range>
build_mlir_patch_map() {
  local patch_range="$1"
  declare -gA MLIR_PATCH_MAP

  split_commit_range "$patch_range"
  local patch_left="$SPLIT_LEFT"
  local patch_right="$SPLIT_RIGHT"

  # Get all patch commits in range, oldest to newest
  local patch_commits
  patch_commits=($(git rev-list --reverse "$patch_left..$patch_right"))

  if [[ $VERBOSE ]]; then
    echo "Building MLIR patch map from $patch_left to $patch_right..."
  fi
  for patch_commit in "${patch_commits[@]}"; do
    local msg sha
    msg=$(git log -1 --pretty=%s "$patch_commit")
    # Extract last 5-40 hex chars after last space (short or full sha)
    sha=$(echo "$msg" | grep -oE '[0-9a-f]{5,40}$')
    # Normalize sha to 7 chars from git
    sha=$(git -C "$TT_METAL_DIR" rev-parse --short "$sha")
    if [[ -n "$sha" ]]; then
      MLIR_PATCH_MAP[$sha]="$patch_commit"
      if [[ $VERBOSE ]]; then
        echo "Mapping metal $sha -> mlir patch $patch_commit"
      fi
    fi
  done
  if [[ $VERBOSE ]]; then
    echo "MLIR patch map built"
  fi
}

# Fetch both endpoints of a commit range, ensure both are valid SHAs, and check ancestor relationship
# Usage: fetch_and_validate_range <range> [repo_dir]
fetch_and_validate_range() {
  local range="$1"
  local repo_dir="${2:-.}"

  # Split range (only supports abc..def)
  split_commit_range "$range"
  local left="$SPLIT_LEFT"
  local right="$SPLIT_RIGHT"

  # For each endpoint, check if it exists locally; if not, fetch all refs, then check again
  # Uses `cat-file -e` instead of `rev-parse --verify` to not just check hash but object too
  for endpoint in "$left" "$right"; do
    if ! git -C "$repo_dir" cat-file -e "$endpoint" &>/dev/null; then
      git -C "$repo_dir" fetch --prune origin &>/dev/null
      if ! git -C "$repo_dir" cat-file -e "$endpoint" &>/dev/null; then
        # As a last resort, try to fetch the specific endpoint (may only work if endpoint is a ref)
        git -C "$repo_dir" fetch --prune origin "$endpoint" &>/dev/null
        if ! git -C "$repo_dir" cat-file -e "$endpoint" &>/dev/null; then
          echo "Commit $endpoint not found locally, after fetching origin, or after fetching origin $endpoint in $repo_dir" >&2
          exit 1
        fi
      fi
    fi
  done

  # Check ancestor relationship (left should be ancestor of right)
  if [ "$left" != "$right" ]; then
    git -C "$repo_dir" merge-base --is-ancestor "$left" "$right" || {
      echo "Error: $left is not an ancestor of $right in $repo_dir" >&2
      exit 1
    }
  fi
}

set_proj_version_in_cmake() {
  local var_name="$1"
  local new_value="$2"
  sed -i -E "s/set\($var_name \"[a-zA-Z0-9]+\"\)/set($var_name \"$new_value\")/" "$CMAKELISTS"
}

set_metal_version_from_mlir() {
  local metal_hash="$1"
  set_proj_version_in_cmake "TT_METAL_VERSION" "$metal_hash"
}

get_metal_commit_hashes() {
  # Get hashes from tt-metal in old-to-new order
  local range="$1"
  if [[ $VERBOSE ]]; then
    echo "Getting commit hashes from $TT_METAL_DIR in range $range..."
  fi
  local hashes=($(git -C "$TT_METAL_DIR" log --reverse --format="%h" "$range"))
  if [[ ${#hashes[@]} -eq 0 ]]; then
    echo "No commits found in range $range in $TT_METAL_DIR" >&2
    echo "Please check the range and try again" >&2
    exit 1
  fi
  METAL_COMMITS=("${hashes[@]}")
}

apply_metal_and_patch_commits() {
  RAND_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 2 | head -n 1)
  BRANCH_NAME="$USER/up_bisect_$(date +%Y_%m_%d)_$RAND_SUFFIX"
  if [[ $VERBOSE ]]; then
    echo "Creating branch: $BRANCH_NAME"
  fi
  git checkout -b "$BRANCH_NAME"

  if [[ $VERBOSE ]]; then
    echo "Applying hashes: ${METAL_COMMITS[*]}"
  fi
  for HASH in "${METAL_COMMITS[@]}"; do
    if [[ $VERBOSE ]]; then
      echo "Setting TT_METAL_VERSION to $HASH"
    fi
    set_metal_version_from_mlir "$HASH"

    git add "$CMAKELISTS"
    git commit --no-verify -m "Set TT_METAL_VERSION to $HASH" >/dev/null

    if [[ ${#MLIR_PATCH_MAP[@]} -gt 0 ]]; then
      # if HASH is a key in patch map, add patch commit and amend with commit message edited
      if [[ -n "${MLIR_PATCH_MAP[$HASH]}" ]]; then
        if [[ $VERBOSE ]]; then
          echo "Cherry-picking patch commit ${MLIR_PATCH_MAP[$HASH]} for metal commit $HASH"
        fi
        git cherry-pick -n "${MLIR_PATCH_MAP[$HASH]}"
        git commit --no-verify --amend -m "Set TT_METAL_VERSION to $HASH (with patch commit: ${MLIR_PATCH_MAP[$HASH]})" >/dev/null
      fi
    fi
  done

  if [[ ! $LOCAL_ONLY ]]; then
    if [[ $VERBOSE ]]; then
      echo "Pushing branch $BRANCH_NAME to remote..."
    fi
    git push --set-upstream origin "$BRANCH_NAME" >/dev/null
  fi
}

main() {
  parse_and_validate_args "$@"

  get_metal_commit_hashes "$METAL_BISECT_WINDOW"

  new_range_start=$(git rev-parse HEAD)
  # Generate a random branch name
  apply_metal_and_patch_commits
  new_range_end=$(git rev-parse HEAD)

  if [[ $VERBOSE ]]; then
    echo "New commits (most recent first):"
    git log --oneline -n ${#METAL_COMMITS[@]}
  fi
  
  echo -e "\nRange of new commits:"
  echo -e "$new_range_start..$new_range_end\n\n"

  local test_script="up/run/mlir.sh"
  echo -e "Please make sure $test_script has minimal repro and run following command to start bisect"
  echo -e "    git bisect start $new_range_end $new_range_start && git bisect run bash $test_script\n"
}

main "$@"
