#!/usr/bin/env bash
#
# graph_visualize.sh - capture a TTNN graph report from a flatbuffer and view it
# in ttnn-visualizer. Encodes the working flow (each step below was a footgun):
#
#   1. ttrt run <fb> --graph-capture report.json
#        NORMAL mode -> real L1 addresses (NO_DISPATCH gives all-zero addrs).
#        Writes the report even if the run crashes, via run.py's flush-on-crash.
#   2. python -m ttnn.graph_report report.json db/        (import -> SQLite)
#   3. stage db/ as $TT_METAL_HOME/generated/ttnn/reports/demo_<name>/
#        The `demo_` prefix is what makes the report show in the UI picker for a
#        fresh browser session. `--profiler-path` only registers it in an
#        auto-opened tab, so headless it leaves the picker empty.
#   4. serve with LAUNCH_BROWSER_ON_START=false + --daemon
#        browser-off avoids the headless webbrowser.open() that wedges the single
#        gevent worker; --daemon survives shell/session teardown. Reuses an
#        already-running server on the port (it serves every demo_* dir).
#   5. open http://127.0.0.1:<port>   (127.0.0.1, NOT localhost - IPv4-only bind)
#
# Usage:
#   graph_visualize.sh <flatbuffer.ttnn> [options]
#
# Options:
#   --name NAME        report name shown in the picker (default: flatbuffer stem)
#   -p, --port N       visualizer port (default: 8011)
#   --out DIR          working dir for report.json + db (default: repo/graph_capture_out/<name>)
#   --svg              generate SVGs during import (slower)
#   --no-serve         capture + import + stage only; don't launch the visualizer
#   --ttrt-args "..."  extra args passed through to `ttrt run`
#   -h, --help         show this help

set -euo pipefail

usage() { sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^#\( \|$\)//'; exit "${1:-0}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../../.." && pwd))"

NAME="" PORT=8011 SVG="" SERVE=1 TTRT_EXTRA="" OUTDIR="" FLATBUFFER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)        NAME="$2"; shift 2 ;;
    -p|--port)     PORT="$2"; shift 2 ;;
    --out)         OUTDIR="$2"; shift 2 ;;
    --svg)         SVG="--svg"; shift ;;
    --no-serve)    SERVE=0; shift ;;
    --ttrt-args)   TTRT_EXTRA="$2"; shift 2 ;;
    -h|--help)     usage 0 ;;
    -*)            echo "unknown option: $1" >&2; usage 1 ;;
    *)             if [[ -z "$FLATBUFFER" ]]; then FLATBUFFER="$1"; shift
                   else echo "unexpected argument: $1" >&2; usage 1; fi ;;
  esac
done

[[ -n "$FLATBUFFER" ]] || { echo "error: no flatbuffer given" >&2; usage 1; }
[[ -e "$FLATBUFFER" ]] || { echo "error: flatbuffer not found: $FLATBUFFER" >&2; exit 1; }

# report name -> sanitized, demo_-prefixed (prefix required for the UI picker)
[[ -n "$NAME" ]] || NAME="$(basename "$FLATBUFFER" .ttnn)"
NAME="demo_$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9_.-' '_')"

# env/activate sets TT_METAL_HOME; it references unset vars, so relax -u for it
if [[ -f "$REPO_ROOT/env/activate" ]]; then
  set +u; # shellcheck disable=SC1091
  source "$REPO_ROOT/env/activate"; set -u
fi
: "${TT_METAL_HOME:?TT_METAL_HOME not set - source env/activate first}"

OUTDIR="${OUTDIR:-$REPO_ROOT/graph_capture_out/$NAME}"
REPORT="$OUTDIR/report.json"; DB="$OUTDIR/db"
REPORTS_DIR="$TT_METAL_HOME/generated/ttnn/reports"; DEMO_DIR="$REPORTS_DIR/$NAME"
mkdir -p "$OUTDIR"

# 1. capture (continue on non-zero: a device crash still leaves a flushed report)
echo "==> [1/4] capturing graph report from: $FLATBUFFER"
# shellcheck disable=SC2086
ttrt run "$FLATBUFFER" --graph-capture "$REPORT" $TTRT_EXTRA \
  || echo "    (ttrt exited non-zero; if it crashed, run.py flush-on-crash should still have written the report)"
[[ -s "$REPORT" ]] || { echo "error: no report at $REPORT" >&2; exit 1; }

# 2. import
echo "==> [2/4] importing -> $DB"
rm -rf "$DB"
# shellcheck disable=SC2086
python -m ttnn.graph_report "$REPORT" "$DB" $SVG

# 3. stage as a demo_ report so the UI lists it for any session
echo "==> [3/4] staging as '$NAME' under $REPORTS_DIR"
mkdir -p "$REPORTS_DIR"; rm -rf "$DEMO_DIR"; cp -r "$DB" "$DEMO_DIR"

# 4. serve (reuse a running server; else launch daemonized with browser-off)
if [[ "$SERVE" -eq 0 ]]; then
  echo "==> [4/4] --no-serve: start later with"
  echo "     LAUNCH_BROWSER_ON_START=false ttnn-visualizer --server --port $PORT --daemon"
  exit 0
fi
if curl -4 -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/"; then
  echo "==> [4/4] visualizer already up on $PORT - refresh the page and pick '$NAME'"
else
  echo "==> [4/4] launching ttnn-visualizer on $PORT (daemon, browser-off)"
  LAUNCH_BROWSER_ON_START=false ttnn-visualizer --server --port "$PORT" --daemon >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    if curl -4 -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/"; then break; fi
    sleep 1
  done
fi

cat <<EOF

  report '$NAME' ready. View it:
    - on this box:   http://127.0.0.1:$PORT      (127.0.0.1, NOT localhost - IPv4-only)
    - from a laptop: run ON THE LAPTOP:
        ssh -L $PORT:127.0.0.1:$PORT <host-you-ssh-into>
      then open http://localhost:$PORT
  then pick '$NAME' from the report list.
EOF
