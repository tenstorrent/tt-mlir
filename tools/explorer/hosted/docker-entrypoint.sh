#!/bin/bash
set -e

cd /workspace
source env/activate
exec /workspace/build/bin/tt-explorer "$@"

