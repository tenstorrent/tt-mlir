#!/bin/bash

# GDB Debug Session for TT-MLIR Gather Crash
# Issue #3849: CPU Fallback Gather Op Crash

set -e

echo "=========================================="
echo "GDB Debug Session for Gather Op Crash"
echo "Issue #3849: CPU Fallback Gather Op Crash"  
echo "=========================================="

cd /home/linux/github/tt-mlir/build-minimal

# Source debug environment
source ../debug_env.sh

echo "Starting GDB session with gather test..."
echo "Commands you can use in GDB:"
echo "  (gdb) run ../debug_gather.mlir --convert-ttir-to-linalg"
echo "  (gdb) bt       # Show stack trace when crash occurs"
echo "  (gdb) info registers"
echo "  (gdb) frame N  # Switch to frame N"
echo "  (gdb) print variable_name"
echo "  (gdb) continue # Continue execution"
echo "  (gdb) quit     # Exit GDB"

exec gdb --args ./bin/ttmlir-opt ../debug_gather.mlir --convert-ttir-to-linalg
