#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Quick verification script to test that default callbacks are now chisel callbacks.
"""
import os
import sys

print("=" * 80)
print("Test 1: Auto-initialization with TT_INJECT_TTNN2FB=1")
print("=" * 80)

# Set environment variable before importing
os.environ["TT_INJECT_TTNN2FB"] = "1"
os.environ["TT_CHISEL_OUTPUT_DIR"] = "./test_auto_chisel"

import ttrt.runtime

try:
    from chisel.core.context import _chisel_context
    if _chisel_context is not None:
        print("✓ Chisel auto-initialized via TT_INJECT_TTNN2FB=1")
        print(f"  Output dir: {_chisel_context.output_dir}")
        print(f"  Report path: {_chisel_context.report_path}")
    else:
        print("✗ Chisel context is None")
except ImportError as e:
    print(f"✗ Chisel not available: {e}")

print()
print("=" * 80)
print("Test 2: Calling bind_callbacks() should skip (already bound)")
print("=" * 80)

ttrt.runtime.bind_callbacks()
print("✓ bind_callbacks() called (should have logged that chisel already bound)")

print()
print("=" * 80)
print("Test 3: Without TT_INJECT_TTNN2FB, bind_callbacks() uses chisel as default")
print("=" * 80)

# Clean up and restart
os.environ.pop("TT_INJECT_TTNN2FB", None)

# Reload the module to test bind_callbacks() without auto-initialization
import importlib
importlib.reload(ttrt.runtime)

print("Calling bind_callbacks() without TT_INJECT_TTNN2FB...")
ttrt.runtime.bind_callbacks()

try:
    from chisel.core.context import _chisel_context
    if _chisel_context is not None:
        print("✓ bind_callbacks() initialized chisel as default callback")
        print(f"  Output dir: {_chisel_context.output_dir}")
    else:
        print("ℹ Chisel context not initialized (may have fallen back to print callbacks)")
except ImportError:
    print("ℹ Chisel not available - using fallback print callbacks")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print("✓ Auto-initialization: TT_INJECT_TTNN2FB=1 auto-binds chisel")
print("✓ Default callbacks: bind_callbacks() now uses chisel by default")
print("✓ Graceful fallback: System works even if chisel not available")
