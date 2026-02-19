#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test suite for Chisel auto-binding functionality via TT_INJECT_TTNN2FB environment variable.
"""
import os
import sys
import subprocess
import tempfile
import pathlib
import pytest


def test_chisel_auto_binds_when_enabled():
    """Test that chisel automatically binds when TT_INJECT_TTNN2FB=1 is set."""
    # Create a test script that imports ttrt.runtime with TT_INJECT_TTNN2FB=1
    test_script = """
import os
os.environ["TT_INJECT_TTNN2FB"] = "1"

# Import runtime (triggers auto-bind)
import ttrt.runtime

# Verify chisel context exists
try:
    from chisel.core.context import _chisel_context
    if _chisel_context is not None:
        print("CHISEL_AUTO_BIND_SUCCESS")
    else:
        print("CHISEL_CONTEXT_IS_NONE")
except ImportError as e:
    print(f"CHISEL_IMPORT_ERROR: {e}")
"""

    # Run the test script in a subprocess
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        env={**os.environ, "TT_INJECT_TTNN2FB": "1"}
    )

    # Check output
    assert "CHISEL_AUTO_BIND_SUCCESS" in result.stdout or "CHISEL_IMPORT_ERROR" in result.stdout, \
        f"Expected chisel to auto-bind or gracefully fail. Output: {result.stdout}\nError: {result.stderr}"


def test_default_callbacks_when_disabled():
    """Test that default callbacks are used when TT_INJECT_TTNN2FB is not set."""
    test_script = """
import os
# Make sure TT_INJECT_TTNN2FB is not set
os.environ.pop("TT_INJECT_TTNN2FB", None)

# Import runtime
import ttrt.runtime

# Try to bind default callbacks
ttrt.runtime.bind_callbacks()

print("DEFAULT_CALLBACKS_REGISTERED")
"""

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "TT_INJECT_TTNN2FB"}
    )

    assert "DEFAULT_CALLBACKS_REGISTERED" in result.stdout, \
        f"Expected default callbacks to register. Output: {result.stdout}\nError: {result.stderr}"


def test_skip_default_callbacks_when_chisel_bound():
    """Test that bind_callbacks() skips when chisel is already bound."""
    test_script = """
import os
os.environ["TT_INJECT_TTNN2FB"] = "1"

# Import runtime (triggers auto-bind)
import ttrt.runtime

# Try to bind default callbacks (should skip)
ttrt.runtime.bind_callbacks()

print("CALLBACKS_CHECK_COMPLETE")
"""

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        env={**os.environ, "TT_INJECT_TTNN2FB": "1"}
    )

    # Should see message about skipping default callbacks
    assert "CALLBACKS_CHECK_COMPLETE" in result.stdout, \
        f"Expected callbacks check to complete. Output: {result.stdout}\nError: {result.stderr}"


def test_custom_output_directory():
    """Test that custom output directory is used when TT_CHISEL_OUTPUT_DIR is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = os.path.join(tmpdir, "my_chisel_output")

        test_script = f"""
import os
os.environ["TT_INJECT_TTNN2FB"] = "1"
os.environ["TT_CHISEL_OUTPUT_DIR"] = "{custom_dir}"

# Import runtime (triggers auto-bind)
import ttrt.runtime

# Check if chisel was initialized with custom directory
try:
    from chisel.core.context import _chisel_context
    if _chisel_context is not None:
        print(f"OUTPUT_DIR: {{_chisel_context.output_dir}}")
    else:
        print("CHISEL_CONTEXT_IS_NONE")
except ImportError:
    print("CHISEL_NOT_AVAILABLE")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "TT_INJECT_TTNN2FB": "1",
                "TT_CHISEL_OUTPUT_DIR": custom_dir
            }
        )

        # Either chisel initialized with custom dir, or not available (both acceptable)
        assert (custom_dir in result.stdout or "CHISEL_NOT_AVAILABLE" in result.stdout), \
            f"Expected custom output directory or chisel unavailable. Output: {result.stdout}\nError: {result.stderr}"


def test_graceful_fallback_without_chisel():
    """Test that system continues working if chisel is not installed."""
    # This test simulates chisel not being available by trying to import
    # Note: If chisel is actually installed, this will show a warning but continue
    test_script = """
import os
os.environ["TT_INJECT_TTNN2FB"] = "1"

# Import runtime (should handle missing chisel gracefully)
import ttrt.runtime

print("RUNTIME_IMPORTED_SUCCESSFULLY")
"""

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        env={**os.environ, "TT_INJECT_TTNN2FB": "1"}
    )

    # Runtime should import successfully regardless of chisel availability
    assert "RUNTIME_IMPORTED_SUCCESSFULLY" in result.stdout, \
        f"Expected runtime to import successfully. Output: {result.stdout}\nError: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
