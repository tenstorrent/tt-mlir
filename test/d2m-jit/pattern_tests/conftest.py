# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for pattern tests.

This extends the parent conftest.py with pattern-specific fixtures.
"""

import pytest
import sys
from pathlib import Path

# Ensure the parent test directory is in the path for imports
test_dir = Path(__file__).parent.parent
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

# Import the discovery module to validate pattern loading
from pattern_tests.discovery import discover_all_pattern_tests


def pytest_collection_modifyitems(config, items):
    """Add markers and metadata to pattern test items."""
    for item in items:
        # Add marker for pattern tests
        item.add_marker(pytest.mark.pattern_test)

        # Extract pattern name from test ID if present
        if "::" in item.nodeid:
            parts = item.nodeid.split("::")
            for part in parts:
                if "eltwise" in part or "pattern" in part:
                    # Extract pattern name from test ID
                    pattern_name = part.split("::")[0] if "::" in part else part
                    item.add_marker(pytest.mark.parametrize_name(pattern_name))
                    break


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "pattern_test: mark test as a pattern test (generated from metadata)"
    )
    config.addinivalue_line("markers", "parametrize_name: pattern name for filtering")


@pytest.fixture(scope="session", autouse=True)
def validate_pattern_discovery():
    """Validate that pattern discovery works at test session start."""
    patterns = discover_all_pattern_tests()
    if not patterns:
        pytest.fail(
            "No pattern test metadata found. "
            "Ensure PATTERN_TEST_METADATA is defined in pattern files."
        )
    print(f"\nDiscovered {len(patterns)} pattern(s) with test metadata")
    for p in patterns:
        print(f"  - {p['pattern_name']}")
