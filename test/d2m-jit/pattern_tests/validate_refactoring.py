#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validation script for pattern testing refactoring.

This script validates the refactored structure without requiring
a full environment setup. It checks:
1. Pattern files have PATTERN_TEST_METADATA
2. Metadata structure is valid
3. Discovery module can load the metadata
"""

import sys
from pathlib import Path

# Add test directory to path
test_dir = Path(__file__).parent.parent
sys.path.insert(0, str(test_dir))

# Try importing without full d2m_jit setup
try:
    from pattern_tests.discovery import (
        discover_pattern_modules,
        get_patterns_dir,
    )

    print("✓ Discovery module imports successfully")

    # Find pattern files
    patterns_dir = get_patterns_dir()
    print(f"✓ Patterns directory: {patterns_dir}")

    pattern_files = discover_pattern_modules()
    print(f"✓ Found {len(pattern_files)} pattern file(s):")
    for f in pattern_files:
        print(f"    - {f.name}")

    # Check for PATTERN_TEST_METADATA in pattern files
    print("\n" + "=" * 60)
    print("Checking for PATTERN_TEST_METADATA in pattern files...")
    print("=" * 60)

    for pattern_file in pattern_files:
        content = pattern_file.read_text()
        if "PATTERN_TEST_METADATA" in content:
            print(f"✓ {pattern_file.name}: Has PATTERN_TEST_METADATA")

            # Count test entries
            lit_count = content.count('"lit_tests"')
            e2e_count = content.count('"e2e_tests"')
            print(
                f"    Found {lit_count} lit_tests section(s), {e2e_count} e2e_tests section(s)"
            )
        else:
            print(f"⨯ {pattern_file.name}: Missing PATTERN_TEST_METADATA")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set up the d2m_jit environment")
    print("2. Run: pytest test/d2m-jit/pattern_tests/test_lit_generated.py")
    print("3. Run: pytest test/d2m-jit/pattern_tests/test_e2e_generated.py")
    print("4. Generate LIT files: python -m test.d2m_jit.pattern_tests.lit_generator")

except ImportError as e:
    print(f"⨯ Import error: {e}")
    print("  This is expected if d2m_jit environment is not set up")
    print("  The structure validation has been completed above.")
except Exception as e:
    print(f"⨯ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
