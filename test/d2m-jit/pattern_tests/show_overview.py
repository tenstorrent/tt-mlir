#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Show overview of the refactored pattern testing structure."""

import sys
from pathlib import Path


def print_section(title, content=None):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    if content:
        print(content)


def main():
    base_dir = Path(__file__).parent

    print_section("D2M-JIT Pattern Testing Refactoring - Overview")

    print(
        """
This refactoring consolidates pattern testing by moving test metadata
directly into pattern definition files, enabling automatic test generation
and execution.
"""
    )

    print_section("Created Files")

    files = {
        "Core Framework": [
            ("__init__.py", "Package initialization"),
            ("discovery.py", "Pattern test discovery utilities"),
            ("test_e2e_generated.py", "Auto-generated E2E tests"),
            ("test_lit_generated.py", "Auto-generated LIT tests"),
            ("lit_generator.py", "Standalone LIT file generator"),
            ("conftest.py", "Pytest configuration"),
            ("validate_refactoring.py", "Validation script"),
        ],
        "Documentation": [
            ("README.md", "Complete framework documentation"),
            ("QUICK_REFERENCE.md", "Quick reference cheat sheet"),
            ("ARCHITECTURE.md", "Architecture and data flow"),
            ("PATTERN_TEMPLATE.py", "Template for new patterns"),
            ("REFACTORING_SUMMARY.md", "Before/after comparison"),
            ("IMPLEMENTATION_SUMMARY.md", "This refactoring summary"),
        ],
    }

    for category, file_list in files.items():
        print(f"\n{category}:")
        for filename, description in file_list:
            exists = "✓" if (base_dir / filename).exists() else "✗"
            print(f"  {exists} {filename:30s} - {description}")

    print_section("Modified Pattern Files")
    print(
        """
  ✓ tools/d2m-jit/patterns/eltwise_exp_to_kernel.py
    → Added PATTERN_TEST_METADATA with LIT and E2E test configs

  ✓ tools/d2m-jit/patterns/eltwise_add_exp_to_kernel.py
    → Added PATTERN_TEST_METADATA with LIT and E2E test configs
"""
    )

    print_section("Quick Start Guide")
    print(
        """
1. Validate the setup (no environment needed):
   $ cd test/d2m-jit/pattern_tests
   $ python3 validate_refactoring.py

2. Run pattern tests (requires d2m_jit environment):
   $ pytest test/d2m-jit/pattern_tests/test_lit_generated.py
   $ pytest test/d2m-jit/pattern_tests/test_e2e_generated.py

3. Run tests for a specific pattern:
   $ pytest test/d2m-jit/pattern_tests/ -k "eltwise_exp"

4. Generate standalone LIT files:
   $ python -m test.d2m_jit.pattern_tests.lit_generator
"""
    )

    print_section("Adding Tests to a Pattern")
    print(
        """
In your pattern file (tools/d2m-jit/patterns/my_pattern.py):

    PATTERN_TEST_METADATA = {
        "pattern_name": "my_pattern",
        "description": "What this pattern does",

        "lit_tests": [
            {
                "name": "test_case",
                "module_text": '''module { ... }''',
                "file_checks": ["CHECK: ...", ...],
            }
        ],

        "e2e_tests": [
            {
                "name": "test_my_pattern_on_device",
                "kernel_fn": my_kernel,
                "input_generator": lambda: {"x": torch.rand(32, 32)},
                "reference_fn": lambda x: torch.exp(x),
                "layout_config": {...},
                "kernel_args": {...},
            }
        ],
    }

See PATTERN_TEMPLATE.py for a complete annotated example.
"""
    )

    print_section("Documentation Map")

    docs = [
        ("README.md", "Start here - complete overview and examples"),
        ("QUICK_REFERENCE.md", "Cheat sheet for common tasks"),
        ("ARCHITECTURE.md", "Visual architecture diagrams"),
        ("PATTERN_TEMPLATE.py", "Annotated template to copy"),
        ("REFACTORING_SUMMARY.md", "Detailed before/after comparison"),
        ("IMPLEMENTATION_SUMMARY.md", "High-level summary (this file)"),
    ]

    for filename, description in docs:
        print(f"  • {filename:30s} - {description}")

    print_section("Current Status")
    print(
        """
  ✓ Framework implemented and functional
  ✓ Discovery and validation working (tested)
  ✓ Two example patterns migrated
  ✓ Complete documentation provided
  ✓ Ready for testing with full d2m_jit environment
"""
    )

    print_section("Next Steps")
    print(
        """
1. Test with full environment:
   - Run pytest tests
   - Verify on-device E2E tests pass

2. Migrate remaining patterns:
   - Add PATTERN_TEST_METADATA to other pattern files

3. Generate LIT files for CI:
   - python -m test.d2m_jit.pattern_tests.lit_generator

4. Extend framework (optional):
   - Add golden data generation
   - Add performance benchmarks
   - Add negative test cases
"""
    )

    print("\n" + "=" * 70)
    print("For detailed information, see the documentation files above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
