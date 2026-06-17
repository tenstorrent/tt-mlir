# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LIT test generator for pattern tests.

This script generates standalone LIT test files from pattern metadata.
Can be run manually to generate/update the LIT test files, or the tests
can be run dynamically via the test_lit_generated.py module.
"""

from pathlib import Path
from typing import Dict, Any
from .discovery import discover_all_pattern_tests, get_patterns_dir


def generate_lit_test_file(metadata: Dict[str, Any], output_dir: Path) -> Path:
    """Generate a LIT test file from pattern metadata.

    Args:
        metadata: Pattern metadata dict containing lit_tests
        output_dir: Directory to write the generated test file

    Returns:
        Path to the generated test file
    """
    pattern_name = metadata["pattern_name"]
    pattern_file = metadata["_pattern_file"]
    description = metadata.get("description", "")
    lit_tests = metadata.get("lit_tests", [])

    if not lit_tests:
        return None

    # Generate test file content
    lines = [
        "# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC",
        "#",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        "# RUN: %python %s 2>&1 | FileCheck %s",
        "# REQUIRES: d2m-jit",
        "",
        f'"""Generated LIT test for {pattern_name} pattern.',
        "",
        f"{description}",
        "",
        "This file is auto-generated from PATTERN_TEST_METADATA in:",
        f"  {pattern_file.relative_to(get_patterns_dir().parent.parent)}",
        "",
        "To regenerate, run:",
        "  python -m test.d2m_jit.pattern_tests.lit_generator",
        '"""',
        "",
        "from ttmlir import ir",
        "import d2m_jit as d2m",
        "from d2m_jit._src.rewrite import _registry",
        "",
        "# Load the pattern (registers via @d2m.pattern on import).",
        "_registry.clear()",
        f"import {pattern_file.stem}  # noqa: F401",
        "",
        "",
        "ctx = ir.Context()",
        "ctx.load_all_available_dialects()",
        "",
    ]

    # Generate a test case for each lit_test
    for i, lit_test in enumerate(lit_tests):
        test_name = lit_test["name"]
        module_text = lit_test["module_text"].strip()
        file_checks = lit_test["file_checks"]
        test_description = lit_test.get("description", "")

        lines.extend(
            [
                "# " + "=" * 70,
                f"# {test_name}",
            ]
        )
        if test_description:
            lines.append(f"# {test_description}")
        lines.extend(
            [
                "# " + "=" * 70,
                f"{test_name}_mod = ir.Module.parse(",
                '    """',
            ]
        )

        # Add module text (indented)
        for line in module_text.split("\n"):
            lines.append(line)

        lines.extend(
            [
                '""",',
                "    ctx,",
                ")",
                "",
                f"d2m.apply_patterns({test_name}_mod)",
                f"{test_name}_mod.operation.verify()",
                f'print("=== {test_name} ===")',
                f"print({test_name}_mod)",
                "",
            ]
        )

        # Add FileCheck directives
        lines.append(f"# CHECK-LABEL: === {test_name} ===")
        for check in file_checks:
            lines.append(f"# {check}")
        lines.append("")

    # Write file
    output_file = output_dir / f"{pattern_name}_pattern_generated.py"
    output_file.write_text("\n".join(lines))

    return output_file


def generate_all_lit_tests(output_dir: Path = None):
    """Generate LIT test files for all patterns with metadata.

    Args:
        output_dir: Directory to write generated files (default: test/d2m-jit/lit_generated/)
    """
    if output_dir is None:
        test_dir = Path(__file__).parent.parent
        output_dir = test_dir / "lit_generated"

    output_dir.mkdir(exist_ok=True)

    # Create/update lit.local.cfg
    lit_cfg = output_dir / "lit.local.cfg"
    lit_cfg.write_text(
        "# Generated LIT tests from pattern metadata\n"
        "# This directory contains auto-generated test files.\n"
    )

    all_metadata = discover_all_pattern_tests()
    generated_files = []

    for metadata in all_metadata:
        if metadata.get("lit_tests"):
            output_file = generate_lit_test_file(metadata, output_dir)
            if output_file:
                generated_files.append(output_file)
                print(f"Generated: {output_file}")

    print(f"\nGenerated {len(generated_files)} LIT test file(s) in {output_dir}")
    return generated_files


if __name__ == "__main__":
    generate_all_lit_tests()
