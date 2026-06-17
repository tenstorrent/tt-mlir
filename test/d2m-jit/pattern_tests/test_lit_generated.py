# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""In-process LIT-style tests for patterns (generated from pattern metadata).

This module runs LIT-style tests directly in pytest without requiring
the LIT test runner. Each lit_test entry in pattern metadata becomes
a test that:

1. Parses the MLIR module from module_text
2. Applies d2m.apply_patterns()
3. Verifies the module
4. Checks the output against FileCheck-style patterns

This is useful for quick iteration without regenerating files or running
the full LIT test suite.
"""

import pytest
import re
from ttmlir import ir
import d2m_jit as d2m
from d2m_jit._src.rewrite import _registry
from .discovery import discover_all_pattern_tests


# Discover all pattern tests at module load time
_ALL_PATTERN_METADATA = discover_all_pattern_tests()


def _parse_filecheck_pattern(check_line: str) -> tuple[str, str, bool]:
    """Parse a FileCheck directive.

    Args:
        check_line: A FileCheck directive like "CHECK: some pattern" or "CHECK-NOT: pattern"

    Returns:
        Tuple of (check_type, pattern, is_regex) where:
        - check_type is "CHECK", "CHECK-NOT", "CHECK-LABEL", etc.
        - pattern is the text/regex to match
        - is_regex indicates if pattern should be treated as regex
    """
    # Match CHECK, CHECK-NOT, CHECK-LABEL, etc.
    match = re.match(r"^CHECK(-\w+)?:\s*(.*)$", check_line)
    if not match:
        raise ValueError(f"Invalid FileCheck pattern: {check_line}")

    check_type = match.group(1) or ""  # e.g., "-NOT", "-LABEL", or ""
    pattern = match.group(2).strip()

    # Convert {{.*}} style patterns to regex
    # This is a simplified version - full FileCheck is more complex
    is_regex = "{{" in pattern or "[[" in pattern
    if is_regex:
        # Convert {{...}} to regex capturing groups
        pattern = re.sub(r"\{\{[^}]*\}\}", r".*", pattern)
        pattern = re.sub(r"\[\[[^\]]*\]\]", r".*", pattern)
        # Escape other regex special chars that aren't part of our pattern
        # (This is simplified - real FileCheck is more sophisticated)

    return (check_type, pattern, is_regex)


def _check_output(output: str, file_checks: list[str]) -> tuple[bool, str]:
    """Verify output against FileCheck-style patterns.

    Args:
        output: The actual output string to check
        file_checks: List of FileCheck directive strings

    Returns:
        Tuple of (success, error_message)
    """
    lines = output.split("\n")
    current_line = 0

    for check in file_checks:
        check_type, pattern, is_regex = _parse_filecheck_pattern(check)

        if check_type == "-LABEL":
            # CHECK-LABEL: must match, resets position
            found = False
            for i, line in enumerate(lines):
                if is_regex:
                    if re.search(pattern, line):
                        current_line = i + 1
                        found = True
                        break
                else:
                    if pattern in line:
                        current_line = i + 1
                        found = True
                        break

            if not found:
                return False, f"CHECK-LABEL failed: pattern '{pattern}' not found"

        elif check_type == "-NOT":
            # CHECK-NOT: must not match in remaining lines
            for line in lines[current_line:]:
                if is_regex:
                    if re.search(pattern, line):
                        return (
                            False,
                            f"CHECK-NOT failed: pattern '{pattern}' found in: {line}",
                        )
                else:
                    if pattern in line:
                        return (
                            False,
                            f"CHECK-NOT failed: pattern '{pattern}' found in: {line}",
                        )

        else:  # plain CHECK
            # CHECK: must match in remaining lines
            found = False
            for i in range(current_line, len(lines)):
                line = lines[i]
                if is_regex:
                    if re.search(pattern, line):
                        current_line = i + 1
                        found = True
                        break
                else:
                    if pattern in line:
                        current_line = i + 1
                        found = True
                        break

            if not found:
                return (
                    False,
                    f"CHECK failed: pattern '{pattern}' not found after line {current_line}",
                )

    return True, ""


def _generate_lit_test_ids():
    """Generate test IDs for parameterization."""
    ids = []
    for metadata in _ALL_PATTERN_METADATA:
        pattern_name = metadata.get("pattern_name", "unknown")
        for lit_test in metadata.get("lit_tests", []):
            test_name = lit_test.get("name", "test")
            ids.append(f"{pattern_name}::{test_name}")
    return ids


def _generate_lit_test_params():
    """Generate test parameters from all pattern metadata."""
    params = []
    for metadata in _ALL_PATTERN_METADATA:
        for lit_test in metadata.get("lit_tests", []):
            params.append((metadata, lit_test))
    return params


@pytest.mark.parametrize(
    "pattern_metadata,lit_test_config",
    _generate_lit_test_params(),
    ids=_generate_lit_test_ids(),
)
def test_pattern_lit_style(pattern_metadata, lit_test_config):
    """Run a LIT-style pattern test in-process.

    Args:
        pattern_metadata: The full PATTERN_TEST_METADATA dict from a pattern file
        lit_test_config: A single lit_test dict from that metadata
    """
    # Clear and load the pattern
    _registry.clear()
    pattern_module = pattern_metadata["_module"]

    # Create MLIR context
    ctx = ir.Context()
    ctx.load_all_available_dialects()

    # Parse the test module
    module_text = lit_test_config["module_text"]
    mod = ir.Module.parse(module_text, ctx)

    # Apply patterns
    d2m.apply_patterns(mod)

    # Verify
    mod.operation.verify()

    # Get output
    output = str(mod)

    # Check against FileCheck patterns
    file_checks = lit_test_config["file_checks"]
    success, error_msg = _check_output(output, file_checks)

    if not success:
        print("\n=== Module Output ===")
        print(output)
        print("\n=== Failed Check ===")
        pytest.fail(error_msg)
