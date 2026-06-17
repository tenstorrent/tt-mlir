# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pattern test discovery and metadata loading utilities.

Scans tools/d2m-jit/patterns/ for pattern files with PATTERN_TEST_METADATA
and provides utilities to load and access test configurations.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_patterns_dir() -> Path:
    """Return the path to the patterns directory."""
    test_dir = Path(__file__).parent.parent
    repo_root = test_dir.parent.parent
    patterns_dir = repo_root / "tools" / "d2m-jit" / "patterns"
    return patterns_dir


def discover_pattern_modules() -> List[Path]:
    """Find all pattern definition files.

    Returns:
        List of paths to pattern .py files (excluding __init__.py)
    """
    patterns_dir = get_patterns_dir()
    if not patterns_dir.exists():
        return []

    pattern_files = []
    for py_file in patterns_dir.glob("*.py"):
        if py_file.name != "__init__.py" and not py_file.name.startswith("_"):
            pattern_files.append(py_file)

    return sorted(pattern_files)


def load_pattern_metadata(pattern_file: Path) -> Optional[Dict[str, Any]]:
    """Load PATTERN_TEST_METADATA from a pattern file.

    Args:
        pattern_file: Path to the pattern .py file

    Returns:
        The PATTERN_TEST_METADATA dict if present, None otherwise
    """
    module_name = f"_dynamic_pattern_{pattern_file.stem}"

    spec = importlib.util.spec_from_file_location(module_name, pattern_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Warning: Failed to load {pattern_file}: {e}")
        return None

    metadata = getattr(module, "PATTERN_TEST_METADATA", None)
    if metadata:
        # Add the module reference for kernel functions
        metadata["_module"] = module
        metadata["_pattern_file"] = pattern_file

    return metadata


def discover_all_pattern_tests() -> List[Dict[str, Any]]:
    """Discover all pattern test metadata.

    Returns:
        List of metadata dictionaries, one per pattern file with test metadata
    """
    pattern_files = discover_pattern_modules()
    all_metadata = []

    for pattern_file in pattern_files:
        metadata = load_pattern_metadata(pattern_file)
        if metadata:
            all_metadata.append(metadata)

    return all_metadata


def get_pattern_metadata_by_name(pattern_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific pattern by name.

    Args:
        pattern_name: The pattern_name field from PATTERN_TEST_METADATA

    Returns:
        The metadata dict if found, None otherwise
    """
    all_metadata = discover_all_pattern_tests()
    for metadata in all_metadata:
        if metadata.get("pattern_name") == pattern_name:
            return metadata
    return None
