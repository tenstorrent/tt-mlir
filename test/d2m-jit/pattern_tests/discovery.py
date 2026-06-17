# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pattern test discovery and metadata loading utilities.

Scans tools/d2m-jit/patterns/ for:
1. Pattern files with PATTERN_TEST_METADATA (legacy dict-based)
2. External .test.yaml configuration files (new YAML-based)

Provides unified interface for accessing test configurations from both sources.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .yaml_loader import discover_yaml_configs, load_yaml_config
from .config_schema import PatternTestConfig


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


def discover_all_pattern_tests() -> List[Union[Dict[str, Any], PatternTestConfig]]:
    """Discover all pattern test metadata from both sources.

    Discovers tests from:
    1. YAML config files (*.test.yaml) - preferred method
    2. Legacy PATTERN_TEST_METADATA dicts in pattern files

    Returns:
        List of test configurations (mix of PatternTestConfig and legacy dicts)
    """
    patterns_dir = get_patterns_dir()
    all_metadata = []

    # First, discover YAML configs
    yaml_configs = discover_yaml_configs(patterns_dir)
    all_metadata.extend(yaml_configs)

    # Track which patterns have YAML configs to avoid duplicates
    yaml_pattern_names = {config.pattern_name for config in yaml_configs}

    # Then, discover legacy dict-based configs (only if no YAML exists)
    pattern_files = discover_pattern_modules()
    for pattern_file in pattern_files:
        # Check if this pattern has a YAML config
        yaml_file = pattern_file.with_suffix(".test.yaml")
        if yaml_file.exists():
            # Skip legacy dict - YAML takes precedence
            continue

        # Load legacy dict-based metadata
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
        if isinstance(metadata, PatternTestConfig):
            if metadata.pattern_name == pattern_name:
                return metadata
        elif metadata.get("pattern_name") == pattern_name:
            return metadata
    return None


def is_yaml_config(metadata: Union[Dict[str, Any], PatternTestConfig]) -> bool:
    """Check if metadata is from a YAML config file.

    Args:
        metadata: Either a PatternTestConfig or legacy dict

    Returns:
        True if from YAML, False if legacy dict
    """
    return isinstance(metadata, PatternTestConfig)


def get_pattern_name(metadata: Union[Dict[str, Any], PatternTestConfig]) -> str:
    """Get pattern name from either config type.

    Args:
        metadata: Either a PatternTestConfig or legacy dict

    Returns:
        Pattern name string
    """
    if isinstance(metadata, PatternTestConfig):
        return metadata.pattern_name
    return metadata.get("pattern_name", "unknown")


def get_lit_tests(metadata: Union[Dict[str, Any], PatternTestConfig]) -> List[Any]:
    """Get LIT tests from either config type.

    Args:
        metadata: Either a PatternTestConfig or legacy dict

    Returns:
        List of LIT test configurations
    """
    if isinstance(metadata, PatternTestConfig):
        return metadata.lit_tests
    return metadata.get("lit_tests", [])


def get_e2e_tests(metadata: Union[Dict[str, Any], PatternTestConfig]) -> List[Any]:
    """Get E2E tests from either config type.

    Args:
        metadata: Either a PatternTestConfig or legacy dict

    Returns:
        List of E2E test configurations
    """
    if isinstance(metadata, PatternTestConfig):
        return metadata.e2e_tests
    return metadata.get("e2e_tests", [])
    all_metadata = discover_all_pattern_tests()
    for metadata in all_metadata:
        if metadata.get("pattern_name") == pattern_name:
            return metadata
    return None
