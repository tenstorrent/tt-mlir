# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration classes for pattern tests.

This module provides dataclasses for defining pattern test metadata with
type safety, validation, and IDE support. Replaces raw dictionary-based
configurations with structured, validated types.
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Any, Optional, Union
from pathlib import Path


@dataclass
class LitTestCase:
    """Configuration for a single LIT-style pattern rewrite test.

    Attributes:
        name: Unique identifier for this test case
        module_text: MLIR module text to parse and transform
        file_checks: List of FileCheck-style patterns to verify
        description: Optional human-readable test description
    """

    name: str
    module_text: str
    file_checks: List[str]
    description: str = ""

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.name:
            raise ValueError("LitTestCase name cannot be empty")
        if not self.module_text.strip():
            raise ValueError(f"LitTestCase '{self.name}' has empty module_text")
        if not self.file_checks:
            raise ValueError(f"LitTestCase '{self.name}' has no file_checks")

        # Validate file_checks format
        for check in self.file_checks:
            if not isinstance(check, str):
                raise TypeError(
                    f"LitTestCase '{self.name}' has non-string file_check: {check}"
                )
            if not check.startswith("CHECK"):
                raise ValueError(
                    f"LitTestCase '{self.name}' has invalid check pattern (must start with CHECK): {check}"
                )


@dataclass
class LayoutConfig:
    """Configuration for d2m.Layout creation.

    Attributes:
        shape: Tensor shape tuple (e.g., (32, 32))
        dtype: d2m dtype (e.g., d2m.float32)
        block_shape: Block shape for tiling [rows, cols]
        grid_shape: Grid shape for distribution [rows, cols]
        tiled: Whether to use tiled layout
        memory_space: Memory space (L1, DRAM, etc.)
    """

    shape: tuple
    dtype: Any  # d2m dtype, avoid hard dependency
    block_shape: List[int]
    grid_shape: List[int]
    tiled: bool = True
    memory_space: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for d2m.Layout(**config)."""
        result = {
            "shape": self.shape,
            "dtype": self.dtype,
            "block_shape": self.block_shape,
            "grid_shape": self.grid_shape,
            "tiled": self.tiled,
        }
        if self.memory_space is not None:
            result["memory_space"] = self.memory_space
        return result

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.shape) < 2:
            raise ValueError(
                f"LayoutConfig shape must have at least 2 dimensions: {self.shape}"
            )
        if len(self.block_shape) != 2:
            raise ValueError(
                f"LayoutConfig block_shape must be [rows, cols]: {self.block_shape}"
            )
        if len(self.grid_shape) != 2:
            raise ValueError(
                f"LayoutConfig grid_shape must be [rows, cols]: {self.grid_shape}"
            )


@dataclass
class E2ETestCase:
    """Configuration for an end-to-end pattern kernel test.

    Attributes:
        name: Unique test identifier (should start with 'test_')
        kernel_fn: The @d2m.kernel decorated function to test
        input_generator: Callable returning dict of input tensors
        reference_fn: Callable computing expected output from inputs
        layout_config: Layout configuration (LayoutConfig or dict)
        kernel_args: Additional kwargs passed to kernel function
        description: Optional human-readable test description
        pcc_threshold: PCC threshold for assertion (default 0.99)
        seed: Random seed for reproducibility (default 0)
    """

    name: str
    kernel_fn: Callable
    input_generator: Callable[[], Dict[str, Any]]
    reference_fn: Callable
    layout_config: Union[LayoutConfig, Dict[str, Any]]
    kernel_args: Dict[str, Any]
    description: str = ""
    pcc_threshold: float = 0.99
    seed: int = 0

    def get_layout_dict(self) -> Dict[str, Any]:
        """Get layout config as dict for d2m.Layout()."""
        if isinstance(self.layout_config, LayoutConfig):
            return self.layout_config.to_dict()
        return self.layout_config

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.name:
            raise ValueError("E2ETestCase name cannot be empty")
        if not self.name.startswith("test_"):
            raise ValueError(f"E2ETestCase name should start with 'test_': {self.name}")
        if not callable(self.kernel_fn):
            raise TypeError(f"E2ETestCase '{self.name}' kernel_fn is not callable")
        if not callable(self.input_generator):
            raise TypeError(
                f"E2ETestCase '{self.name}' input_generator is not callable"
            )
        if not callable(self.reference_fn):
            raise TypeError(f"E2ETestCase '{self.name}' reference_fn is not callable")
        if not isinstance(self.kernel_args, dict):
            raise TypeError(f"E2ETestCase '{self.name}' kernel_args must be a dict")
        if not 0 <= self.pcc_threshold <= 1:
            raise ValueError(
                f"E2ETestCase '{self.name}' pcc_threshold must be in [0, 1]: {self.pcc_threshold}"
            )

        # Validate layout config
        if isinstance(self.layout_config, LayoutConfig):
            self.layout_config.validate()


@dataclass
class PatternTestConfig:
    """Complete test configuration for a pattern.

    Attributes:
        pattern_name: Unique identifier for the pattern
        description: Human-readable description of what the pattern does
        lit_tests: List of LIT test cases
        e2e_tests: List of E2E test cases
        tags: Optional tags for filtering/categorization
        skip_reason: If set, all tests for this pattern will be skipped
    """

    pattern_name: str
    description: str
    lit_tests: List[LitTestCase] = field(default_factory=list)
    e2e_tests: List[E2ETestCase] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    skip_reason: Optional[str] = None

    # Internal fields (set by discovery)
    _module: Any = field(default=None, repr=False)
    _pattern_file: Optional[Path] = field(default=None, repr=False)

    def validate(self) -> None:
        """Validate the entire configuration."""
        if not self.pattern_name:
            raise ValueError("PatternTestConfig pattern_name cannot be empty")

        if not self.lit_tests and not self.e2e_tests:
            raise ValueError(f"Pattern '{self.pattern_name}' has no tests defined")

        # Validate all test cases
        for lit_test in self.lit_tests:
            lit_test.validate()

        for e2e_test in self.e2e_tests:
            e2e_test.validate()

        # Check for duplicate names
        lit_names = [t.name for t in self.lit_tests]
        if len(lit_names) != len(set(lit_names)):
            raise ValueError(
                f"Pattern '{self.pattern_name}' has duplicate LIT test names"
            )

        e2e_names = [t.name for t in self.e2e_tests]
        if len(e2e_names) != len(set(e2e_names)):
            raise ValueError(
                f"Pattern '{self.pattern_name}' has duplicate E2E test names"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility)."""
        result = asdict(self)
        # Remove internal fields
        result.pop("_module", None)
        result.pop("_pattern_file", None)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternTestConfig":
        """Create from dictionary (for backward compatibility)."""
        # Extract fields
        pattern_name = data.get("pattern_name", "")
        description = data.get("description", "")

        # Convert lit_tests
        lit_tests = []
        for lit_data in data.get("lit_tests", []):
            if isinstance(lit_data, LitTestCase):
                lit_tests.append(lit_data)
            else:
                lit_tests.append(LitTestCase(**lit_data))

        # Convert e2e_tests
        e2e_tests = []
        for e2e_data in data.get("e2e_tests", []):
            if isinstance(e2e_data, E2ETestCase):
                e2e_tests.append(e2e_data)
            else:
                # Convert layout_config if it's a dict
                if "layout_config" in e2e_data and isinstance(
                    e2e_data["layout_config"], dict
                ):
                    layout_dict = e2e_data["layout_config"]
                    if "shape" in layout_dict and "dtype" in layout_dict:
                        e2e_data["layout_config"] = LayoutConfig(**layout_dict)
                e2e_tests.append(E2ETestCase(**e2e_data))

        return cls(
            pattern_name=pattern_name,
            description=description,
            lit_tests=lit_tests,
            e2e_tests=e2e_tests,
            tags=data.get("tags", []),
            skip_reason=data.get("skip_reason"),
            _module=data.get("_module"),
            _pattern_file=data.get("_pattern_file"),
        )


# Type alias for backward compatibility
PatternTestMetadata = Union[PatternTestConfig, Dict[str, Any]]
