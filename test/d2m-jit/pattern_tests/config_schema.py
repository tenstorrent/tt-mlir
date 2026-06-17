# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema and models for external pattern test files.

This module defines the structure for YAML/TOML test configuration files
that live alongside pattern definition files.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import importlib.util


@dataclass
class LitTestConfig:
    """Configuration for a LIT-style pattern rewrite test.

    Attributes:
        name: Unique test case identifier
        module_text: MLIR module IR to parse and transform
        file_checks: List of FileCheck directive strings
        description: Optional explanation of what this test validates
    """

    name: str
    module_text: str
    file_checks: List[str]
    description: str = ""


@dataclass
class LayoutConfig:
    """Configuration for d2m.Layout construction.

    Attributes:
        shape: Tensor shape tuple (e.g., (32, 32))
        dtype: Data type name (e.g., "float32")
        block_shape: Block shape for tiling
        grid_shape: Grid shape for device distribution
        tiled: Whether to use tiled layout
        memory_space: Memory space (e.g., "L1", "DRAM")
    """

    shape: tuple
    dtype: str = "float32"
    block_shape: List[int] = field(default_factory=lambda: [1, 1])
    grid_shape: List[int] = field(default_factory=lambda: [1, 1])
    tiled: bool = True
    memory_space: str = "L1"


@dataclass
class InputConfig:
    """Configuration for generating test input tensors.

    Attributes:
        name: Parameter name
        shape: Tensor shape
        generator: Type of random generator (e.g., "uniform", "normal", "randn")
        dtype: PyTorch dtype string
        range_min: Minimum value for uniform distribution
        range_max: Maximum value for uniform distribution
        mean: Mean for normal distribution
        std: Standard deviation for normal distribution
    """

    name: str
    shape: tuple
    generator: str = "uniform"
    dtype: str = "float32"
    range_min: float = -1.0
    range_max: float = 1.0
    mean: float = 0.0
    std: float = 1.0


@dataclass
class E2ETestConfig:
    """Configuration for an end-to-end pattern kernel test.

    Attributes:
        name: Test function name
        description: Human-readable test description
        kernel: Name of the kernel function (string reference)
        inputs: List of input configurations
        reference: Python expression for expected output (e.g., "torch.exp(x)")
        layout: Layout configuration
        kernel_args: Additional keyword arguments for kernel call
        pcc_threshold: PCC threshold for correctness check
        seed: Random seed for reproducibility
    """

    name: str
    kernel: str  # String reference to kernel function
    inputs: List[InputConfig]
    reference: str  # Python expression string
    layout: LayoutConfig
    description: str = ""
    kernel_args: Dict[str, Any] = field(default_factory=dict)
    pcc_threshold: float = 0.99
    seed: int = 0


@dataclass
class PatternTestConfig:
    """Top-level configuration for a pattern's tests.

    Attributes:
        pattern_name: Unique identifier for the pattern
        pattern_module: Python module path (e.g., "eltwise_exp_to_kernel")
        description: Human-readable pattern description
        lit_tests: List of LIT test configurations
        e2e_tests: List of E2E test configurations
        tags: Optional tags for filtering/categorization
    """

    pattern_name: str
    pattern_module: str
    description: str = ""
    lit_tests: List[LitTestConfig] = field(default_factory=list)
    e2e_tests: List[E2ETestConfig] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Runtime fields (not in YAML)
    _config_file: Optional[Path] = field(default=None, repr=False, compare=False)
    _pattern_module_ref: Optional[Any] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding runtime fields."""
        result = asdict(self)
        result.pop("_config_file", None)
        result.pop("_pattern_module_ref", None)
        return result

    def get_kernel_function(self, kernel_name: str) -> Callable:
        """Resolve kernel function by name from the pattern module.

        Args:
            kernel_name: Name of the kernel function

        Returns:
            The actual callable kernel function

        Raises:
            AttributeError: If kernel not found in pattern module
        """
        if self._pattern_module_ref is None:
            raise RuntimeError(f"Pattern module not loaded for {self.pattern_name}")

        if not hasattr(self._pattern_module_ref, kernel_name):
            raise AttributeError(
                f"Kernel '{kernel_name}' not found in module {self.pattern_module}"
            )

        return getattr(self._pattern_module_ref, kernel_name)

    def load_pattern_module(self, patterns_dir: Path):
        """Load the pattern module and store reference.

        Args:
            patterns_dir: Directory containing pattern modules
        """
        if self._pattern_module_ref is not None:
            return  # Already loaded

        pattern_file = patterns_dir / f"{self.pattern_module}.py"
        if not pattern_file.exists():
            raise FileNotFoundError(f"Pattern module not found: {pattern_file}")

        spec = importlib.util.spec_from_file_location(self.pattern_module, pattern_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {pattern_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._pattern_module_ref = module


def validate_config(config: PatternTestConfig) -> List[str]:
    """Validate a pattern test configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if not config.pattern_name:
        errors.append("pattern_name is required")

    if not config.pattern_module:
        errors.append("pattern_module is required")

    # Validate LIT tests
    for i, lit_test in enumerate(config.lit_tests):
        if not lit_test.name:
            errors.append(f"lit_tests[{i}]: name is required")
        if not lit_test.module_text:
            errors.append(f"lit_tests[{i}]: module_text is required")
        if not lit_test.file_checks:
            errors.append(f"lit_tests[{i}]: file_checks cannot be empty")

    # Validate E2E tests
    for i, e2e_test in enumerate(config.e2e_tests):
        if not e2e_test.name:
            errors.append(f"e2e_tests[{i}]: name is required")
        if not e2e_test.kernel:
            errors.append(f"e2e_tests[{i}]: kernel is required")
        if not e2e_test.inputs:
            errors.append(f"e2e_tests[{i}]: inputs cannot be empty")
        if not e2e_test.reference:
            errors.append(f"e2e_tests[{i}]: reference is required")

        # Check input names are unique
        input_names = [inp.name for inp in e2e_test.inputs]
        if len(input_names) != len(set(input_names)):
            errors.append(f"e2e_tests[{i}]: duplicate input names")

    return errors
