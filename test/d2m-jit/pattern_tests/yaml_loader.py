# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""YAML configuration file loader for pattern tests.

This module handles loading and parsing external YAML test configuration files.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

from .config_schema import (
    PatternTestConfig,
    LitTestConfig,
    E2ETestConfig,
    LayoutConfig,
    InputConfig,
    validate_config,
)


def load_yaml_config(config_file: Path) -> Optional[PatternTestConfig]:
    """Load a pattern test configuration from a YAML file.

    Args:
        config_file: Path to the .test.yaml file

    Returns:
        PatternTestConfig object if successful, None if file doesn't exist

    Raises:
        ValueError: If YAML is malformed or validation fails
        yaml.YAMLError: If YAML syntax is invalid
    """
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_file}: {e}")

    if data is None:
        return None

    # Parse the configuration
    try:
        config = _parse_config_dict(data)
        config._config_file = config_file

        # Validate
        errors = validate_config(config)
        if errors:
            raise ValueError(
                f"Validation errors in {config_file}:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        return config

    except Exception as e:
        raise ValueError(f"Error parsing {config_file}: {e}")


def _parse_config_dict(data: Dict[str, Any]) -> PatternTestConfig:
    """Parse a dictionary into a PatternTestConfig.

    Args:
        data: Dictionary loaded from YAML

    Returns:
        PatternTestConfig object
    """
    # Parse LIT tests
    lit_tests = []
    for lit_data in data.get("lit_tests", []):
        lit_tests.append(
            LitTestConfig(
                name=lit_data["name"],
                module_text=lit_data["module_text"],
                file_checks=lit_data["file_checks"],
                description=lit_data.get("description", ""),
            )
        )

    # Parse E2E tests
    e2e_tests = []
    for e2e_data in data.get("e2e_tests", []):
        # Parse inputs
        inputs = []
        for inp_data in e2e_data["inputs"]:
            inputs.append(
                InputConfig(
                    name=inp_data["name"],
                    shape=tuple(inp_data["shape"]),
                    generator=inp_data.get("generator", "uniform"),
                    dtype=inp_data.get("dtype", "float32"),
                    range_min=inp_data.get("range_min", -1.0),
                    range_max=inp_data.get("range_max", 1.0),
                    mean=inp_data.get("mean", 0.0),
                    std=inp_data.get("std", 1.0),
                )
            )

        # Parse layout
        layout_data = e2e_data["layout"]
        layout = LayoutConfig(
            shape=tuple(layout_data["shape"]),
            dtype=layout_data.get("dtype", "float32"),
            block_shape=layout_data.get("block_shape", [1, 1]),
            grid_shape=layout_data.get("grid_shape", [1, 1]),
            tiled=layout_data.get("tiled", True),
            memory_space=layout_data.get("memory_space", "L1"),
        )

        e2e_tests.append(
            E2ETestConfig(
                name=e2e_data["name"],
                kernel=e2e_data["kernel"],
                inputs=inputs,
                reference=e2e_data["reference"],
                layout=layout,
                description=e2e_data.get("description", ""),
                kernel_args=e2e_data.get("kernel_args", {}),
                pcc_threshold=e2e_data.get("pcc_threshold", 0.99),
                seed=e2e_data.get("seed", 0),
            )
        )

    return PatternTestConfig(
        pattern_name=data["pattern_name"],
        pattern_module=data["pattern_module"],
        description=data.get("description", ""),
        lit_tests=lit_tests,
        e2e_tests=e2e_tests,
        tags=data.get("tags", []),
    )


def discover_yaml_configs(
    patterns_dir: Path, load_modules: bool = True
) -> List[PatternTestConfig]:
    """Discover all .test.yaml files in the patterns directory.

    Args:
        patterns_dir: Directory containing pattern modules
        load_modules: Whether to load pattern modules (requires full environment)

    Returns:
        List of PatternTestConfig objects
    """
    configs = []

    for yaml_file in patterns_dir.glob("*.test.yaml"):
        try:
            config = load_yaml_config(yaml_file)
            if config:
                # Optionally load the pattern module reference
                if load_modules:
                    try:
                        config.load_pattern_module(patterns_dir)
                    except Exception as e:
                        print(
                            f"Warning: Could not load pattern module for {yaml_file.name}: {e}"
                        )
                configs.append(config)
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file}: {e}")

    return configs


def generate_yaml_template(pattern_name: str, pattern_module: str) -> str:
    """Generate a YAML template for a new pattern.

    Args:
        pattern_name: Unique pattern identifier
        pattern_module: Python module name (without .py)

    Returns:
        YAML template string
    """
    template = f"""# Pattern test configuration for {pattern_name}
# See test/d2m-jit/pattern_tests/README.md for full documentation

pattern_name: {pattern_name}
pattern_module: {pattern_module}
description: |
  Brief description of what this pattern does

tags:
  - eltwise
  - fusion

# LIT tests verify that IR rewriting works correctly
lit_tests:
  - name: {pattern_name}_positive
    description: Pattern matches and rewrites successfully
    module_text: |
      module {{
        func.func @forward(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {{
          %r = "ttir.some_op"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
          return %r : tensor<32x32xf32>
        }}
      }}
    file_checks:
      - "CHECK-LABEL: func.func @forward"
      - "CHECK-NOT:    ttir.some_op"
      - "CHECK:        d2m.generic"

# E2E tests verify kernel correctness on device
e2e_tests:
  - name: test_{pattern_name}_kernel_on_device
    description: Kernel produces correct results on device
    kernel: my_kernel_function_name  # Must exist in pattern module

    inputs:
      - name: x
        shape: [32, 32]
        generator: uniform  # or 'normal', 'randn', 'ones', 'zeros'
        dtype: float32
        range_min: -1.0
        range_max: 1.0

    reference: "torch.exp(x)"  # Python expression for expected output

    layout:
      shape: [32, 32]
      dtype: float32
      block_shape: [1, 1]
      grid_shape: [1, 1]
      tiled: true
      memory_space: L1

    kernel_args:
      m_blocks: 1
      n_blocks: 1
      grid: [1, 1]

    pcc_threshold: 0.99
    seed: 0
"""
    return template
