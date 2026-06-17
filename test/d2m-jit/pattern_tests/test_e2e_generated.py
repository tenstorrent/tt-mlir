# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest-based E2E tests for pattern kernels (generated from pattern metadata).

This module dynamically generates pytest test functions based on test
configurations from:
1. External YAML config files (*.test.yaml) - preferred method
2. Legacy PATTERN_TEST_METADATA dicts in pattern files

Each e2e_test entry becomes a parameterized test that:
1. Generates input tensors
2. Computes the expected output
3. Runs the kernel function on device
4. Asserts PCC between device output and expected output
"""

import pytest
import torch

import d2m_jit as d2m
from test.d2m_jit.utils import assert_pcc
from .discovery import (
    discover_all_pattern_tests,
    is_yaml_config,
    get_pattern_name,
    get_e2e_tests,
)
from .test_runtime import create_e2e_test_runner
from .config_schema import PatternTestConfig, E2ETestConfig


# Discover all pattern tests at module load time
_ALL_PATTERN_METADATA = discover_all_pattern_tests()


def _generate_test_ids():
    """Generate test IDs for parameterization."""
    ids = []
    for metadata in _ALL_PATTERN_METADATA:
        pattern_name = get_pattern_name(metadata)
        for e2e_test in get_e2e_tests(metadata):
            if isinstance(e2e_test, E2ETestConfig):
                test_name = e2e_test.name
            else:
                test_name = e2e_test.get("name", "test")
            ids.append(f"{pattern_name}::{test_name}")
    return ids


def _generate_test_params():
    """Generate test parameters from all pattern metadata."""
    params = []
    for metadata in _ALL_PATTERN_METADATA:
        for e2e_test in get_e2e_tests(metadata):
            params.append((metadata, e2e_test))
    return params


@pytest.mark.parametrize(
    "pattern_metadata,e2e_test_config",
    _generate_test_params(),
    ids=_generate_test_ids(),
)
def test_pattern_kernel_e2e(pattern_metadata, e2e_test_config):
    """Run a pattern kernel E2E test on device.

    This is a parametrized test that runs once for each e2e_test entry
    in all pattern test configurations (YAML or dict-based).

    Args:
        pattern_metadata: Either PatternTestConfig (YAML) or dict (legacy)
        e2e_test_config: Either E2ETestConfig (YAML) or dict (legacy)
    """
    # Check if this is a YAML config
    if isinstance(pattern_metadata, PatternTestConfig) and isinstance(
        e2e_test_config, E2ETestConfig
    ):
        # New YAML-based config - use runtime to execute
        test_runner = create_e2e_test_runner(e2e_test_config, pattern_metadata)
        test_runner()
    else:
        # Legacy dict-based config - use old execution path
        _run_legacy_e2e_test(pattern_metadata, e2e_test_config)


def _run_legacy_e2e_test(pattern_metadata, e2e_test_config):
    """Run E2E test using legacy dict-based config format.

    Args:
        pattern_metadata: Legacy dict from PATTERN_TEST_METADATA
        e2e_test_config: Legacy dict for a single e2e_test
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Extract test configuration
    kernel_fn = e2e_test_config["kernel_fn"]
    input_generator = e2e_test_config["input_generator"]
    reference_fn = e2e_test_config["reference_fn"]
    layout_config = e2e_test_config["layout_config"]
    kernel_args = e2e_test_config["kernel_args"]

    # Generate inputs
    inputs = input_generator()

    # Compute expected output
    expected = reference_fn(**inputs)

    # Create layout
    L = d2m.Layout(**layout_config)

    # Convert inputs to device
    device_inputs = {}
    for name, tensor in inputs.items():
        device_inputs[name] = d2m.to_layout(tensor, L)

    # Create output tensor
    out_d = d2m.empty(L)

    # Call kernel with inputs, output, and kernel args
    # The kernel signature is: kernel_fn(*inputs, out, *kernel_args)
    input_tensors = list(device_inputs.values())
    kernel_fn(*input_tensors, out_d, **kernel_args)

    # Get result from device
    actual = out_d.to_host()

    # Assert PCC
    assert_pcc(expected, actual)
