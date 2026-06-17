# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime utilities for executing tests from YAML configurations.

This module provides functions to convert YAML test configs into executable
Python code (generating inputs, evaluating reference expressions, etc.).
"""

import torch
from typing import Dict, Any, Callable
import d2m_jit as d2m

from .config_schema import InputConfig, E2ETestConfig, LayoutConfig


def generate_input_tensor(config: InputConfig) -> torch.Tensor:
    """Generate an input tensor according to the configuration.

    Args:
        config: Input configuration specifying shape, generator type, etc.

    Returns:
        Generated PyTorch tensor
    """
    shape = config.shape
    dtype = getattr(torch, config.dtype)

    if config.generator == "uniform":
        # Uniform distribution in [range_min, range_max]
        tensor = torch.rand(*shape, dtype=dtype)
        tensor = tensor * (config.range_max - config.range_min) + config.range_min

    elif config.generator == "normal":
        # Normal distribution with mean and std
        tensor = torch.randn(*shape, dtype=dtype) * config.std + config.mean

    elif config.generator == "randn":
        # Standard normal (mean=0, std=1)
        tensor = torch.randn(*shape, dtype=dtype)

    elif config.generator == "ones":
        tensor = torch.ones(*shape, dtype=dtype)

    elif config.generator == "zeros":
        tensor = torch.zeros(*shape, dtype=dtype)

    elif config.generator == "arange":
        # Sequential values reshaped to target shape
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        tensor = torch.arange(total_elements, dtype=dtype).reshape(shape)

    else:
        raise ValueError(f"Unknown generator type: {config.generator}")

    return tensor


def create_input_generator(
    configs: list[InputConfig],
) -> Callable[[], Dict[str, torch.Tensor]]:
    """Create a function that generates inputs according to configs.

    Args:
        configs: List of input configurations

    Returns:
        Function that returns dict mapping input names to tensors
    """

    def generator():
        return {config.name: generate_input_tensor(config) for config in configs}

    return generator


def create_reference_function(expression: str, input_names: list[str]) -> Callable:
    """Create a reference function from a Python expression string.

    Args:
        expression: Python expression (e.g., "torch.exp(x)" or "torch.exp(a + b)")
        input_names: List of input parameter names

    Returns:
        Function that takes inputs as kwargs and returns expected output
    """
    # Create a function that evaluates the expression
    # Build the function dynamically
    param_str = ", ".join(input_names)
    func_code = f"lambda {param_str}: {expression}"

    # Evaluate in a namespace with torch available
    namespace = {"torch": torch}
    try:
        func = eval(func_code, namespace)
    except Exception as e:
        raise ValueError(f"Invalid reference expression '{expression}': {e}")

    return func


def create_layout(config: LayoutConfig) -> d2m.Layout:
    """Create a d2m.Layout from configuration.

    Args:
        config: Layout configuration

    Returns:
        d2m.Layout object
    """
    # Convert dtype string to d2m dtype
    dtype_map = {
        "float32": d2m.float32,
        "float16": d2m.float16,
        "bfloat16": d2m.bfloat16,
        "int32": d2m.int32,
        "int8": d2m.int8,
        "uint8": d2m.uint8,
    }

    dtype = dtype_map.get(config.dtype, d2m.float32)

    return d2m.Layout(
        shape=config.shape,
        dtype=dtype,
        block_shape=config.block_shape,
        grid_shape=config.grid_shape,
        tiled=config.tiled,
    )


def convert_kernel_args(kernel_args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML kernel args to Python types.

    Args:
        kernel_args: Dictionary of kernel arguments from YAML

    Returns:
        Converted dictionary with proper Python types
    """
    result = {}
    for key, value in kernel_args.items():
        # Convert lists to tuples where appropriate (e.g., grid)
        if key == "grid" and isinstance(value, list):
            result[key] = tuple(value)
        else:
            result[key] = value
    return result


def create_e2e_test_runner(config: E2ETestConfig, pattern_config) -> Callable:
    """Create a test runner function from E2E test configuration.

    Args:
        config: E2E test configuration
        pattern_config: Parent PatternTestConfig for resolving kernel function

    Returns:
        Function that runs the test (takes no args, asserts on failure)
    """
    # Get kernel function
    kernel_fn = pattern_config.get_kernel_function(config.kernel)

    # Create input generator
    input_generator = create_input_generator(config.inputs)

    # Create reference function
    input_names = [inp.name for inp in config.inputs]
    reference_fn = create_reference_function(config.reference, input_names)

    # Create layout
    layout = create_layout(config.layout)

    # Convert kernel args
    kernel_args = convert_kernel_args(config.kernel_args)

    def run_test():
        """Execute the E2E test."""
        # Import here to avoid circular dependency
        from ..utils import assert_pcc

        # Set seed
        torch.manual_seed(config.seed)

        # Generate inputs
        inputs = input_generator()

        # Compute expected output
        expected = reference_fn(**inputs)

        # Convert inputs to device
        device_inputs = {}
        for name, tensor in inputs.items():
            device_inputs[name] = d2m.to_layout(tensor, layout)

        # Create output tensor
        out_d = d2m.empty(layout)

        # Call kernel
        input_tensors = list(device_inputs.values())
        kernel_fn(*input_tensors, out_d, **kernel_args)

        # Get result from device
        actual = out_d.to_host()

        # Assert PCC
        assert_pcc(expected, actual, threshold=config.pcc_threshold)

    return run_test
