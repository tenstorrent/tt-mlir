# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simple test model for PythonModelRunner API testing."""

import ttnn


def forward(inputs, device):
    """Simple forward function that adds two tensors.

    Args:
        inputs: List of input tensors
        device: The mesh device

    Returns:
        List containing the result tensor
    """
    print(f"[Python] Received {len(inputs)} inputs")
    print(f"[Python] Device: {device}")

    if len(inputs) < 2:
        raise ValueError("Expected at least 2 input tensors")

    # Simple add operation
    result = ttnn.add(inputs[0], inputs[1])

    print("[Python] Computed result")
    return [result]
