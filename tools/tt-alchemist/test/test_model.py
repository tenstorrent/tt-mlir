# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simple test model for PythonModelRunner API testing."""

import ttnn


def forward(inputs, device):
    """Simple forward function that adds two tensors.

    Args:
        inputs: List of PyCapsule objects wrapping ttnn::Tensor pointers
        device: PyCapsule object wrapping ttnn::MeshDevice pointer

    Returns:
        List containing the result tensor (as PyCapsule or ttnn.Tensor)
    """
    print(f"[Python] Received {len(inputs)} inputs")
    print(f"[Python] Input types: {[type(x).__name__ for x in inputs]}")
    print(f"[Python] Device type: {type(device).__name__}")

    # Inputs should be PyCapsules - we can't unwrap them in Python
    # But we can work with them if Python code knows how to handle capsules
    # For now, just pass them through - they contain the C++ pointers

    # Actually, the Python ttnn module should provide a way to work with these
    # Let's try to see if we can extract the tensors

    # If they're already ttnn.Tensor (from tt-metal's bindings), use directly
    if isinstance(inputs[0], ttnn.Tensor):
        print("[Python] Inputs are already ttnn.Tensor objects")
        tensor1 = inputs[0]
        tensor2 = inputs[1]
        mesh_device = device
    else:
        # They're capsules - we can't use them in Python ttnn operations
        print(f"[Python] Inputs are PyCapsules - cannot unwrap in Python")
        print(
            f"[Python] This approach requires C++ to create actual Python ttnn.Tensor objects"
        )
        # Return the capsules as-is
        return inputs

    # Simple add operation
    result = ttnn.add(tensor1, tensor2)

    print("[Python] Computed result")
    return [result]
