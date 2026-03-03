# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simple test model for PythonModelRunner API testing (no ttnn dependency)."""


def forward(inputs, device):
    """Simple forward function that just echoes inputs.

    Args:
        inputs: List of input values
        device: Device info

    Returns:
        Modified list
    """
    print(f"[Python] Received {len(inputs)} inputs")
    print(f"[Python] Device: {device}")
    print("[Python] Test successful!")
    return inputs
