# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TensorPool: dict mapping SSA names to golden torch.Tensor values."""


class TensorPool(dict):
    """Dict mapping SSA name (str) -> torch.Tensor (golden output)."""
