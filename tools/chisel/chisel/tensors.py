# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TensorPool: dict mapping SSA names to golden torch.Tensor values."""


class TensorPool(dict):
    """Dict mapping SSA name (str) -> GoldenMapTensor (golden output).

    Single-device programs store GoldenMapTensor({0: tensor}, (1,1)).
    Multi-device programs store one shard per device keyed by device index.
    """
