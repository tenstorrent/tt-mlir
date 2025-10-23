# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional


class Stream:
    def __init__(self, tensor, num_buffers: Optional[int] = None):
        assert hasattr(
            tensor, "_global_name"
        ), "Stream must be created from a top level tensor argument"
        self.name = tensor._global_name
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        assert num_buffers is None, "Unsupported"
        self.num_buffers = num_buffers
