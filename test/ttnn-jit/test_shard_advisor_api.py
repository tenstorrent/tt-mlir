# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn
import torch

import ttnn_jit
from ttnn_jit._src.shard_advisor import AdvisorReport
from utils import create_dram_tensor


@pytest.mark.forked
def test_shard_advisor_decorator_end_to_end(device):
    @ttnn_jit.shard_advisor(optimization_level=2)
    def model(a, b):
        c = ttnn.matmul(a, b)
        return ttnn.add(c, c)

    a = create_dram_tensor(device, (256, 256), torch.bfloat16)
    b = create_dram_tensor(device, (256, 256), torch.bfloat16)

    report = model(a, b)
    assert isinstance(report, AdvisorReport)
    assert "L1 Sharding Advisor" in report.text
    assert hasattr(model, "advisor")
