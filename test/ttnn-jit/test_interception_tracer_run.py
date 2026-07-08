# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport
from utils import create_dram_tensor


@pytest.mark.forked
def test_interception_advisor_cross_function_with_weight(device):
    W = create_dram_tensor(device, (512, 512), torch.bfloat16)  # captured weight

    def _linear(a, b):
        return ttnn.matmul(a, b)  # in a helper -> cross-fn interception

    def mlp(x):
        h = _linear(x, W)
        h = ttnn.relu(h)
        rows, cols = h.shape[0], h.shape[1]
        h = ttnn.reshape(h, shape=(rows, cols))
        return ttnn.add(h, h)

    advisor = ShardAdvisor(mlp, optimization_level=2, tracer="interception")
    x = create_dram_tensor(device, (256, 512), torch.bfloat16)
    report = advisor.run(x)

    assert isinstance(report, AdvisorReport)
    assert report.trace.function_name == "mlp"
    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text
