# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport
from _autoport.functional_decoder import _FunctionalMLP
from models.common.modules.lazy_weight import LazyWeight


def _lazy(dev, shape):
    return LazyWeight(
        source=torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=dev,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_interception_traces_functional_mlp(device):
    # Llama-3.1-8B MLP shapes: hidden=4096, intermediate=14336. Weights are
    # stored transposed: gate/up [1,1,4096,14336], down [1,1,14336,4096].
    mlp = _FunctionalMLP(
        gate=_lazy(device, (1, 1, 4096, 14336)),
        up=_lazy(device, (1, 1, 4096, 14336)),
        down=_lazy(device, (1, 1, 14336, 4096)),
        activation_dtype=ttnn.bfloat16,
    )

    def traced(x):
        return mlp.forward(x)

    advisor = ShardAdvisor(traced, optimization_level=2, tracer="interception")
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    report = advisor.run(x)

    assert isinstance(report, AdvisorReport)
    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text
    # The MLP has 3 linears -> at least 3 layout decisions in the report.
    assert report.trace.total_ops >= 3
