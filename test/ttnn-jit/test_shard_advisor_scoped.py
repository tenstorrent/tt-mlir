# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport


def _mk(dev, shape):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_default_pipeline_is_scoped():
    adv = ShardAdvisor(lambda x: x)
    assert adv.pipeline == "scoped"


@pytest.mark.forked
def test_scoped_ffn_reports_and_is_one_to_one(device):
    # Two chained matmuls: the traced graph has exactly 2 ttnn.matmul ops; the
    # scoped pipeline must not fuse/decompose them away.
    w1 = _mk(device, (1, 1, 128, 256))
    w2 = _mk(device, (1, 1, 256, 128))

    def ffn(x):
        h = ttnn.matmul(x, w1)
        return ttnn.matmul(h, w2)

    report = ShardAdvisor(ffn, optimization_level=2, tracer="interception").run(
        _mk(device, (1, 1, 128, 128))
    )
    assert isinstance(report, AdvisorReport)
    assert "=> FINAL:" in report.text
    # 1:1: exactly two matmuls in the final IR (no fusion, no extra decomposed ops).
    assert report.ttnn_mlir.count('"ttnn.matmul"') == 2
