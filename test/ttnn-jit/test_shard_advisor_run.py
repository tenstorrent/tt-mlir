# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
import pytest

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport
from utils import create_dram_tensor


def _matmul(a, b):
    return ttnn.matmul(a, b)


@pytest.mark.forked
def test_shard_advisor_run_produces_report(device):
    a = create_dram_tensor(device, (256, 256), torch.bfloat16)
    b = create_dram_tensor(device, (256, 256), torch.bfloat16)

    advisor = ShardAdvisor(_matmul, optimization_level=2)
    report = advisor.run(a, b)

    assert isinstance(report, AdvisorReport)
    assert report.trace.function_name == "_matmul"
    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text
    # optimization_level=2 runs the spill pass.
    assert report.trace.spill.ran is True
