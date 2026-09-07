# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ResNet18 forward pass — end-to-end correctness via torch.compile."""

import pytest
import torch

from tt_kurbla.torch.testing import ExecutionMode, assert_close_cpu_vs_tt

torchvision = pytest.importorskip("torchvision")

_DTYPE = torch.bfloat16


def test_resnet18_compile() -> None:
    model = torchvision.models.resnet18(weights=None).to(_DTYPE).eval()
    x = torch.randn(1, 3, 224, 224, dtype=_DTYPE)
    assert_close_cpu_vs_tt(model, x, atol=0.1, rtol=0.1, mode=ExecutionMode.COMPILE)
