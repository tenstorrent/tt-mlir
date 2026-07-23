# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch unit tests for golden.metrics.get_relative_l2 (no device needed)."""
import math

import pytest
import torch

from golden.metrics import get_relative_l2


def test_rel_l2_zero_for_identical_tensors():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert get_relative_l2(x, x.clone()) == 0.0


def test_rel_l2_scale_is_caught():
    # The whole point of the metric: PCC is 1.0 under pure scaling, rel_l2 is not.
    golden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    device = 2.0 * golden
    # ||device - golden|| / ||golden|| == ||golden|| / ||golden|| == 1.0
    assert get_relative_l2(golden, device) == pytest.approx(1.0)


def test_rel_l2_small_for_close_tensors():
    golden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    device = golden + 1e-4
    assert get_relative_l2(golden, device) < 1e-2


def test_rel_l2_zero_golden_matching_is_zero():
    z = torch.zeros(8)
    assert get_relative_l2(z, z.clone()) == 0.0


def test_rel_l2_zero_golden_differing_is_inf():
    golden = torch.zeros(8)
    device = torch.ones(8)
    assert math.isinf(get_relative_l2(golden, device))


def test_rel_l2_nan_propagates():
    golden = torch.tensor([1.0, float("nan"), 3.0])
    device = torch.tensor([1.0, 2.0, 3.0])
    assert math.isnan(get_relative_l2(golden, device))


def test_rel_l2_empty_is_zero():
    assert get_relative_l2(torch.empty(0), torch.empty(0)) == 0.0


def test_rel_l2_bf16_upcast_does_not_underflow():
    # Tiny bf16 values: float64 norm must not collapse to zero.
    golden = torch.full((16,), 1e-3, dtype=torch.bfloat16)
    device = golden.clone()
    assert get_relative_l2(golden, device) == 0.0
    device2 = (golden.to(torch.float64) * 1.5).to(torch.bfloat16)
    assert get_relative_l2(golden, device2) == pytest.approx(0.5, abs=1e-2)
