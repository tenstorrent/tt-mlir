# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch

from golden.metrics import compute_atol, compute_pcc, compute_rtol


# ---------------------------------------------------------------------------
# compute_pcc
# ---------------------------------------------------------------------------


class TestComputePcc:
    def test_identical_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert compute_pcc(x, x.clone()) == pytest.approx(1.0)

    def test_negated_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = -x
        assert compute_pcc(x, y) == pytest.approx(-1.0)

    def test_all_nan(self):
        x = torch.full((4,), float("nan"))
        y = torch.full((4,), float("nan"))
        assert compute_pcc(x, y) == pytest.approx(1.0)

    def test_all_inf_same_sign(self):
        x = torch.full((4,), float("inf"))
        y = torch.full((4,), float("inf"))
        assert compute_pcc(x, y) == pytest.approx(1.0)

    def test_all_inf_opposite_sign(self):
        x = torch.full((4,), float("inf"))
        y = torch.full((4,), float("-inf"))
        assert compute_pcc(x, y) == pytest.approx(0.0)

    def test_constant_tensors_equal(self):
        x = torch.full((5,), 3.14)
        y = torch.full((5,), 3.14)
        # Both constant -> variance = 0 -> fallback to equality check -> 1.0
        assert compute_pcc(x, y) == pytest.approx(1.0)

    def test_constant_tensors_unequal(self):
        x = torch.full((5,), 1.0)
        y = torch.full((5,), 2.0)
        assert compute_pcc(x, y) == pytest.approx(0.0)

    def test_single_element_equal(self):
        x = torch.tensor([42.0])
        y = torch.tensor([42.0])
        assert compute_pcc(x, y) == pytest.approx(1.0)

    def test_single_element_unequal(self):
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        assert compute_pcc(x, y) == pytest.approx(0.0)

    def test_shape_mismatch_returns_minus_one(self):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        # shapes can't be reconciled for these sizes
        result = compute_pcc(x, y)
        assert result == -1

    def test_known_positive_correlation(self):
        x = torch.arange(1.0, 11.0)
        y = 2.0 * x + 1.0
        assert compute_pcc(x, y) == pytest.approx(1.0)

    def test_known_diff_large_noise(self):
        torch.manual_seed(0)
        x = torch.arange(1.0, 101.0)
        y = torch.randn(100) * 100
        result = compute_pcc(x, y)
        assert result < 0.5  # low correlation expected

    def test_2d_tensors(self):
        x = torch.arange(1.0, 13.0).reshape(3, 4)
        y = x.clone()
        assert compute_pcc(x, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_atol
# ---------------------------------------------------------------------------


class TestComputeAtol:
    def test_identical_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert compute_atol(x, x.clone()) == pytest.approx(0.0)

    def test_known_diff(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.5, 2.0, 3.0])
        assert compute_atol(x, y) == pytest.approx(0.5)

    def test_negated_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = -x
        assert compute_atol(x, y) == pytest.approx(6.0)

    def test_all_nan(self):
        x = torch.full((4,), float("nan"))
        y = torch.full((4,), float("nan"))
        assert compute_atol(x, y) == pytest.approx(0.0)

    def test_all_inf_same(self):
        x = torch.full((4,), float("inf"))
        y = torch.full((4,), float("inf"))
        assert compute_atol(x, y) == pytest.approx(0.0)

    def test_all_inf_different(self):
        x = torch.full((4,), float("inf"))
        y = torch.full((4,), float("-inf"))
        assert compute_atol(x, y) == math.inf

    def test_single_element(self):
        x = torch.tensor([5.0])
        y = torch.tensor([3.0])
        assert compute_atol(x, y) == pytest.approx(2.0)

    def test_shape_mismatch_returns_minus_one(self):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        assert compute_atol(x, y) == -1

    def test_2d_tensors(self):
        x = torch.zeros(3, 4)
        y = torch.ones(3, 4)
        assert compute_atol(x, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_rtol
# ---------------------------------------------------------------------------


class TestComputeRtol:
    def test_identical_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert compute_rtol(x, x.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_known_diff(self):
        # rel_err = max(|x - y| / (|x| + 1e-8))
        x = torch.tensor([10.0])
        y = torch.tensor([11.0])
        expected = abs(10.0 - 11.0) / (10.0 + 1e-8)
        assert compute_rtol(x, y) == pytest.approx(expected, rel=1e-4)

    def test_all_nan(self):
        x = torch.full((4,), float("nan"))
        y = torch.full((4,), float("nan"))
        assert compute_rtol(x, y) == pytest.approx(0.0)

    def test_all_inf_same(self):
        x = torch.full((4,), float("inf"))
        y = torch.full((4,), float("inf"))
        assert compute_rtol(x, y) == pytest.approx(0.0)

    def test_all_inf_different(self):
        x = torch.full((4,), float("inf"))
        y = torch.full((4,), float("-inf"))
        assert compute_rtol(x, y) == math.inf

    def test_shape_mismatch_returns_minus_one(self):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        assert compute_rtol(x, y) == -1

    def test_single_element(self):
        x = torch.tensor([100.0])
        y = torch.tensor([110.0])
        expected = abs(100.0 - 110.0) / (100.0 + 1e-8)
        assert compute_rtol(x, y) == pytest.approx(expected, rel=1e-4)
