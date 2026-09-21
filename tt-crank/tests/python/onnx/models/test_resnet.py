# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Whole-model parity on the TT EP at default options: torchvision resnet18 with random
weights, exported f32 with a dynamic batch pinned via free_dims."""

import numpy as np
import pytest
import torch
import torchvision

import tt_crank.onnx as tt_onnx

BATCH = 4
_DUMMY = torch.randn(1, 3, 224, 224)


def _assert_matches_cpu(model: bytes, batch: int, min_pcc: float) -> None:
    inputs = {"x": tt_onnx.randn(batch, 3, 224, 224)}
    session = tt_onnx.session(model, free_dims={"N": batch})
    (got,) = tt_onnx.run(session, inputs)
    (want,) = tt_onnx.cpu_golden(model, inputs)
    pcc = tt_onnx.pcc(got, want)
    assert pcc > min_pcc, f"logit PCC too low: {pcc}"
    return got, want


@pytest.fixture(scope="module")
def resnet18() -> bytes:
    torch.manual_seed(0)
    return tt_onnx.export_torch(
        torchvision.models.resnet18(weights=None), _DUMMY, dynamic_batch=True
    )


def test_resnet18_matches_cpu(resnet18: bytes) -> None:
    got, want = _assert_matches_cpu(resnet18, BATCH, min_pcc=0.998)
    np.testing.assert_array_equal(got.argmax(axis=1), want.argmax(axis=1))
