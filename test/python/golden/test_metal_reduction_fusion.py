# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


_INPUT_SHAPE = (256, 256)
_REDUCE_DIM = [1]
_DTYPE = torch.bfloat16


def _exp_sum_atol(shape, dtype: torch.dtype) -> float:
    per_elem_tol = 0.02 if dtype == torch.bfloat16 else 0.0005
    return shape[0] * shape[1] * per_elem_tol


def _exp_mean_atol(shape, dim_arg, dtype: torch.dtype) -> float:
    reduction_size = math.prod(shape[d] for d in dim_arg)
    return _exp_sum_atol(shape, dtype) / reduction_size


def _build_exp_reduce(input_shape, dim_arg, dtype: torch.dtype, reduce_type: str):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def exp_reduce(in0: Operand, builder: TTIRBuilder) -> Operand:
            in_tensor = torch.randn(input_shape, dtype=dtype).clamp(-1.0, 1.0)
            builder.set_goldens(inputs={in0: in_tensor})
            after_exp = builder.exp(in0)
            if reduce_type == "sum":
                return builder.sum(after_exp, dim_arg=dim_arg, keep_dim=True)
            if reduce_type == "mean":
                return builder.mean(after_exp, dim_arg=dim_arg, keep_dim=True)
            raise ValueError(f"Unsupported reduce_type: {reduce_type}")

    return module


@pytest.mark.parametrize("target", ["ttmetal"])
def test_exp_reduce_sum_fusion(target: str, request, device):
    """Test single-core `exp -> sum` with eltwise->reduction fusion enabled."""
    compile_and_execute_ttir(
        _build_exp_reduce(_INPUT_SHAPE, _REDUCE_DIM, _DTYPE, "sum"),
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[
            "override-device-shape=1,1",
            "enable-eltwise-reduction-fusion=true",
        ],
        atol=_exp_sum_atol(_INPUT_SHAPE, _DTYPE),
        save_artifacts=True,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_exp_reduce_mean_fusion(target: str, request, device):
    """Test single-core `exp -> mean` with eltwise->reduction fusion enabled."""
    compile_and_execute_ttir(
        _build_exp_reduce(_INPUT_SHAPE, _REDUCE_DIM, _DTYPE, "mean"),
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[
            "override-device-shape=1,1",
            "enable-eltwise-reduction-fusion=true",
        ],
        atol=_exp_mean_atol(_INPUT_SHAPE, _REDUCE_DIM, _DTYPE),
        save_artifacts=True,
    )
