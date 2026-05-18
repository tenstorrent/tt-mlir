# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""E2E test for the MLP (FF1/FF2/FF3) + residual-add pattern.

Pattern C of the activation-dtype-lowering pass. The three matmuls all lower
to bfp_bf8; the gate `multiply` carries `dtype = bfp_bf8`; the final residual
`add` carries `dtype = bf16`.

Scaffold — requires multi-device system + hardware validation.
"""
import pytest
import torch
from typing import List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.skip(
    reason="Scaffold — requires multi-device system + hardware validation."
)
@x86_only
@pytest.mark.parametrize(
    "shapes",
    [((32, 128), (128, 256), (256, 128), (32, 128))],
    ids=["mlp_32x128_256"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_mlp_residual_dtype_lowering(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """FF1, FF3 -> silu/multiply -> FF2 -> add(residual)."""
    x_shape, up_shape, down_shape, res_shape = shapes

    def module(builder: TTIRBuilder):
        @builder.func(
            [x_shape, up_shape, up_shape, down_shape, res_shape],
            [dtype] * 5,
        )
        def mlp(
            x: Operand,
            w_ff1: Operand,
            w_ff3: Operand,
            w_ff2: Operand,
            residual: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            ff1 = builder.matmul(x, w_ff1, unit_attrs=unit_attrs)
            ff1 = builder.reduce_scatter(ff1, cluster_axis=0, scatter_dim=1)
            ff1 = builder.all_gather(ff1, cluster_axis=0, all_gather_dim=1)
            silu = builder.silu(ff1, unit_attrs=unit_attrs)

            ff3 = builder.matmul(x, w_ff3, unit_attrs=unit_attrs)
            ff3 = builder.reduce_scatter(ff3, cluster_axis=0, scatter_dim=1)
            ff3 = builder.all_gather(ff3, cluster_axis=0, all_gather_dim=1)

            gate = builder.multiply(silu, ff3, unit_attrs=unit_attrs)

            ff2 = builder.matmul(gate, w_ff2, unit_attrs=unit_attrs)
            ff2 = builder.reduce_scatter(ff2, cluster_axis=1, scatter_dim=1)
            ff2 = builder.all_gather(ff2, cluster_axis=1, all_gather_dim=1)

            return builder.add(residual, ff2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        argument_types_string=(
            "mlp=input,parameter,parameter,parameter,input"
        ),
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["enable-activation-dtype-lowering=true"],
        pcc=0.98,
    )
