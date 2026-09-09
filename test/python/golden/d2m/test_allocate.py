# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs

from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder

# borrow currently constrained way to build matmul inputs:
from d2m.test_matmul import create_matmul_constrained_inputs as create_matmul_inputs

# borrow currently constrained way to build reduction inputs:
from d2m.test_reductions import (
    create_reductions_constrained_inputs as create_reduction_inputs,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("k", [4, 8])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_allocate_matmul(m: int, k: int, n: int, target: str, request, device):
    tile_size = 32
    lhs = (
        m * tile_size,
        k * tile_size,
    )
    rhs = (
        k * tile_size,
        n * tile_size,
    )

    options = [
        f"override-device-shape=1,1",
        f"num-stream-buffers=1",
        # request the allocator to attempt to minimize stream buffer sizes
        # and reblock streams accordingly:
        f"test-buffer-size-policy=min",
    ]

    compile_and_execute_ttir(
        create_matmul_inputs(lhs, rhs),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("m", [8])
@pytest.mark.parametrize("n", [8])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_allocate_max(
    m: int,
    n: int,
    dim_arg: int,
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    options = [
        # request the allocator to attempt to minimize stream buffer sizes
        # and reblock streams accordingly:
        f"test-buffer-size-policy=min",
    ]

    compile_and_execute_ttir(
        create_reduction_inputs(shape, "max", dim_arg, keep_dim, dtype),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "allow_l1_output_spilling", [True, False], ids=["enabled", "disabled"]
)
def test_allocate_spills_internal_generic_outputs(
    allow_l1_output_spilling: bool, target: str, request, device
):
    # Keep many intermediate generic outputs live so the allocator must spill
    # some of them using the real device grid and L1 capacity.
    shape = (1536, 1536)
    num_branches = 12
    dtype = torch.float32

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def live_fanout_reduce(in0: Operand, in1: Operand, builder: TTIRBuilder):
            lhs = builder.neg(in0)
            rhs = builder.abs(in1)
            vals = [lhs, rhs]

            cur = builder.add(lhs, rhs)
            vals.append(cur)
            for i in range(num_branches):
                if i % 4 == 0:
                    cur = builder.add(cur, lhs)
                elif i % 4 == 1:
                    cur = builder.add(cur, rhs)
                elif i % 4 == 2:
                    cur = builder.neg(cur)
                else:
                    cur = builder.abs(cur)
                vals.append(cur)

            acc = vals[0]
            for val in vals[1:]:
                acc = builder.add(acc, val)
            return builder.sigmoid(acc)

    options = [
        f"allow-l1-output-spilling={str(allow_l1_output_spilling).lower()}",
        "enable-elementwise-fusion=false",
    ]

    def run_test():
        compile_and_execute_ttir(
            module,
            target=target,
            device=device,
            custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
            **get_request_kwargs(request),
        )

    if allow_l1_output_spilling:
        run_test()
    else:
        with pytest.raises(Exception, match="exceeds memory capacity"):
            run_test()
