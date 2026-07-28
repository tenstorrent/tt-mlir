# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

# The DiT adaLN fusion runs at the TTNN level (ttnn-fusing pass), folding the
# lowered matmul + multiply + add epilogue directly into a single
# ttnn.dit_matmul_addcmul_fused op. There is no TTIR-level op; the primitive
# ops stay device-agnostic until the TTNN backend opportunistically fuses them.


def check_op(mlir_file: str, op_name: str, dialect: str = "ttnn") -> bool:
    """Check if an op exists in the MLIR file."""
    op_pattern = f"{dialect}.{op_name}"
    with open(mlir_file, "r") as f:
        return any(op_pattern in line for line in f)


# =============================================================================
# out = residual + gate * matmul(x, w), fused into ttnn.dit_matmul_addcmul_fused.
#
# The gate is broadcast across the M (row) dimension: the underlying tt-metal
# kernel treats the gate (addcmul_input_tensor2) as a single row broadcast
# across M whenever it spans one 32-row tile, mirroring adaLN modulation where
# the gate is per-block. The gate here therefore has identical rows.
#
# Kept to a single on-device case on purpose: the lit test covers the matmul and
# linear rewrite variants structurally, and running two dit-fused programs in one
# process currently trips a mesh-shape assertion during the second compile.
# =============================================================================
@pytest.mark.parametrize("m,k,n", [(32, 128, 256)], ids=["32x128x256"])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_dit_matmul_addcmul_fusion(
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Numerically validate the fused adaLN gated-residual matmul epilogue."""
    x_shape = (m, k)
    w_shape = (k, n)
    gate_shape = (m, n)
    res_shape = (m, n)
    shapes = [x_shape, w_shape, gate_shape, res_shape]
    dtypes = [dtype] * 4

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def dit_matmul_addcmul(
            x: Operand,
            w: Operand,
            gate: Operand,
            residual: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            x_data = torch.randn(x_shape, dtype=dtype)
            w_data = torch.randn(w_shape, dtype=dtype)
            # Gate broadcasts across M (rows identical) -> matches the kernel's
            # single-row-broadcast semantics for a single-tile M.
            gate_data = torch.randn((1, n), dtype=dtype).expand(m, n).contiguous()
            res_data = torch.randn(res_shape, dtype=dtype)

            # Reference: residual + gate * (x @ w), computed in fp32 then cast.
            proj = x_data.float() @ w_data.float()
            golden_output = (res_data.float() + gate_data.float() * proj).to(dtype)

            proj_op = builder.matmul(x, w)
            gated = builder.multiply(proj_op, gate)
            result = builder.add(residual, gated)

            builder.set_goldens(
                {x: x_data, w: w_data, gate: gate_data, residual: res_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(
        output_path, "dit_matmul_addcmul_fused"
    ), "Gated-residual matmul epilogue should be fused"
