# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param("lofi", id="lofi"),
        pytest.param("hifi2", id="hifi2"),
        pytest.param("hifi4", id="hifi4"),
    ],
)
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [
        pytest.param("true", id="fp32_true"),
        pytest.param("false", id="fp32_false"),
    ],
)
@pytest.mark.parametrize(
    "optimizer_enabled",
    [
        pytest.param(
            "true",
            id="optimizer_true",
            marks=pytest.mark.skip(
                reason="Optimizer enabled: skip all tests until builder can run with optimizer."
            ),
        ),
        pytest.param("false", id="optimizer_false"),
    ],
)
def test_conv2d_compute_config(
    request, device, math_fidelity, fp32_dest_acc_en, optimizer_enabled
):
    """Test conv2d with compute kernel config overrides for math fidelity and fp32 accumulation"""

    batch_size = 1
    in_channels = 64
    out_channels = 128
    input_height = 32
    input_width = 32
    kernel_height = 3
    kernel_width = 3
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    shapes = [
        (batch_size, input_height, input_width, in_channels),
        (out_channels, in_channels, kernel_height, kernel_width),
        (1, 1, 1, out_channels),
    ]
    dtypes = [torch.bfloat16, torch.bfloat16, torch.bfloat16]

    def test_module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def conv2d(
            in0: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.conv2d(
                in0,
                weight,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                unit_attrs=unit_attrs,
            )

    # Build pipeline options with compute config overrides
    pipeline_options = [
        f"compute-cfg-math-fidelity={math_fidelity}",
        f"compute-cfg-fp32-dest-acc-en={fp32_dest_acc_en}",
        f"enable-optimizer={optimizer_enabled}",
    ]

    compile_and_execute_ttir(
        test_module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=pipeline_options,
        target="ttnn",
    )


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param("lofi", id="lofi"),
        pytest.param("hifi2", id="hifi2"),
        pytest.param("hifi4", id="hifi4"),
    ],
)
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [
        pytest.param("true", id="fp32_true"),
        pytest.param("false", id="fp32_false"),
    ],
)
def test_sum_compute_config(request, device, math_fidelity, fp32_dest_acc_en):
    """Test sum reduction with compute kernel config overrides"""

    shapes = [(4, 32, 32)]
    dtypes = [torch.bfloat16]

    def test_module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def sum_op(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.sum(in0, dim_arg=[-1], keep_dim=True, unit_attrs=unit_attrs)

    pipeline_options = [
        f"compute-cfg-math-fidelity={math_fidelity}",
        f"compute-cfg-fp32-dest-acc-en={fp32_dest_acc_en}",
    ]

    compile_and_execute_ttir(
        test_module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=pipeline_options,
        target="ttnn",
    )


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param("lofi", id="lofi"),
        pytest.param("hifi2", id="hifi2"),
        pytest.param("hifi4", id="hifi4"),
    ],
)
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [
        pytest.param("true", id="fp32_true"),
        pytest.param("false", id="fp32_false"),
    ],
)
def test_softmax_compute_config(request, device, math_fidelity, fp32_dest_acc_en):
    """Test softmax with compute kernel config overrides"""

    shapes = [(4, 32, 32)]
    dtypes = [torch.bfloat16]

    def test_module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def softmax_op(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.softmax(in0, dimension=-1, unit_attrs=unit_attrs)

    pipeline_options = [
        f"compute-cfg-math-fidelity={math_fidelity}",
        f"compute-cfg-fp32-dest-acc-en={fp32_dest_acc_en}",
    ]

    compile_and_execute_ttir(
        test_module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=pipeline_options,
        target="ttnn",
    )


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param("lofi", id="lofi"),
        pytest.param("hifi2", id="hifi2"),
        pytest.param("hifi4", id="hifi4"),
    ],
)
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [
        pytest.param("true", id="fp32_true"),
        pytest.param("false", id="fp32_false"),
    ],
)
def test_matmul_compute_config(request, device, math_fidelity, fp32_dest_acc_en):
    """Test matmul with compute kernel config overrides"""

    shapes = [(32, 64), (64, 128)]
    dtypes = [torch.bfloat16, torch.bfloat16]

    def test_module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def matmul_op(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    pipeline_options = [
        f"compute-cfg-math-fidelity={math_fidelity}",
        f"compute-cfg-fp32-dest-acc-en={fp32_dest_acc_en}",
    ]

    compile_and_execute_ttir(
        test_module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=pipeline_options,
        target="ttnn",
    )


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param("lofi", id="lofi"),
        pytest.param("hifi2", id="hifi2"),
        pytest.param("hifi4", id="hifi4"),
    ],
)
@pytest.mark.parametrize(
    "fp32_dest_acc_en",
    [
        pytest.param("true", id="fp32_true"),
        pytest.param("false", id="fp32_false"),
    ],
)
def test_rmsnorm_compute_config(request, device, math_fidelity, fp32_dest_acc_en):
    """Test RMSNorm with compute kernel config overrides"""

    shapes = [(4, 32, 128)]
    dtypes = [torch.bfloat16]

    def test_module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rmsnorm_op(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.rms_norm(in0, normalized_shape=[128], unit_attrs=unit_attrs)

    pipeline_options = [
        f"compute-cfg-math-fidelity={math_fidelity}",
        f"compute-cfg-fp32-dest-acc-en={fp32_dest_acc_en}",
    ]

    compile_and_execute_ttir(
        test_module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=pipeline_options,
        target="ttnn",
    )
