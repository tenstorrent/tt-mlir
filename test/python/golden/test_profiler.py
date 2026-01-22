# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import os
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only
from profiler import *

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir, build_module
from builder.base.builder_enums import *
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
os.environ["TRACY_ON_DEMAND"] = "1"


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttnn"])
def test_matmul(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    pipeline_options = []
    # with trace("z_artifacts", 8086):
    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )

    device.read_device_profiler_results()


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_matmul2(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    pipeline_options = []
    # with trace("z1_artifacts", 8086):
    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )

    device.read_device_profiler_results()
