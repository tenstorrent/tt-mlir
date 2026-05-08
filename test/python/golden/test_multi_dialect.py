# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only, get_request_kwargs

from builder.base.multi_dialect_builder import MultiDialectBuilder
from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.base.builder_apis import compile_and_execute_multi, build_module
from builder.base.builder_enums import *
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    SkipIf,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn" | SkipIf("sim")])
def test_multi(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: MultiDialectBuilder):
        @builder.func([shape, shape], [torch.float32, torch.float32])
        def div(
            in0: Operand,
            in1: Operand,
            builder: MultiDialectBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            add = builder.add(in0, in1)
            cos = builder.ttnn.cos(add)
            return builder.div(cos, in1)

    compile_and_execute_multi(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
