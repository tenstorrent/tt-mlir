# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from collections import OrderedDict

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_apis import experimental_build_stablehlo_module

pytestmark = pytest.mark.frontend("shlo")

from builder.base.builder_utils import (
    stablehlo_pipeline,
    stablehlo_to_ttir_pipeline,
)


def multiple_meshes(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.add(in0, in1)


@pytest.mark.parametrize(
    "test_fn",
    [
        multiple_meshes,
    ],
)
def test_binary_ops(test_fn: Callable, request, device):
    module, shlo_builder = experimental_build_stablehlo_module(
        test_fn,
        [(128, 128), (128, 128)],
        [torch.float32, torch.float32],
        mesh_name=["mesh0", "mesh1"],
        mesh_dict=[
            OrderedDict([("x", 2), ("y", 2)]),
            OrderedDict([("x", 2), ("y", 2)]),
        ],
    )
