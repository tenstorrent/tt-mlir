# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only, get_request_kwargs

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn, build_module
from builder.base.builder_enums import *
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize("target", ["ttnn"])
def test_debug_annotate_op(target: str, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def debug_annotate(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
        ):
            add0 = builder.add(in0, in1)
            annotate0 = builder.annotate(add0, "This is from KV cache")
            sigmoid0 = builder.sigmoid(annotate0)
            return sigmoid0

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_debug_breakpoint_op(target: str, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def debug_breakpoint(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
        ):
            add0 = builder.add(in0, in1)
            builder.breakpoint(add0)
            sigmoid0 = builder.sigmoid(add0)
            return sigmoid0

    build_module(module, "ttnn")


@pytest.mark.parametrize("target", ["ttnn"])
def test_debug_print_op(target: str, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def debug_print(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
        ):
            add0 = builder.add(in0, in1)
            builder.print(add0, "execution has reached here")
            sigmoid0 = builder.sigmoid(add0)
            return sigmoid0

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_debug_memory_snapshot_op(target: str, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def debug_memory_snapshot(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
        ):
            add0 = builder.add(in0, in1)
            builder.memory_snapshot(add0, "/tmp/memory_dump.txt")
            sigmoid0 = builder.sigmoid(add0)
            return sigmoid0

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_debug_dump_op(target: str, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def debug_dump(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
        ):
            add0 = builder.add(in0, in1)
            builder.dump_tensor(add0, "/tmp/add0_dump.tensorbin")
            sigmoid0 = builder.sigmoid(add0)
            return sigmoid0

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_debug_region_ops(target: str, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def debug_regions(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
        ):
            add0 = builder.add(in0, in1)
            region_start0 = builder.region_start(add0, "embedding_region")
            sigmoid0 = builder.sigmoid(region_start0)
            region_end0 = builder.region_end(sigmoid0, "embedding_region")
            return region_end0

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
