# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect
import pytest

from ttmlir.test_utils import compile_as_module
from ttmlir.ttir_builder import Operand, TTIRBuilder, Attribute, UnitAttr, TypeInfo
from ttmlir.dialects import ttir
from ttmlir.ir import *
from ttmlir.passes import GoldenTensor, CallbackTag, DataType
from ttrt.common.util import Binary
from ttmlir.compile_and_run import ttir_to_ttnn, ttnn_to_flatbuffer
from ttmlir.ir import Module
from ttrt.binary import get_module_tags


@pytest.mark.parametrize(
    "tags",
    [(True, True), (True, False), (False, False)],
)
def test_callback_tags():
    def test_simple_callback(in0: Operand, in1: Operand, builder: TTIRBuilder):
        result = builder.add(in0, in1)
        print("Builder first loc: ", builder.get_loc())
        builder.set_callback_kv(builder.get_loc(), tags)

    module, builder2 = compile_as_mlir_module(
        test_simple_callback, [(64, 128), (64, 128)]
    )
    buffer = ttnn_to_flatbuffer(module)
    print("Builder second loc: ", builder2.get_loc())

    for (
        op
    ) in (
        module
    ):  # I think you have to write/pybind something for this, even get all tags from module
        # runtime_tags = ttrt.runtime.get_op_tags(fb.handle, 0, opContext)
        runtime_tags = get_module_tags(fb.handle, builder2.get_loc())
        print("Runtime_tags: ", runtime_tags, type(runtime_tags))
        assert (
            runtime_tags == tags
        ), f"Callback tags are wrong, expected {tags}, got {runtime_tags}"


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run TTIR Builder Op tests")
    parser.add_argument(
        "--path",
        type=str,
        help="Optional output path for the flatbuffer. Creates path if supplied path doesn't exist",
    )
    args = parser.parse_args()

    if args.path and os.path.exists(args.path):
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        set_output_path(args.path)

    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )

    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
