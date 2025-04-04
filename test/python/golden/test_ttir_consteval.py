# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder


@compile_to_flatbuffer(
    [
        (32, 32),
        (32, 32),
        (32, 32),
        (32, 32),
    ],
    targets=["ttnn"],
    argument_types_string="test_simple=input,parameter,parameter,constant",
)
def test_simple(
    in0: Operand,  # input
    in1: Operand,  # param
    in2: Operand,  # param
    in3: Operand,  # const
    builder: TTIRBuilder,
):
    add1 = builder.add(in0, in1)
    add3 = builder.add(in1, in2)
    add5 = builder.add(in2, in3)
    sub7 = builder.subtract(add3, add5)
    mul9 = builder.multiply(add1, sub7)
    return mul9


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run TTIR Builder Model tests")
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
