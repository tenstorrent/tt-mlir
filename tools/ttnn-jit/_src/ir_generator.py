# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttnn_jit._src.tracing_compiler import TracingCompiler
from ttnn_jit._src.tensor_translator import (
    _create_dram_tensor_layout,
    _create_sharded_tensor_layout,
    _calculate_tile_shape,
)
from ttmlir.ir import InsertionPoint, RankedTensorType, Location
from ttmlir.dialects import ttir, func
import ttnn


def print_and_verify_ir(ir, method_name, debug):
    if debug:
        print("---- IR Dump after " + method_name + " ----")
        print(ir)
    ir.operation.verify()


def create_output_layout_from_memory_config(
    ctx, memory_config, tensor_shape, element_type, debug
):

def generate_ir(f, debug, memory_config, *args, **kwargs):
    """Generate IR from tracing compilation."""
    compiler = TracingCompiler(f, *args, memory_config=memory_config, **kwargs)
    ir, output_type = compiler.compile()
    print_and_verify_ir(ir, "TracingCompiler (Tracing-based)", debug)
    return ir, output_type
