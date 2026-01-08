# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttnn_jit._src.tracing_compiler import TracingCompiler


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
    ir = compiler.compile()
    # Insert output layout conversion if memory_config is provided
    ir = insert_output_layout_conversion(ir, memory_config, debug)
    print_and_verify_ir(ir, "TracingCompiler (Tracing-based)", debug)
    return ir
